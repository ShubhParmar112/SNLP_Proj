"""Week 3 core model: transformer ABSA with aspect-specific attention.

One pass over the comment produces, for every one of the 8 aspect categories, a 5-way
decision {none, positive, negative, neutral, conflict}, plus the document sentiment.
That is joint aspect-category sentiment analysis: detection and polarity come out of the
same decision, so the model is trained directly on the tuple the project is scored on.

    comment --> DeBERTa-v3 encoder --> token states H (T x d)
                  |
                  +-- [CLS] ------------------------------> document sentiment (3-way)
                  |
                  +-- 8 learned aspect queries attend over H
                        alpha_a = softmax(q_a . W_k H)       (where does aspect a live?)
                        z_a     = sum_t alpha_a,t H_t
                        [z_a ; CLS] -> aspect a's own head -> {none, pos, neg, neu, conflict}

Why aspect attention rather than one [CLS] classifier: a comment like "The lectures were
informative but the pace was too fast" carries opposite polarities in different places.
A single pooled vector has to squeeze both into one representation; a per-aspect
attention lets `teaching` read "lectures were informative" and `pace` read "pace was too
fast". The attention weights also give the evidence span the API contract asks for.

Evidence spans are supervised, not just hoped for: where the gold data marks the words
behind an aspect, an auxiliary loss pushes that aspect's attention onto those tokens
(loss = -log sum of attention on the gold span). Weighted by `span_weight`; 0 turns it off.

Preprocessing is Branch B (src/preprocess.py): PII redaction and whitespace only. The
classical branch's lowercasing, stopword removal and NOT_ marking would destroy the
casing, function words and word pieces the pretrained encoder relies on.
"""

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import ASPECTS, DOC_SENTIMENTS
from preprocess import preprocess_transformer

SLOT_LABELS = ["none", "positive", "negative", "neutral", "conflict"]
SLOT_INDEX = {l: i for i, l in enumerate(SLOT_LABELS)}
DOC_INDEX = {l: i for i, l in enumerate(DOC_SENTIMENTS)}
ASPECT_INDEX = {a: i for i, a in enumerate(ASPECTS)}


# --- Model -------------------------------------------------------------------

class AspectAttentionABSA(nn.Module):
    def __init__(self, encoder_name=None, n_aspects=len(ASPECTS), n_slot=len(SLOT_LABELS),
                 n_doc=len(DOC_SENTIMENTS), dropout=0.1, encoder_config=None):
        super().__init__()
        # encoder_config: rebuild the architecture from a saved checkpoint without
        # downloading the pretrained weights again (they are overwritten on load anyway).
        # dtype=float32 is required: transformers 5 otherwise loads the checkpoint's own
        # dtype (fp16 for DeBERTa-v3), and AdamW on fp16 weights diverges to NaN within
        # the first epoch. Mixed precision is applied at run time by autocast instead.
        self.encoder = (AutoModel.from_config(encoder_config) if encoder_config is not None
                        else AutoModel.from_pretrained(encoder_name, dtype=torch.float32))
        self.encoder.float()
        d = self.encoder.config.hidden_size
        self.d = d
        self.aspect_queries = nn.Parameter(torch.randn(n_aspects, d) * 0.02)
        self.key = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        # One head per aspect: whether an aspect is present depends on which aspect it is.
        self.slot_w = nn.Parameter(torch.empty(n_aspects, 2 * d, n_slot))
        nn.init.xavier_uniform_(self.slot_w)
        self.slot_b = nn.Parameter(torch.zeros(n_aspects, n_slot))
        self.doc_head = nn.Linear(d, n_doc)

    def forward(self, input_ids, attention_mask):
        H = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state   # B,T,d
        H = H.float()
        K = self.key(H)
        scores = torch.einsum("btd,ad->bat", K, self.aspect_queries.float()) / math.sqrt(self.d)
        scores = scores.masked_fill(attention_mask[:, None, :] == 0, -1e9)
        alpha = scores.softmax(dim=-1)                                     # B,A,T
        pooled = torch.einsum("bat,btd->bad", alpha, H)                     # B,A,d
        cls = H[:, 0]
        feat = torch.cat([pooled, cls[:, None, :].expand_as(pooled)], dim=-1)
        slot_logits = torch.einsum("bak,akn->ban", self.dropout(feat),
                                   self.slot_w.float()) + self.slot_b.float()
        doc_logits = self.doc_head(self.dropout(cls))
        return slot_logits, doc_logits, alpha


# --- Encoding ----------------------------------------------------------------

def encode_targets(tuples):
    """Set of (aspect, polarity) -> length-8 vector of slot indices."""
    y = np.zeros(len(ASPECTS), dtype=np.int64)          # 0 = none
    for a, p in tuples:
        y[ASPECT_INDEX[a]] = SLOT_INDEX[p]
    return y


def decode_slots(slot_idx):
    """Length-8 vector of slot indices -> set of (aspect, polarity)."""
    return {(ASPECTS[i], SLOT_LABELS[int(s)]) for i, s in enumerate(slot_idx) if s != 0}


def decide_slots(slot_probs, thresholds=None):
    """8x5 slot probabilities -> one slot index per aspect.

    thresholds=None is the Week 3 v1 rule: plain argmax over the five slots.
    With thresholds (one per aspect, Week 4 v1.1), an aspect is present when
    P(present) = 1 - P(none) reaches its threshold, and its polarity is the most
    probable of the four polarity slots. Separating the two decisions lets the
    presence cut-off be tuned per aspect without touching the polarity choice.
    """
    p = np.asarray(slot_probs)
    if thresholds is None:
        return p.argmax(-1)
    t = np.array([thresholds[a] for a in ASPECTS])
    present = (1.0 - p[:, 0]) >= t
    return np.where(present, p[:, 1:].argmax(-1) + 1, 0)


def span_token_mask(offsets, start, end):
    """Tokens whose character range overlaps [start, end). Special tokens excluded."""
    return np.array([(e > s) and (s < end) and (e > start) for s, e in offsets], dtype=bool)


class Encoder:
    """Tokenises Branch-B text and locates gold evidence spans in token space."""

    def __init__(self, tokenizer_name, max_length=256):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __call__(self, texts, tuples=None, spans=None):
        """texts: raw comments. spans: per comment {aspect: evidence_text} (optional)."""
        proc = [preprocess_transformer(t) for t in texts]
        enc = self.tok(proc, truncation=True, max_length=self.max_length,
                       return_offsets_mapping=True)
        items = []
        for i, text in enumerate(proc):
            item = {"input_ids": enc["input_ids"][i],
                    "offsets": [tuple(o) for o in enc["offset_mapping"][i]],
                    "text": text,
                    "truncated": len(enc["input_ids"][i]) >= self.max_length}
            if tuples is not None:
                item["y"] = encode_targets(tuples[i])
            if spans is not None:
                masks = {}
                for aspect, ev in (spans[i] or {}).items():
                    ev = preprocess_transformer(ev) if ev else ""
                    j = text.find(ev) if ev else -1
                    if j < 0:
                        continue     # span lost to redaction or truncation: no supervision
                    m = span_token_mask(item["offsets"], j, j + len(ev))
                    if m.any():
                        masks[ASPECT_INDEX[aspect]] = m
                item["span_masks"] = masks
            items.append(item)
        return items


def collate(items, pad_id):
    T = max(len(it["input_ids"]) for it in items)
    B = len(items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    span = torch.zeros((B, len(ASPECTS), T), dtype=torch.bool)
    for b, it in enumerate(items):
        n = len(it["input_ids"])
        ids[b, :n] = torch.tensor(it["input_ids"])
        att[b, :n] = 1
        for a, m in it.get("span_masks", {}).items():
            span[b, a, :n] = torch.from_numpy(m[:n])
    out = {"input_ids": ids, "attention_mask": att, "span_mask": span}
    if "y" in items[0]:
        out["y"] = torch.from_numpy(np.stack([it["y"] for it in items]))
    if "doc" in items[0]:
        out["doc"] = torch.tensor([it["doc"] for it in items])
    return out


def batches(items, batch_size, shuffle, rng=None, token_budget=2048):
    """Length-bucketed batches: sorting by length cuts padding, and therefore GPU time,
    roughly in half on this corpus (median 46 tokens, max 256 after truncation).

    token_budget caps batch_size x longest_length, so the few very long comments come in
    smaller batches instead of overflowing a 4 GB laptop GPU."""
    lengths = [len(it["input_ids"]) for it in items]
    order = np.argsort(lengths, kind="stable")
    chunks, cur = [], []
    for i in order:
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * lengths[i] > token_budget):
            chunks.append(np.array(cur))
            cur = []
        cur.append(i)
    if cur:
        chunks.append(np.array(cur))
    if shuffle:
        rng.shuffle(chunks)
    for c in chunks:
        yield c


def loss_fn(slot_logits, doc_logits, alpha, batch, span_weight):
    slot_loss = F.cross_entropy(slot_logits.reshape(-1, slot_logits.size(-1)),
                                batch["y"].reshape(-1))
    doc_loss = F.cross_entropy(doc_logits, batch["doc"])
    has_span = batch["span_mask"].any(dim=-1)                          # B,A
    if span_weight > 0 and has_span.any():
        mass = (alpha * batch["span_mask"].float()).sum(-1)            # B,A
        span_loss = -torch.log(mass[has_span].clamp_min(1e-6)).mean()
    else:
        span_loss = torch.zeros((), device=slot_logits.device)
    return slot_loss + doc_loss + span_weight * span_loss, {
        "slot": slot_loss.item(), "doc": doc_loss.item(), "span": span_loss.item()}


# --- Inference helpers -------------------------------------------------------

# A clause ends at punctuation or at a contrast/coordination word: "the lectures were
# informative | but the pace was too fast | and the assignments were difficult".
CLAUSE_BREAK = re.compile(r"[.,;:!?()\[\]\n]|\b(?:but|however|although|though|whereas|"
                          r"while|and|yet)\b", re.I)


def clause_window(text, s, e, max_words):
    """Widen [s, e) to the clause that contains it, capped at max_words words.

    The cap keeps the words nearest the original span, so a long run-on sentence still
    yields a phrase rather than the whole comment.
    """
    left = 0
    for m in CLAUSE_BREAK.finditer(text, 0, s):
        left = m.end()
    m = CLAUSE_BREAK.search(text, e)
    right = m.start() if m else len(text)
    words = [w for w in re.finditer(r"\S+", text[left:right])]
    if not words:
        return s, e
    spans = [(left + w.start(), left + w.end()) for w in words]
    if max_words and len(spans) > max_words:
        centre = next((k for k, (ws, we) in enumerate(spans) if we > s), 0)
        lo = max(0, min(centre - max_words // 2, len(spans) - max_words))
        spans = spans[lo:lo + max_words]
    return min(s, spans[0][0]), max(e, spans[-1][1])


def evidence_span(alpha_row, offsets, text, rel=0.5, clause_max_words=None):
    """Evidence for one aspect, as a character span of the analysed text.

    Starts at the highest-attention token and grows left/right while neighbours keep at
    least `rel` of the peak weight (Week 3 rule). The supervised attention turned out to
    be almost one-hot, so that window is usually one word; with clause_max_words set
    (Week 4), the window is widened to its clause, capped at that many words.
    Returns {"text", "start", "end"}.
    """
    valid = [i for i, (s, e) in enumerate(offsets) if e > s]
    if not valid:
        return None
    a = np.asarray(alpha_row, dtype=float)
    peak = max(valid, key=lambda i: a[i])
    lo = hi = peak
    thr = rel * a[peak]
    while lo - 1 in valid and a[lo - 1] >= thr:
        lo -= 1
    while hi + 1 in valid and a[hi + 1] >= thr:
        hi += 1
    s, e = offsets[lo][0], offsets[hi][1]
    while s < e and text[s].isspace():
        s += 1
    if clause_max_words is not None:
        s, e = clause_window(text, s, e, clause_max_words)
    return {"text": text[s:e], "start": int(s), "end": int(e)}


@torch.no_grad()
def predict(model, items, pad_id, device, batch_size=32):
    """Returns per item: slot probabilities (8x5), doc probabilities (3), attention (8xT)."""
    model.eval()
    out = [None] * len(items)
    for idx in batches(items, batch_size, shuffle=False):
        b = collate([items[i] for i in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            slot_logits, doc_logits, alpha = model(b["input_ids"].to(device),
                                                   b["attention_mask"].to(device))
        sp = slot_logits.float().softmax(-1).cpu().numpy()
        dp = doc_logits.float().softmax(-1).cpu().numpy()
        al = alpha.float().cpu().numpy()
        for k, i in enumerate(idx):
            n = len(items[i]["input_ids"])
            out[i] = {"slot_probs": sp[k], "doc_probs": dp[k], "alpha": al[k, :, :n]}
    return out
