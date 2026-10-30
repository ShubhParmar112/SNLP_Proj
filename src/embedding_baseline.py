"""Week 4 comparison model: frozen sentence embeddings + logistic regression.

The handout's first example of an "improved approach" is TF-IDF -> Sentence Transformer.
This model is exactly that, and it sits between the two systems we already have:

    TF-IDF + LogReg            words counted, no pretraining
    MiniLM embeddings + LogReg pretrained sentence meaning, but frozen and one vector
                               per comment
    DeBERTa ABSA (Week 3)      pretrained AND fine-tuned, one attention per aspect

So it answers a question the Week 3 result alone cannot: is the gain from pretrained
representations as such, or from fine-tuning with aspect-specific attention?

It uses the same output formulation as the transformer - for each of the 8 aspects a
5-way decision {none, positive, negative, neutral, conflict} - so the two differ only in
the representation, not in the task framing. One vector per comment is the key
limitation being tested: a mixed review has to be squeezed into a single point.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from config import ASPECTS, DOC_SENTIMENTS, RANDOM_SEED
from preprocess import preprocess_transformer
from transformer_absa import SLOT_LABELS, decide_slots, decode_slots, encode_targets

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@torch.no_grad()
def embed(texts, model_name=MODEL_NAME, device=None, batch_size=64, max_length=256):
    """Mean-pooled, L2-normalised sentence embeddings (the model card's own recipe)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    proc = [preprocess_transformer(t) for t in texts]
    out = []
    for i in range(0, len(proc), batch_size):
        enc = tok(proc[i:i + batch_size], padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        H = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        e = (H * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        out.append(F.normalize(e, dim=-1).cpu().numpy())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(out)


class EmbeddingABSA:
    """Eight 5-way logistic regressions (one per aspect) + one for document sentiment."""

    def __init__(self, class_weight=None, C=1.0):
        self.class_weight, self.C = class_weight, C

    def _lr(self, class_weight):
        return LogisticRegression(max_iter=3000, C=self.C, class_weight=class_weight,
                                  random_state=RANDOM_SEED)

    def fit(self, X, tuples, doc):
        Y = np.stack([encode_targets(t) for t in tuples])
        self.slot_clfs = [self._lr(self.class_weight).fit(X, Y[:, a])
                          for a in range(len(ASPECTS))]
        self.doc_clf = self._lr("balanced").fit(X, doc)
        return self

    def slot_probs(self, X):
        """(n, 8, 5) probabilities; slots unseen in training for an aspect get 0."""
        P = np.zeros((X.shape[0], len(ASPECTS), len(SLOT_LABELS)))
        for a, clf in enumerate(self.slot_clfs):
            P[:, a, clf.classes_] = clf.predict_proba(X)
        return P

    def predict_sets(self, X, thresholds=None):
        return [decode_slots(decide_slots(p, thresholds)) for p in self.slot_probs(X)]

    def doc_predict(self, X):
        return list(self.doc_clf.predict(X))
