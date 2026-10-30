"""Train the Week 3 transformer ABSA model (src/transformer_absa.py).

    python src/train_transformer.py                       # DeBERTa-v3-base, seed 42
    python src/train_transformer.py --seed 43 --span-weight 0

Test-set discipline (Evaluation_Protocol.md section 4.2): this script reads the train and
validation splits only. Model selection (best epoch) uses validation tuple-F1. The test
split is not loaded, so it cannot leak into any decision made here.

Weights are large (~740 MB) and are written outside the synced project folder, to
$SNLP_ARTIFACTS (default D:/snlp-feedback/checkpoints). Metrics and validation
predictions go to results/ inside the repo.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import (ASPECTS, CHECKPOINT_ROOT, DATASET_TAG, DATASET_VERSION,
                    DOC_SENTIMENTS, PROCESSED, RAW, RESULTS)
from data import to_model_format, validate
from metrics import (absa_tuple_f1, aspect_detection_metrics,
                     polarity_accuracy_on_detected, sentiment_metrics)
from transformer_absa import (DOC_INDEX, SLOT_LABELS, AspectAttentionABSA, Encoder,
                              batches, collate, decode_slots, loss_fn, predict)

ARTIFACTS = CHECKPOINT_ROOT      # see config.py: $SNLP_ARTIFACTS, ./models, or D:
MODEL_VERSION = "transformer-absa-0.2.0"     # 0.1.x = dataset v1.0, 0.2.x = v1.1


def load_split(path, names=("train", "validation")):
    """Comments, gold tuples, doc labels and evidence spans for the requested splits."""
    df_long, problems = validate(path, strict=True)
    if problems:
        raise SystemExit("annotation file failed validation")
    df = to_model_format(df_long)
    spans = {}
    for fid, g in df_long[df_long["aspect_category"] != ""].groupby("feedback_id"):
        spans[fid] = {r["aspect_category"]: r["aspect_term"] for _, r in g.iterrows()
                      if r["aspect_term"]}
    split_ids = json.loads((PROCESSED / "splits.json").read_text())
    by_id = df.set_index("feedback_id")
    out = {}
    for name in names:
        ids = split_ids[name]
        sub = by_id.loc[ids]
        out[name] = {"ids": ids,
                     "texts": sub["feedback_text"].tolist(),
                     "tuples": sub["tuples"].tolist(),
                     "doc": sub["doc_sentiment"].tolist(),
                     "spans": [spans.get(i, {}) for i in ids]}
    return out


def evaluate(preds, gold_tuples, gold_doc):
    pred_sets = [decode_slots(p["slot_probs"].argmax(-1)) for p in preds]
    t = absa_tuple_f1(gold_tuples, pred_sets)
    pol = polarity_accuracy_on_detected(gold_tuples, pred_sets)
    Yg = np.array([[int(any(a == x for x, _ in s)) for a in ASPECTS] for s in gold_tuples])
    Yp = np.array([[int(any(a == x for x, _ in s)) for a in ASPECTS] for s in pred_sets])
    asp = aspect_detection_metrics(Yg, Yp, ASPECTS)
    doc_pred = [DOC_SENTIMENTS[int(p["doc_probs"].argmax())] for p in preds]
    doc = sentiment_metrics(gold_doc, doc_pred, DOC_SENTIMENTS)
    return {"tuple_f1": t["f1"], "tuple_p": t["precision"], "tuple_r": t["recall"],
            "polarity_acc_on_detected": pol["accuracy"], "n_detected": pol["n_detected"],
            "aspect_micro_f1": asp["micro_f1"], "aspect_macro_f1": asp["macro_f1_all"],
            "doc_accuracy": doc["accuracy"], "doc_macro_f1": doc["macro_f1"]}, pred_sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW / "annotations.csv"))
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--span-weight", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-embeddings", type=int, default=1,
                    help="freeze the 128k-token embedding table (98M of DeBERTa-v3-base's "
                         "184M parameters) so fp32 weights + AdamW state fit in 4 GB")
    ap.add_argument("--token-budget", type=int, default=1024)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    name = a.name or (f"{a.encoder.split('/')[-1]}_sw{a.span_weight:g}_s{a.seed}"
                      f"_{DATASET_TAG}")
    out_dir = ARTIFACTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_split(a.data)
    enc = Encoder(a.encoder, a.max_length)
    tr = enc(data["train"]["texts"], data["train"]["tuples"], data["train"]["spans"])
    va = enc(data["validation"]["texts"], data["validation"]["tuples"])
    for it, d in zip(tr, data["train"]["doc"]):
        it["doc"] = DOC_INDEX[d]
    for it, d in zip(va, data["validation"]["doc"]):
        it["doc"] = DOC_INDEX[d]
    pad_id = enc.tok.pad_token_id
    n_span = sum(len(it["span_masks"]) for it in tr)
    n_pairs = sum(len(t) for t in data["train"]["tuples"])
    print(f"train {len(tr)}  validation {len(va)}  device {device}")
    print(f"evidence spans located for {n_span}/{n_pairs} training (aspect, polarity) pairs")

    model = AspectAttentionABSA(a.encoder).to(device)
    if a.freeze_embeddings:
        for p in model.encoder.embeddings.parameters():
            p.requires_grad = False
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_all / 1e6:.0f}M total, {n_train / 1e6:.0f}M trainable")
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    steps_per_epoch = sum(1 for _ in batches(tr, a.batch, False, token_budget=a.token_budget))
    total = steps_per_epoch * a.epochs
    warm = int(a.warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else max(0.0, (total - s) / max(1, total - warm)))

    history, best, best_state = [], -1.0, None
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    for epoch in range(1, a.epochs + 1):
        model.train()
        parts, te = {"slot": [], "doc": [], "span": []}, time.time()
        for idx in batches(tr, a.batch, shuffle=True, rng=rng, token_budget=a.token_budget):
            b = {k: v.to(device) for k, v in collate([tr[i] for i in idx], pad_id).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                slot_logits, doc_logits, alpha = model(b["input_ids"], b["attention_mask"])
            loss, p = loss_fn(slot_logits, doc_logits, alpha, b, a.span_weight)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {epoch}: {p}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k in parts:
                parts[k].append(p[k])
        preds = predict(model, va, pad_id, device)
        m, _ = evaluate(preds, data["validation"]["tuples"], data["validation"]["doc"])
        row = {"epoch": epoch, "train_loss": {k: float(np.mean(v)) for k, v in parts.items()},
               "val": m, "epoch_seconds": round(time.time() - te, 1)}
        history.append(row)
        print(f"epoch {epoch}: loss slot={row['train_loss']['slot']:.3f} "
              f"doc={row['train_loss']['doc']:.3f} span={row['train_loss']['span']:.3f} | "
              f"val tuple-F1={m['tuple_f1']:.3f} aspect-F1={m['aspect_micro_f1']:.3f} "
              f"doc-macroF1={m['doc_macro_f1']:.3f} ({row['epoch_seconds']}s)", flush=True)
        if m["tuple_f1"] > best:
            best = m["tuple_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    preds = predict(model, va, pad_id, device)
    m, pred_sets = evaluate(preds, data["validation"]["tuples"], data["validation"]["doc"])
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0

    # --- save weights (outside the repo) and metrics (inside it) ---------------
    torch.save(best_state, out_dir / "model.pt")
    model.encoder.config.save_pretrained(out_dir / "encoder_config")
    enc.tok.save_pretrained(out_dir / "tokenizer")
    card = {"model_version": MODEL_VERSION, "name": name, "encoder": a.encoder,
            "max_length": a.max_length, "aspects": ASPECTS, "slot_labels": SLOT_LABELS,
            "doc_labels": DOC_SENTIMENTS, "hyperparameters": vars(a),
            "best_epoch": best_epoch, "validation": m, "trained": datetime.now().isoformat(
                timespec="seconds"), "train_size": len(tr), "validation_size": len(va),
            "dataset_version": DATASET_VERSION, "test_split_used": False}
    (out_dir / "card.json").write_text(json.dumps(card, indent=2))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"week3_train_{name}.json").write_text(json.dumps(
        {**card, "history": history, "train_minutes": round((time.time() - t0) / 60, 1),
         "peak_gpu_mb": round(peak_mb), "checkpoint_dir": str(out_dir)}, indent=2))
    (RESULTS / f"week3_val_predictions_{name}.json").write_text(json.dumps(
        [{"feedback_id": i, "gold": sorted(f"{x}:{y}" for x, y in g),
          "pred": sorted(f"{x}:{y}" for x, y in p),
          "slot_probs": np.round(pr["slot_probs"], 4).tolist(),
          "doc_probs": np.round(pr["doc_probs"], 4).tolist()}
         for i, g, p, pr in zip(data["validation"]["ids"], data["validation"]["tuples"],
                                pred_sets, preds)]))
    print(f"\nbest epoch {best_epoch}: val tuple-F1={m['tuple_f1']:.3f} "
          f"P={m['tuple_p']:.3f} R={m['tuple_r']:.3f} | peak GPU {peak_mb:.0f} MB | "
          f"{(time.time() - t0) / 60:.1f} min")
    print(f"weights -> {out_dir}")


if __name__ == "__main__":
    main()
