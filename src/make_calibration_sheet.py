"""Blank calibration sheet for the team's mapping check (Guidelines section 5, Round 0).

    python src/make_calibration_sheet.py

EduRABSA's labels come from one expert annotator and were mapped onto our taxonomy by
rules in build_dataset.py. Nobody on the team has yet checked that the mapped labels
match how *we* would label the same comments. This round does that.

40 comments are drawn from the TRAINING split only (seed 42), so validation and test
stay unseen. Each member fills aspect_category / polarity / aspect_term for every
comment without looking at the gold file, saving as
data/fixtures/calibration_<annotator_id>.csv. Then:

    python src/iaa.py <combined csv>        # member vs member, and member vs mapped gold
"""

import csv
import json
import random

from config import FIXTURES, PROCESSED, RANDOM_SEED, RAW
from data import to_model_format, validate


def main(n=40):
    df = to_model_format(validate(str(RAW / "annotations.csv"), strict=False)[0])
    train_ids = json.loads((PROCESSED / "splits.json").read_text())["train"]
    pick = random.Random(RANDOM_SEED).sample(sorted(train_ids), n)
    texts = df.set_index("feedback_id").loc[pick, "feedback_text"]
    out = FIXTURES / "calibration_sheet_blank.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feedback_id", "feedback_text", "annotator_id", "doc_sentiment",
                    "aspect_category", "polarity", "aspect_term", "span_start",
                    "span_end", "uncertain", "double_annotated", "notes"])
        for fid, text in texts.items():
            w.writerow([fid, text, "", "", "", "", "", -1, -1, 0, 1, ""])
    print(f"{n} training-split comments -> {out}")


if __name__ == "__main__":
    main()
