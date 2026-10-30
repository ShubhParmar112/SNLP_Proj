"""Central configuration. Every script imports from here so that the label sets,
the random seed and the split proportions cannot drift between experiments."""

import os
from pathlib import Path

RANDOM_SEED = 42

# v1.0: EduRABSA mapped by the Week 1-2 rules.
# v1.1: lectures rule (build_dataset.py, Rule 2b) - lecture comments that EduRABSA files
#       under "Course - Learning activity" now count as teaching, as Guidelines 2.1 says.
DATASET_VERSION = "v1.1-edurabsa"
DATASET_TAG = "d1.1"          # suffix on model checkpoints trained on this version

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FIXTURES = DATA / "fixtures"
RESULTS = ROOT / "results"

# Trained models (738 MB each) are kept OUTSIDE the repository. Where to find them:
#   1. $SNLP_ARTIFACTS, if set;
#   2. else the repo's own models/ folder, if it exists (unzip the model there);
#   3. else Krish's original location, D:/snlp-feedback/checkpoints.
def _checkpoint_root():
    if os.environ.get("SNLP_ARTIFACTS"):
        return Path(os.environ["SNLP_ARTIFACTS"])
    if (ROOT / "models").is_dir():
        return ROOT / "models"
    return Path("D:/snlp-feedback/checkpoints")


CHECKPOINT_ROOT = _checkpoint_root()
ASSET_MODEL_NAME = "deberta-v3-base_sw0.5_s42_d1.1"     # the model the API serves

for _d in (RAW, PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Label sets (see docs/Annotation_Guidelines.md §2) -----------------------

ASPECTS = [
    "teaching",
    "faculty_interaction",
    "course_content",
    "pace",
    "assessment",
    "workload",
    "resources",
    "other",
]

POLARITIES = ["positive", "negative", "neutral", "conflict"]
DOC_SENTIMENTS = ["positive", "neutral", "negative"]

# --- Splitting ---------------------------------------------------------------

SPLIT_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
NEAR_DUPLICATE_THRESHOLD = 0.95

# --- Evaluation --------------------------------------------------------------

N_BOOTSTRAP = 1000
CV_FOLDS = 5
LOW_SUPPORT_THRESHOLD = 20  # categories below this are flagged, not averaged
IAA_TARGET_KAPPA = 0.60
