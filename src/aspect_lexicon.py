"""Aspect lexicon for the transparent Baseline B.

Two guards against circular evaluation (docs/Evaluation_Protocol.md §5):

1. build_from_training() only ever sees the training split. A lexicon tuned by looking
   at validation or test comments measures nothing.
2. Every term records where it came from, written to docs/lexicon_provenance.md, so a
   reader can check the lexicon was not reverse-engineered from the gold labels.

The seed lexicon below is deliberately small and general. The data-driven step expands
it from training-split statistics, which is the part that can actually be defended.
"""

import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ASPECTS

# Manually written from domain knowledge BEFORE seeing any annotation.
# Keep this list short — it is a starting point, not the finished lexicon.
SEED_LEXICON = {
    "teaching": ["lecture", "teaching", "explain", "explanation", "clear", "clarity",
                 "example", "concept", "taught", "teach", "understandable"],
    "faculty_interaction": ["doubt", "available", "approachable", "responsive",
                            "helpful", "office hour", "reply", "feedback", "support"],
    "course_content": ["content", "syllabus", "topic", "material", "relevant",
                       "relevance", "curriculum", "depth", "module", "coverage"],
    "pace": ["pace", "fast", "slow", "rushed", "rush", "hurry", "quick", "speed",
             "time to cover", "moved on"],
    "assessment": ["assignment", "exam", "quiz", "test", "project", "grading",
                   "marks", "grade", "question paper", "evaluation", "viva"],
    "workload": ["workload", "deadline", "too much", "heavy", "burden", "submission",
                 "pressure", "overload", "back to back", "time to finish"],
    "resources": ["slide", "notes", "recording", "lms", "portal", "textbook", "book",
                  "lab", "equipment", "software", "classroom", "projector", "resource"],
}


class AspectLexicon:
    def __init__(self, lexicon=None):
        self.lexicon = {a: list(v) for a, v in (lexicon or SEED_LEXICON).items()}
        self.provenance = {
            a: {t: "seed (domain knowledge, pre-annotation)" for t in v}
            for a, v in self.lexicon.items()
        }

    # --- construction --------------------------------------------------------

    def build_from_training(self, train_texts, train_aspect_sets,
                            top_k=8, min_count=3):
        """Expand the seed lexicon using training-split statistics only.

        For each aspect, score candidate terms by how much more often they appear in
        comments carrying that aspect than in the corpus overall (a simple
        discriminative ratio), and keep the top_k that clear min_count.
        """
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_count,
                              sublinear_tf=True, stop_words="english")
        try:
            X = vec.fit_transform(train_texts)
        except ValueError:
            return self  # corpus too small to expand; seed lexicon stands
        vocab = np.array(vec.get_feature_names_out())

        overall = np.asarray(X.mean(axis=0)).ravel()
        doc_counts = Counter()
        for s in train_aspect_sets:
            doc_counts.update(s)

        for aspect in ASPECTS:
            if aspect == "other":
                continue
            rows = [i for i, s in enumerate(train_aspect_sets) if aspect in s]
            if len(rows) < min_count:
                continue
            in_aspect = np.asarray(X[rows].mean(axis=0)).ravel()
            ratio = in_aspect / (overall + 1e-9)
            for idx in np.argsort(-ratio)[: top_k * 4]:
                term = str(vocab[idx])
                if ratio[idx] < 2.0 or in_aspect[idx] <= 0:
                    continue
                if term in self.lexicon.setdefault(aspect, []):
                    continue
                if len([t for t in self.lexicon[aspect]
                        if self.provenance[aspect].get(t, "").startswith("derived")]) >= top_k:
                    break
                self.lexicon[aspect].append(term)
                self.provenance.setdefault(aspect, {})[term] = (
                    f"derived from training split: appears {ratio[idx]:.1f}x more in "
                    f"'{aspect}' comments (n={len(rows)})"
                )
        return self

    # --- inference -----------------------------------------------------------

    def extract(self, text):
        """Return {aspect: match_strength} for aspects whose terms appear in text.

        match_strength is the count of distinct matched terms — a transparent tally,
        NOT a probability. The original spec reported values like 0.91 as
        'confidence' for lexicon output, which is meaningless: a keyword match has no
        probability attached to it. See API_Specification.md on score_type.
        """
        low = str(text).lower()
        hits = defaultdict(int)
        for aspect, terms in self.lexicon.items():
            for t in terms:
                if t in low:
                    hits[aspect] += 1
        return dict(hits)

    def predict_sets(self, texts):
        out = []
        for t in texts:
            hits = self.extract(t)
            out.append(set(hits) if hits else set())
        return out

    # --- persistence ---------------------------------------------------------

    def save(self, path, provenance_path=None):
        with open(path, "w") as f:
            json.dump(self.lexicon, f, indent=2, sort_keys=True)
        if provenance_path:
            lines = ["# Aspect Lexicon Provenance", "",
                     "Every term and where it came from. Terms marked *derived* were",
                     "computed from the **training split only** — no validation or test",
                     "comment was inspected while building this lexicon.", ""]
            for aspect in sorted(self.provenance):
                lines.append(f"## `{aspect}`", )
                lines.append("")
                lines.append("| Term | Source |")
                lines.append("|---|---|")
                for term, src in sorted(self.provenance[aspect].items()):
                    lines.append(f"| `{term}` | {src} |")
                lines.append("")
            with open(provenance_path, "w") as f:
                f.write("\n".join(lines))
        return self

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
