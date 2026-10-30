"""Baseline models.

Every model is an sklearn Pipeline. This is not stylistic: it makes the leakage rule
from docs/Evaluation_Protocol.md §4.2 structurally impossible to violate, because the
vectoriser can only ever be fitted inside .fit() on the training fold.

Three models:
    SentimentBaseline  — Task A, document-level sentiment
    AspectBaseline     — Task B, multi-label aspect detection
    ABSABaseline       — Task C, (aspect, polarity) tuple extraction
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from config import ASPECTS, RANDOM_SEED
from preprocess import preprocess_classical


def _vectoriser(use_char=True):
    """Word 1-2 grams plus character 3-5 grams.

    The character branch matters here: student feedback contains misspellings
    ('assignmnet'), inconsistent spacing and code-switching, all of which produce
    out-of-vocabulary word tokens but remain matchable at the character level.
    """
    parts = [("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, max_features=20000))]
    if use_char:  # switched off only for the word-vs-char ablation
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              min_df=2, sublinear_tf=True,
                                              max_features=30000)))
    return FeatureUnion(parts)


def _classifier(class_weight="balanced"):
    """class_weight='balanced' is the default deliberately.

    Student feedback is imbalanced, and an unweighted classifier on imbalanced data
    collapses onto the majority class while still reporting a respectable accuracy.
    Run the ablation in run_baseline.py to show the effect rather than assuming it.
    """
    return LogisticRegression(max_iter=2000, class_weight=class_weight,
                              random_state=RANDOM_SEED, C=1.0)


class SentimentBaseline:
    """Task A: TF-IDF -> Logistic Regression -> positive/neutral/negative."""

    def __init__(self, negation=True, class_weight="balanced"):
        self.negation = negation
        self.pipe = Pipeline([("tfidf", _vectoriser()),
                              ("clf", _classifier(class_weight))])

    def _prep(self, texts):
        return [preprocess_classical(t, negation=self.negation) for t in texts]

    def fit(self, texts, y):
        self.pipe.fit(self._prep(texts), y)
        return self

    def predict(self, texts):
        return self.pipe.predict(self._prep(texts))

    def predict_proba(self, texts):
        return self.pipe.predict_proba(self._prep(texts))


class AspectBaseline:
    """Task B: multi-label aspect detection, one-vs-rest logistic regression."""

    def __init__(self, negation=True, threshold=0.5, use_char=True):
        self.negation = negation
        self.threshold = threshold
        self.pipe = Pipeline([
            ("tfidf", _vectoriser(use_char)),
            ("clf", OneVsRestClassifier(_classifier())),
        ])

    def _prep(self, texts):
        return [preprocess_classical(t, negation=self.negation) for t in texts]

    def fit(self, texts, Y):
        self.Y_cols = [i for i in range(np.asarray(Y).shape[1])
                       if len(set(np.asarray(Y)[:, i])) > 1]
        self.pipe.fit(self._prep(texts), np.asarray(Y)[:, self.Y_cols])
        return self

    def predict(self, texts):
        proba = self.pipe.predict_proba(self._prep(texts))
        out = np.zeros((len(texts), len(ASPECTS)), dtype=int)
        out[:, self.Y_cols] = (proba >= self.threshold).astype(int)
        return out

    def predict_sets(self, texts):
        Y = self.predict(texts)
        return [{ASPECTS[i] for i in np.where(row)[0]} for row in Y]


class ABSABaseline:
    """Task C: the core model.

    Two stages:
      1. detect which aspect categories the comment discusses (AspectBaseline);
      2. for each detected aspect, classify its polarity with a classifier that sees
         both the text and the aspect identity.

    Stage 2 is trained on (comment, aspect) pairs, with the aspect injected as an
    explicit token ('__ASPECT_pace__'). That single shared model handles all aspects,
    which matters on a small corpus: eight separate per-aspect classifiers would each
    see too few examples to learn anything.
    """

    def __init__(self, negation=True, threshold=0.5, use_lexicon=False, lexicon=None,
                 use_char=True):
        self.negation = negation
        self.use_lexicon = use_lexicon
        self.lexicon = lexicon
        self.aspect_model = AspectBaseline(negation=negation, threshold=threshold,
                                           use_char=use_char)
        self.polarity_pipe = Pipeline([("tfidf", _vectoriser(use_char)),
                                       ("clf", _classifier())])

    def _prep(self, text):
        return preprocess_classical(text, negation=self.negation)

    def _pair(self, text, aspect):
        return f"__ASPECT_{aspect}__ {self._prep(text)}"

    def fit(self, texts, Y_aspects, tuple_sets):
        if not self.use_lexicon:
            self.aspect_model.fit(texts, Y_aspects)

        pair_X, pair_y = [], []
        for text, tuples in zip(texts, tuple_sets):
            for aspect, polarity in tuples:
                pair_X.append(self._pair(text, aspect))
                pair_y.append(polarity)

        self.fallback_polarity = (
            max(set(pair_y), key=pair_y.count) if pair_y else "neutral"
        )
        if len(set(pair_y)) > 1:
            self.polarity_pipe.fit(pair_X, pair_y)
            self._fitted = True
        else:
            self._fitted = False
        return self

    def predict_sets(self, texts):
        if self.use_lexicon:
            aspect_sets = self.lexicon.predict_sets(texts)
        else:
            aspect_sets = self.aspect_model.predict_sets(texts)

        out = []
        for text, aspects in zip(texts, aspect_sets):
            if not aspects:
                out.append(set())
                continue
            if self._fitted:
                pairs = [self._pair(text, a) for a in sorted(aspects)]
                polarities = self.polarity_pipe.predict(pairs)
            else:
                polarities = [self.fallback_polarity] * len(aspects)
            out.append({(a, p) for a, p in zip(sorted(aspects), polarities)})
        return out

    def predict_with_scores(self, texts):
        """Structured output matching the API contract, with honest score types."""
        results = []
        for text in texts:
            if self.use_lexicon:
                hits = self.lexicon.extract(text)
                aspects = sorted(hits)
                scores = {a: float(hits[a]) for a in aspects}
                score_type = "lexicon_match_count"
            else:
                sets = self.aspect_model.predict_sets([text])[0]
                aspects = sorted(sets)
                scores = {a: None for a in aspects}
                score_type = "uncalibrated_probability"

            items = []
            for a in aspects:
                if self._fitted:
                    pairs = [self._pair(text, a)]
                    probs = self.polarity_pipe.predict_proba(pairs)[0]
                    classes = list(self.polarity_pipe.named_steps["clf"].classes_)
                    j = int(np.argmax(probs))
                    polarity, pol_score = classes[j], float(probs[j])
                else:
                    polarity, pol_score = self.fallback_polarity, None
                items.append({
                    "aspect": a,
                    "polarity": polarity,
                    "aspect_score": scores.get(a),
                    "aspect_score_type": score_type,
                    "polarity_score": pol_score,
                    "polarity_score_type": "uncalibrated_probability",
                })
            results.append(items)
        return results
