"""Text preprocessing.

Two separate branches, deliberately. The original plan used one pipeline feeding both
the classical and the transformer model, which is a mistake: lemmatisation, stopword
removal and lowercasing destroy exactly the information a pretrained transformer relies
on (casing, punctuation, function words, subword morphology).

    Branch A (classical)   : aggressive normalisation for TF-IDF
    Branch B (transformer) : minimal normalisation only

PII removal and whitespace normalisation are shared and always applied first.
"""

import re

# --- Shared: PII redaction ---------------------------------------------------

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\d{10}\b")
_ROLL = re.compile(r"\b[A-Z]\d{3,4}\b")          # e.g. E029 — matches our own roll format
_URL = re.compile(r"https?://\S+|www\.\S+")
_HONORIFIC = re.compile(
    r"\b(?:Prof|Professor|Dr|Mr|Mrs|Ms|Sir|Madam|Ma'am)\.?\s+[A-Z][a-z]+\b"
)


def redact_pii(text: str) -> str:
    """Remove direct identifiers. Applied before anything else, always."""
    text = _URL.sub("[URL]", text)
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _HONORIFIC.sub("[INSTRUCTOR]", text)
    text = _ROLL.sub("[ID]", text)
    return text


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# --- Branch B: transformer ---------------------------------------------------

def preprocess_transformer(text: str) -> str:
    """Minimal. Casing and punctuation are signal for a pretrained model, not noise."""
    return normalise_whitespace(redact_pii(text))


# --- Branch A: classical -----------------------------------------------------

# A general stopword list would delete the negation and contrast words that carry
# most of the sentiment signal in student feedback. This list is the standard English
# stopword set with those terms explicitly held back.
_KEEP = {
    "not", "no", "nor", "never", "none", "nothing", "cannot", "cant", "dont",
    "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont", "wouldnt",
    "shouldnt", "couldnt", "hardly", "barely", "scarcely", "rarely", "seldom",
    "but", "however", "although", "though", "yet", "except", "unless",
    "too", "very", "more", "most", "less", "least", "only", "just", "still",
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "if", "while", "of", "at", "by", "for", "with",
    "about", "into", "through", "during", "to", "from", "in", "out", "on", "off",
    "then", "once", "here", "there", "all", "any", "both", "each", "few", "other",
    "some", "such", "own", "same", "so", "than", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "as", "because", "since", "also", "get", "got",
} - _KEEP

_NEGATORS = {
    "not", "no", "never", "cannot", "cant", "dont", "doesnt", "didnt", "isnt",
    "wasnt", "arent", "werent", "wont", "wouldnt", "shouldnt", "couldnt",
    "hardly", "barely", "scarcely", "without",
}

# A negation's scope ends at a clause boundary, not at an arbitrary word count.
_CLAUSE_BOUNDARY = {"but", "however", "although", "though", "yet", "and", "."}

_CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
    "'ve": " have", "'m": " am", "'d": " would",
}


def _expand_contractions(text: str) -> str:
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    return text


def mark_negation(tokens: list[str]) -> list[str]:
    """Prefix tokens inside a negation's scope with NOT_.

    'the lectures were not very informative'
        -> [..., 'not', 'NOT_very', 'NOT_informative']

    This lets a bag-of-words model distinguish 'informative' from 'NOT_informative',
    which it otherwise cannot do. Scope runs to the next clause boundary rather than
    a fixed window, so 'not clear but the notes are good' does not wrongly negate
    'good'. Its contribution is measured as an ablation in run_baseline.py.
    """
    out, negating = [], False
    for tok in tokens:
        if tok in _CLAUSE_BOUNDARY:
            negating = False
            out.append(tok)
            continue
        out.append(f"NOT_{tok}" if negating else tok)
        if tok in _NEGATORS:
            negating = True
    return out


def preprocess_classical(text: str, negation: bool = True,
                         drop_stopwords: bool = True) -> str:
    """Normalise for TF-IDF. Returns a whitespace-joined token string."""
    text = normalise_whitespace(redact_pii(text)).lower()
    text = _expand_contractions(text)
    text = re.sub(r"[^a-z0-9\s.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.replace(".", " . ").split()
    if negation:
        tokens = mark_negation(tokens)
    tokens = [t for t in tokens if t != "."]
    if drop_stopwords:
        tokens = [
            t for t in tokens
            if t.replace("NOT_", "") not in _STOPWORDS and len(t) > 1
        ]
    return " ".join(tokens)
