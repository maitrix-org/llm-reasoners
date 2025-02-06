import logging
import re

import ftfy

log = logging.getLogger(__name__)


class LazySpacy:
    """Lazily load the spacy pipeline when needed to save memory."""

    def __init__(self, model: str):
        self.model = model
        self.pipe = None

    def _load_pipe(self):
        import spacy

        self.pipe = spacy.load('en_core_web_sm')

    def __call__(self, *args, **kwargs):
        if self.pipe is None:
            self._load_pipe()
        return self.pipe(*args, **kwargs)


nlp = LazySpacy('en_core_web_sm')


def normalize(text, remove_stopwords=False):
    """
    Normalize a given string for string-based metrics. Specifically, this does the following:
    - fix encoding errors (ftfy)
    - normalize numbers
    - lemmatize words
    - remove stopwords (optional)
    - remove punctuation
    - remove redundant whitespace
    """
    text = str(text).lower()
    text = ftfy.fix_text(text)
    text = normalize_numbers(text)
    text = lemmatize(text, remove_stopwords=remove_stopwords)
    text = remove_punct(text)
    text = normalize_whitespace(text)
    return text


def normalize_numbers(text: str):
    """Use regex to normalize numbers with commas"""
    # numbers with commas
    comma_sub_text = re.sub(
        r'(\d+,)+\d+(\.\d+)?', lambda m: m[0].replace(',', ''), text
    )
    return comma_sub_text


def lemmatize(text: str, remove_stopwords=False):
    """Return a normalized string with each word replaced by its lemmatized version."""
    doc = nlp(text)
    if remove_stopwords:
        return ' '.join(tok.lemma_ for tok in doc if not tok.is_stop)
    return ' '.join(tok.lemma_ for tok in doc)


def remove_punct(text: str):
    """Remove all punctuation from the string."""
    return re.sub(r'[,.?!:;]', '', text)


def normalize_whitespace(text: str):
    """Replace all whitespace with a single space."""
    return re.sub(r'\s+', ' ', text)
