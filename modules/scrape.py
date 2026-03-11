import re
import fitz
import json
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normalizes text by converting to lowercase, expanding contractions, and removing special characters."""

    text = text.lower()

    text = re.sub(r"\'s\b", " is", text)     # it's -> it is, John's -> John is
    text = re.sub(r"n\'t\b", " not", text)   # wasn't -> was not
    text = re.sub(r"\'re\b", " are", text)   # they're -> they are
    text = re.sub(r"\'ll\b", " will", text)  # i'll -> i will
    text = re.sub(r"\'m\b", " am", text)     # i'm -> i am
    text = re.sub(r"\'ve\b", " have", text)  # we've -> we have
    text = re.sub(r"\'d\b", " would", text)  # he'd -> he would

    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)
    text = re.sub(r'[\"\']', ' ', text)

    text = text.replace('`', "'")

    text = re.sub(r'http\S+|www\S+|@\w+', '', text)  # remove urls
    text = re.sub(r'[^a-z0-9\s]', ' ', text)         # keep only alphanumeric chars

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_text(pdf_path: str | Path) -> list[dict]:
    """Extracts text from a PDF file and normalizes it."""

    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = str(page.get_text()) or ""
        clean_text = normalize_text(text)
        texts.append(clean_text)

    pages = []
    for i, text in enumerate(texts):
        pages.append({
            "source": pdf_path,
            "page": i + 1,
            "text": text,
        })

    with open("data/extracted.json", "w") as f:
        json.dump(pages, f, indent=4)

    return pages
