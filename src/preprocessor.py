import re

def preprocess_text(text: str) -> str:
    """Basic cleaning: lowercase, remove non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text.strip() 