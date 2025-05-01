from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Tuple
from src.preprocessor import preprocess_text

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Return sentiment label and score using VADER with enhanced sensitivity."""
    analyzer = SentimentIntensityAnalyzer()
    
    # Get sentence-level scores
    sentences = text.split('.')
    sentence_scores = []
    for sentence in sentences:
        if sentence.strip():
            scores = analyzer.polarity_scores(sentence)
            # Calculate a more sensitive score that considers the balance of pos/neg
            sensitive_score = scores['pos'] - scores['neg']
            # If there's significant negativity, amplify it
            if scores['neg'] > 0.2:
                sensitive_score -= scores['neg'] * 0.5
            sentence_scores.append(sensitive_score)
    
    if not sentence_scores:
        return "neutral", 0.0
    
    # Find the most extreme scores
    max_score = max(sentence_scores)
    min_score = min(sentence_scores)
    avg_score = sum(sentence_scores) / len(sentence_scores)
    
    # Use the score that represents the strongest emotion
    if abs(min_score) > abs(max_score):
        final_score = min_score
    else:
        final_score = max_score
    
    # If there's a mix of positive and negative, bias towards the negative
    if max_score > 0.2 and min_score < -0.2:
        final_score = (final_score + min_score) / 2
    
    # Normalize score to [-1, 1] range
    final_score = max(min(final_score, 1.0), -1.0)
    
    # Determine label with more sensitive thresholds
    if final_score >= 0.4:
        label = "very positive"
    elif final_score >= 0.1:
        label = "positive"
    elif final_score > -0.1:
        label = "neutral"
    elif final_score > -0.4:
        label = "negative"
    else:
        label = "very negative"
    
    return label, final_score

def process_entries(entries: List[Dict]) -> List[Dict]:
    """Process each entry to add cleaned text and sentiment analysis."""
    processed = []
    for entry in entries:
        cleaned = preprocess_text(entry["entry"])
        label, score = analyze_sentiment(cleaned)
        processed.append({
            "date": entry["date"],
            "entry": entry["entry"],
            "cleaned_entry": cleaned,
            "sentiment_label": label,
            "sentiment_score": score
        })
    return processed 