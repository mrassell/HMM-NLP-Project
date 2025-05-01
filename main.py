from src.data_loader import load_entries
from src.sentiment_analyzer import process_entries
from src.hmm_modeler import train_hmm
from src.visualizer import plot_sentiment_over_time, plot_entry_lengths

import json
import os

def main():
    # Load entries
    entries = load_entries("entries.json")
    
    # Process entries
    processed_entries = process_entries(entries)
    
    # Ensure outputs directory exists
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Save processed entries
    with open("outputs/processed_entries.json", "w") as f:
        json.dump(processed_entries, f, indent=2)
    
    # Extract sentiment scores and dates
    sentiment_scores = [entry["sentiment_score"] for entry in processed_entries]
    dates = [entry["date"] for entry in processed_entries]
    
    # Print debug information
    print("\n🔍 Sentiment Analysis Debug:")
    for entry, score in zip(processed_entries, sentiment_scores):
        print(f"Date: {entry['date']}")
        print(f"Score: {score:.4f}")
        print(f"Label: {entry['sentiment_label']}")
        print("---")
    
    # Train HMM
    trans_mat, hidden_states = train_hmm(processed_entries, n_states=5)
    
    # Print HMM debug information
    print("\n🔍 HMM State Debug:")
    for date, state in zip(dates, hidden_states):
        print(f"Date: {date}")
        print(f"Hidden State: {state}")
        print("---")
    
    # Generate visualizations
    plot_sentiment_over_time(processed_entries, "outputs/plots/sentiment_over_time.png")
    plot_entry_lengths(processed_entries, "outputs/plots/entry_lengths.png")
    
    # Print summary statistics
    print("\n📊 Analysis Summary:")
    print(f"Total entries processed: {len(processed_entries)}")
    print(f"Average entry length: {sum(len(e['entry']) for e in processed_entries) / len(processed_entries):.0f} characters")
    
    # Print sentiment distribution with all categories
    total_very_positive = sum(1 for e in processed_entries if e["sentiment_label"] == "very positive")
    total_positive = sum(1 for e in processed_entries if e["sentiment_label"] == "positive")
    total_neutral = sum(1 for e in processed_entries if e["sentiment_label"] == "neutral")
    total_negative = sum(1 for e in processed_entries if e["sentiment_label"] == "negative")
    total_very_negative = sum(1 for e in processed_entries if e["sentiment_label"] == "very negative")
    
    print("\n🎭 Sentiment Distribution:")
    print(f"Very positive entries: {total_very_positive} ({total_very_positive/len(processed_entries)*100:.1f}%)")
    print(f"Positive entries: {total_positive} ({total_positive/len(processed_entries)*100:.1f}%)")
    print(f"Neutral entries: {total_neutral} ({total_neutral/len(processed_entries)*100:.1f}%)")
    print(f"Negative entries: {total_negative} ({total_negative/len(processed_entries)*100:.1f}%)")
    print(f"Very negative entries: {total_very_negative} ({total_very_negative/len(processed_entries)*100:.1f}%)")
    
    print("\n✅ All done! Check the outputs/ folder for results!")

if __name__ == "__main__":
    main() 