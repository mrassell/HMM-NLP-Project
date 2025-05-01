import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import numpy as np
import os

def plot_sentiment_over_time(entries: List[Dict], output_path: str):
    dates = [entry["date"] for entry in entries][::-1]
    scores = [entry["sentiment_score"] for entry in entries][::-1]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=scores, marker="o")
    plt.title("Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    
    plt.ylim(-1, 1)
    plt.yticks(np.arange(-1, 1.1, 0.2))
    
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_entry_lengths(entries: List[Dict], output_path: str):
    plt.figure(figsize=(10, 6))
    
    lengths = [len(entry["entry"]) for entry in entries]
    
    sns.histplot(lengths, bins=20)
    plt.title("Distribution of Entry Lengths")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    
    mean_length = sum(lengths) / len(lengths)
    plt.axvline(mean_length, color='r', linestyle='--', label=f'Mean: {mean_length:.0f} chars')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close() 