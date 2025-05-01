# Diary Sentiment Analysis with HMM

A Python project that analyzes diary entries using Hidden Markov Models (HMM) and sentiment analysis to track emotional patterns over time.

## Features

- Sentiment analysis of diary entries using VADER
- Custom Hidden Markov Model implementation from scratch
- Visualization of sentiment trends over time
- Analysis of entry lengths and writing patterns

## Project Structure

```
.
├── src/
│   ├── data_loader.py      # Loads diary entries from JSON
│   ├── hmm_modeler.py      # Custom HMM implementation
│   ├── preprocessor.py     # Text preprocessing utilities
│   ├── sentiment_analyzer.py# Sentiment analysis using VADER
│   └── visualizer.py       # Data visualization functions
├── outputs/
│   ├── plots/             # Generated visualizations
│   └── processed_entries.json
├── entries.json           # Input diary entries
└── requirements.txt       # Project dependencies
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

This will:
1. Load and process diary entries
2. Perform sentiment analysis
3. Train HMM model
4. Generate visualizations
5. Save results in the outputs directory

## Visualizations

- **Sentiment Over Time**: Shows how emotional sentiment changes across entries
- **Entry Length Distribution**: Displays the distribution of entry lengths

## Dependencies

- numpy
- textblob
- nltk
- matplotlib
- seaborn
- vaderSentiment
- pandas 