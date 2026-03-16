# FinBERT Financial News Sentiment Analysis and Stock Impact

## Overview

This project analyzes financial news articles using **FinBERT**, a transformer-based language model specialized for financial sentiment analysis. The system collects recent stock-related news using NewsAPI, extracts article content, classifies sentiment using FinBERT, and analyzes how news sentiment correlates with stock price movements.

The objective is to explore whether financial news sentiment can influence short-term stock price changes.

## Key Features

* Fetch real-time financial news using NewsAPI
* Extract article content automatically
* Perform sentiment classification using **FinBERT**
* Convert sentiment into numerical scores
* Fetch stock market data using Yahoo Finance
* Calculate next-day stock price impact
* Analyze correlation between sentiment and price change
* Visualize results using a heatmap

## Technologies Used

* Python
* Hugging Face Transformers
* FinBERT (ProsusAI/finbert)
* NewsAPI
* yFinance
* Newspaper3k
* Pandas
* NumPy
* Matplotlib
* Seaborn

## Model Used

### FinBERT

FinBERT is a **BERT-based transformer model trained on financial text** to perform sentiment analysis on financial news and reports.

Model:
ProsusAI/finbert

FinBERT classifies text into:

* Positive
* Negative
* Neutral

Example usage:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

classifier("Tesla stock rises after strong earnings report")
```

## Project Workflow

1. Fetch financial news articles using NewsAPI.
2. Extract full article text using Newspaper3k.
3. Apply FinBERT to determine sentiment.
4. Convert sentiment labels into numerical sentiment scores.
5. Retrieve stock price data using Yahoo Finance.
6. Calculate next-day percentage price change.
7. Analyze correlation between sentiment and stock price movement.
8. Visualize stock impact using a heatmap.

## Example Output

The system produces:

* Sentiment classification for each news article
* Confidence scores from FinBERT
* Stock price movement percentage
* Correlation between sentiment and stock impact
* Heatmap visualization showing sentiment influence on stocks

## Installation

Clone the repository:

```bash
git clone https://github.com/SATYAJEET-JADHAV/Github-final-project.git
```

Install dependencies:

```bash
pip install transformers torch newsapi-python yfinance newspaper3k pandas numpy matplotlib seaborn
```

## Running the Project

Run the Python script or Jupyter notebook to:

1. Fetch financial news articles
2. Analyze sentiment using FinBERT
3. Retrieve stock price data
4. Generate sentiment vs price impact analysis

## Example Use Case

Example query:

```
stock earnings
```

The system retrieves related financial news, analyzes sentiment, and evaluates whether the sentiment correlates with stock price movement.

## Future Improvements

* Add support for multiple financial news sources
* Implement real-time sentiment tracking dashboard
* Improve ticker extraction using Named Entity Recognition (NER)
* Train a model to predict stock price movement from sentiment signals

## Author

Satyajeet Jadhav
Electronics and Telecommunication Engineering
Vishwakarma Institute of Technology, Pune
