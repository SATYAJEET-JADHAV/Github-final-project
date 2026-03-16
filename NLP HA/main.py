import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from datetime import datetime, timedelta
import numpy as np
from newsapi import NewsApiClient

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

newsapi = NewsApiClient(api_key="797aa33860474495856b2528d941aa26")  

query = "stock earnings"

try:
    all_articles = newsapi.get_everything(q=query,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=5)
    articles = all_articles['articles']
    print(f"Fetched {len(articles)} articles.")
except Exception as e:
    print(f"Error fetching articles: {e}")
    articles = []

news_data = []
for article in articles:
    news_data.append({
        "url": article['url'],
        "ticker": article['title'].split()[0] 
    })

def get_sentiment_from_url(news_item):
    try:
        article = Article(news_item['url'])
        article.download()
        article.parse()
        text = article.text

        if not text:
            print(f"Skipping empty article at {news_item['url']}")
            return None

        result = sentiment_pipeline(text[:512])[0]
        news_item['sentiment'] = result['label']
        news_item['confidence'] = result['score']
        news_item['date'] = article.publish_date or datetime.now()
        news_item['text'] = text
        return news_item
    except Exception as e:
        print(f"Error processing {news_item['url']}: {e}")
        return None

results = [get_sentiment_from_url(item) for item in news_data]
results = [r for r in results if r is not None]
df = pd.DataFrame(results)

if df.empty:
    print("No sentiment data found.")
else:
    print(f"Sentiment data before adding sentiment_score:\n{df[['ticker', 'sentiment', 'confidence']].head()}")

sentiment_map = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0}
df['sentiment_score'] = df['sentiment'].map(sentiment_map)

if 'sentiment_score' in df.columns:
    print("Sentiment scores added successfully.")
else:
    print("Error: Sentiment scores not added.")

def get_price_change(ticker, date):
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.history(period="1d", start=date.strftime('%Y-%m-%d'), end=(date + timedelta(days=2)).strftime('%Y-%m-%d'))
        
        if stock_info.empty:
            print(f"Skipping invalid or delisted ticker: {ticker}")
            return None
        
        open_price = stock_info.iloc[0]['Close']
        next_day_close = stock_info.iloc[1]['Close']
        change_pct = ((next_day_close - open_price) / open_price) * 100
        return round(change_pct, 2)
    except Exception as e:
        print(f"Error fetching stock data for {ticker} on {date}: {e}")
        return None

df['impact_pct'] = df.apply(lambda row: get_price_change(row['ticker'], row['date']), axis=1)

if df['impact_pct'].isnull().all():
    print("No price impact data found.")
else:
    print(f"Price impact data: {df[['ticker', 'impact_pct']].head()}")

df.dropna(subset=['impact_pct'], inplace=True)

if len(df) < 2:
    print("Not enough data to compute correlation.")
else:
    correlation = df[['sentiment_score', 'impact_pct']].corr().iloc[0, 1]
    print(f"Correlation between sentiment and price change: {correlation:.2f}")

    heatmap_data = df.pivot_table(index='ticker', values='impact_pct', aggfunc='mean')

    plt.figure(figsize=(6, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", center=0, fmt=".2f")
    plt.title("Average Sentiment Impact on Stock Price")
    plt.ylabel("Stock Ticker")
    plt.tight_layout()
    plt.show()
