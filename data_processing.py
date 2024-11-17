# data_processing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    def to_sentiment(rating):
        rating = int(rating)
        if rating <= 2:
            return 0
        elif rating == 3:
            return 1
        else: 
            return 2
    df['sentiment'] = df.score.apply(to_sentiment)
    sns.countplot(df.sentiment)
    plt.xlabel('Review Sentiment')
    plt.show()
    return df
