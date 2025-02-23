from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import sqlite3
import requests
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

DATABASE = 'newsdata.db'
NEWS_API_KEY = '0b5613df3080480aab10fd39b84ddbaa' 
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
FLAG_THRESHOLD = 1  
RETRAIN_FLAGGED_COUNT = 10  

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, title TEXT, description TEXT, url TEXT, prediction TEXT, 
            percentage REAL, flagged_count INTEGER DEFAULT 0, fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flagged_articles (id TEXT PRIMARY KEY, title TEXT, description TEXT, url TEXT,
            prediction TEXT, percentage REAL, flagged_count INTEGER, flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    ''')
    cursor.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()

init_db()  

def train_model():
    fake_path = os.path.join("database", "Fake.csv")
    true_path = os.path.join("database", "True.csv")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df])
    
    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'], df['label'], test_size=0.2, random_state=42
    )
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', max_features=5000)),
        ('lda', LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20)),
        ('clf', PassiveAggressiveClassifier(max_iter=50))
    ])
    
    pipeline.fit(X_train, y_train)
    print("Initial Training Accuracy:", pipeline.score(X_train, y_train))
    return pipeline

model = train_model()

def retrain_model_with_flagged():
    """
    Retrains the model by combining the original training data with flagged articles.
    Flagged articles are assumed to be fake (label 0). After retraining,
    the flagged_articles table is cleared.
    """
    fake_path = os.path.join("database", "Fake.csv")
    true_path = os.path.join("database", "True.csv")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    original_df = pd.concat([fake_df, true_df])
    original_df['content'] = original_df['title'].astype(str) + " " + original_df['text'].astype(str)
    
    conn = get_db_connection()
    flagged_df = pd.read_sql("SELECT * FROM flagged_articles", conn)
    conn.close()
    
    if not flagged_df.empty:
        flagged_df['content'] = flagged_df['title'].astype(str) + " " + flagged_df['description'].astype(str)
        flagged_df['label'] = 0 
        combined_df = pd.concat([original_df, flagged_df], ignore_index=True)
    else:
        combined_df = original_df
    
    combined_df['processed_content'] = combined_df['content'].apply(preprocess_text)
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', max_features=5000)),
        ('lda', LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20)),
        ('clf', PassiveAggressiveClassifier(max_iter=50))
    ])
    
    pipeline.fit(combined_df['processed_content'], combined_df['label'])
    training_accuracy = pipeline.score(combined_df['processed_content'], combined_df['label'])
    print("Retraining Accuracy:", training_accuracy)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flagged_articles")
    conn.commit()
    cursor.close()
    conn.close()
    
    return pipeline

def fetch_news_from_api(country='us'):
    params = {
        'country': country,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        return articles
    return []

def update_news_in_db():
    articles = fetch_news_from_api()
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        for article in articles:
            article_id = article.get("url")
            if not article_id:
                continue
            title = article.get("title", "")
            description = article.get("description", "")
            cursor.execute("SELECT * FROM news WHERE id = ?", (article_id,))
            result = cursor.fetchone()
            if result:
                continue

            content = (title or "") + " " + (description or "")
            processed_content = preprocess_text(content)
            score = model.decision_function([processed_content])[0]
            probability = 1 / (1 + np.exp(-score))
            prediction = "True" if probability > 0.51 else "Fake"
            percentage = round(probability * 100, 2)

            cursor.execute('''
                INSERT OR IGNORE INTO news (id, title, description, url, prediction, percentage, flagged_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            ''', (article_id, title, description, article_id, prediction, percentage))
        conn.commit()
    except sqlite3.OperationalError as e:
        print("Database error:", e)
    finally:
        cursor.close()
        conn.close()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data.get("content", "")
    if not content:
        return jsonify({"error": "No content provided"}), 400
    processed_content = preprocess_text(content)
    score = model.decision_function([processed_content])[0]
    probability = 1 / (1 + np.exp(-score))
    percentage = round(probability * 100, 2)
    classification = "True" if probability > 0.51 else "Fake"
    return jsonify({"prediction": classification, "percentage": percentage})

@app.route('/news', methods=['GET'])
def news():
    update_news_in_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM news")
    rows = cursor.fetchall()
    articles = []
    for row in rows:
        articles.append({
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "url": row["url"],
            "prediction": row["prediction"],
            "percentage": row["percentage"],
            "flagged_count": row["flagged_count"]
        })
    conn.close()
    return jsonify(articles)

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Accepts feedback in the form of "flag" to mark an article as potentially fake.
    The article's flag count is incremented. When it reaches the threshold,
    the article is archived (removed from main news). When the total number of
    flagged articles reaches RETRAIN_FLAGGED_COUNT, the model is retrained using these flagged articles.
    """
    data = request.get_json()
    article_id = data.get("id")
    feedback_type = data.get("feedback") 

    if not article_id or feedback_type != "flag":
        return jsonify({"error": "Invalid input"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM news WHERE id = ?", (article_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Article not found or already archived"}), 404

    flagged_count = row["flagged_count"] + 1
    cursor.execute("UPDATE news SET flagged_count = ? WHERE id = ?", (flagged_count, article_id))
    
    if flagged_count >= FLAG_THRESHOLD:
        cursor.execute('''
            INSERT OR IGNORE INTO flagged_articles 
            (id, title, description, url, prediction, percentage, flagged_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (row["id"], row["title"], row["description"], row["url"], row["prediction"], row["percentage"], flagged_count))
        cursor.execute("DELETE FROM news WHERE id = ?", (article_id,))
    
    conn.commit()
    cursor.close()
    conn.close()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM flagged_articles")
    count_row = cursor.fetchone()
    flagged_total = count_row["cnt"]
    cursor.close()
    conn.close()

    global model
    if flagged_total >= RETRAIN_FLAGGED_COUNT:
        print(f"Retraining model with {flagged_total} flagged articles...")
        model = retrain_model_with_flagged()

    return jsonify({"message": "Feedback recorded", "flag_count": flagged_count})

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Manually retrain the model using both the original dataset and any flagged articles.
    """
    global model
    model = retrain_model_with_flagged()
    return jsonify({"message": "Model retrained successfully"})

if __name__ == '__main__':
    app.run(debug=True)
