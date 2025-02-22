# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import os
# import json
# import sqlite3
# import requests
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.model_selection import train_test_split

# app = Flask(__name__)

# # Constants and database settings
# DATABASE = 'newsdata.db'
# NEWS_API_KEY = '0b5613df3080480aab10fd39b84ddbaa'  # Replace with your NewsAPI key.
# NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
# FLAG_THRESHOLD = 10  # When an article is flagged 10 times, archive it.

# # -------------------------------------------------------------------
# # Database functions
# # -------------------------------------------------------------------
# def get_db_connection():
#     conn = sqlite3.connect(DATABASE, check_same_thread=False)
#     conn.row_factory = sqlite3.Row
#     return conn

# def init_db():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Create table for fetched news articles
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS news (
#             id TEXT PRIMARY KEY,
#             title TEXT,
#             description TEXT,
#             url TEXT,
#             prediction TEXT,
#             percentage REAL,
#             flagged_count INTEGER DEFAULT 0,
#             fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     ''')
#     # Create table for flagged (archived) articles
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS flagged_articles (
#             id TEXT PRIMARY KEY,
#             title TEXT,
#             description TEXT,
#             url TEXT,
#             prediction TEXT,
#             percentage REAL,
#             flagged_count INTEGER,
#             flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     ''')
#     cursor.execute("PRAGMA journal_mode=WAL;")
#     conn.commit()
#     conn.close()

# init_db()  # Initialize our database upon startup.

# # -------------------------------------------------------------------
# # Model Training
# # -------------------------------------------------------------------
# def train_model():
#     # Load CSV datasets from project/database folder.
#     fake_path = os.path.join("database", "Fake.csv")
#     true_path = os.path.join("database", "True.csv")
#     fake_df = pd.read_csv(fake_path)
#     true_df = pd.read_csv(true_path)
    
#     # Assign labels: 0 for fake, 1 for true.
#     fake_df['label'] = 0
#     true_df['label'] = 1
#     df = pd.concat([fake_df, true_df])
    
#     # Combine title and text columns for better feature extraction.
#     df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    
#     # Split dataset for training (optional testing and tuning).
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['content'], df['label'], test_size=0.2, random_state=42
#     )
    
#     # Build a pipeline using TF-IDF vectorization and Passive Aggressive Classifier.
#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
#         ('clf', PassiveAggressiveClassifier(max_iter=50))
#     ])
#     pipeline.fit(X_train, y_train)
#     print("Training Accuracy:", pipeline.score(X_train, y_train))
#     return pipeline

# model = train_model()  # Global classifier variable

# # -------------------------------------------------------------------
# # NewsAPI Integration and Database Update
# # -------------------------------------------------------------------
# def fetch_news_from_api(country='us'):
#     params = {
#         'country': country,
#         'apiKey': NEWS_API_KEY
#     }
#     response = requests.get(NEWS_API_URL, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         articles = data.get('articles', [])
#         return articles
#     return []

# def update_news_in_db():
#     articles = fetch_news_from_api()
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         for article in articles:
#             article_id = article.get("url")
#             if not article_id:
#                 continue
#             title = article.get("title", "")
#             description = article.get("description", "")

#             cursor.execute("SELECT * FROM news WHERE id = ?", (article_id,))
#             result = cursor.fetchone()
#             if result:
#                 continue

#             content = (title or "") + " " + (description or "")
#             score = model.decision_function([content])[0]
#             probability = 1 / (1 + np.exp(-score))
#             prediction = "True" if probability > 0.51 else "Fake"
#             percentage = round(probability * 100, 2)

#             cursor.execute('''
#                 INSERT OR IGNORE INTO news (id, title, description, url, prediction, percentage, flagged_count)
#                 VALUES (?, ?, ?, ?, ?, ?, 0)
#             ''', (article_id, title, description, article_id, prediction, percentage))
#         conn.commit()
#     except sqlite3.OperationalError as e:
#         print("Database error:", e)
#     finally:
#         cursor.close()
#         conn.close()

# # -------------------------------------------------------------------
# # Flask Endpoints
# # -------------------------------------------------------------------
# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     content = data.get("content", "")
#     if not content:
#         return jsonify({"error": "No content provided"}), 400
#     pred = model.predict([content])[0]
#     score = model.decision_function([content])[0]
#     probability = 1 / (1 + np.exp(-score))
#     percentage = round(probability * 100, 2)
#     classification = "True" if probability > 0.51 else "Fake"
#     return jsonify({"prediction": classification, "percentage": percentage})

# @app.route('/news', methods=['GET'])
# def news():
#     # Update the database with the latest news from the API.
#     update_news_in_db()
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM news")
#     rows = cursor.fetchall()
#     articles = []
#     for row in rows:
#         articles.append({
#             "id": row["id"],
#             "title": row["title"],
#             "description": row["description"],
#             "url": row["url"],
#             "prediction": row["prediction"],
#             "percentage": row["percentage"],
#             "flagged_count": row["flagged_count"]
#         })
#     conn.close()
#     return jsonify(articles)

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     data = request.get_json()
#     article_id = data.get("id")
#     feedback_type = data.get("feedback")
    
#     if not article_id or feedback_type != "flag":
#         return jsonify({"error": "Invalid input"}), 400

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     # Retrieve the article from the news table.
#     cursor.execute("SELECT flagged_count, * FROM news WHERE id = ?", (article_id,))
#     row = cursor.fetchone()
#     if not row:
#         conn.close()
#         return jsonify({"error": "Article not found or already archived"}), 404

#     # Increment flag count.
#     flagged_count = row["flagged_count"] + 1
#     cursor.execute("UPDATE news SET flagged_count = ? WHERE id = ?", (flagged_count, article_id))
    
#     # Once threshold is reached, move the article to the flagged_articles table.
#     if flagged_count >= FLAG_THRESHOLD:
#         cursor.execute('''
#             INSERT OR IGNORE INTO flagged_articles 
#             (id, title, description, url, prediction, percentage, flagged_count)
#             VALUES (?, ?, ?, ?, ?, ?, ?)
#         ''', (row["id"], row["title"], row["description"], row["url"], row["prediction"], row["percentage"], flagged_count))
#         cursor.execute("DELETE FROM news WHERE id = ?", (article_id,))
    
#     conn.commit()
#     conn.close()
#     return jsonify({"message": "Feedback recorded", "flag_count": flagged_count})

# @app.route('/retrain', methods=['POST'])
# def retrain():
#     global model
#     # Retrain the model (you may enhance this by incorporating flagged articles if labeled)
#     model = train_model()
#     return jsonify({"message": "Model retrained successfully"})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Constants and database settings
DATABASE = 'newsdata.db'
NEWS_API_KEY = '0b5613df3080480aab10fd39b84ddbaa'  # Replace with your NewsAPI key.
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'
FLAG_THRESHOLD = 10  # When an article is flagged 10 times, it is archived.
RETRAIN_FLAGGED_COUNT = 10  # When there are 10 flagged articles in total, retraining is triggered.

# -------------------------------------------------------------------
# Preprocessing Function
# -------------------------------------------------------------------
def preprocess_text(text):
    """
    Lowercase, remove non-alphabetic characters, tokenize,
    and remove stopwords.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# -------------------------------------------------------------------
# Database Functions
# -------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create table for fetched news articles.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            url TEXT,
            prediction TEXT,
            percentage REAL,
            flagged_count INTEGER DEFAULT 0,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    # Create table for flagged (archived) articles.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flagged_articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            url TEXT,
            prediction TEXT,
            percentage REAL,
            flagged_count INTEGER,
            flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    cursor.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()

init_db()  # Initialize our database upon startup.

# -------------------------------------------------------------------
# Model Training with LDA Feature Extraction and Passive-Aggressive Classifier
# -------------------------------------------------------------------
def train_model():
    # Load CSV datasets from the project/database folder.
    fake_path = os.path.join("database", "Fake.csv")
    true_path = os.path.join("database", "True.csv")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    # Assign labels: 0 for fake, 1 for true.
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df])
    
    # Combine title and text columns.
    df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    # Preprocess content.
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    # Split dataset for training.
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_content'], df['label'], test_size=0.2, random_state=42
    )
    
    # Build a pipeline: CountVectorizer -> LDA -> PassiveAggressiveClassifier.
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', max_features=5000)),
        ('lda', LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20)),
        ('clf', PassiveAggressiveClassifier(max_iter=50))
    ])
    
    pipeline.fit(X_train, y_train)
    print("Initial Training Accuracy:", pipeline.score(X_train, y_train))
    return pipeline

# Global model variable
model = train_model()

# -------------------------------------------------------------------
# Retraining Using Flagged Articles (Reinforcement Learning)
# -------------------------------------------------------------------
def retrain_model_with_flagged():
    """
    Retrains the model by combining the original training data with flagged articles.
    Flagged articles are assumed to be fake (label 0). After retraining,
    the flagged_articles table is cleared.
    """
    # Load original training data.
    fake_path = os.path.join("database", "Fake.csv")
    true_path = os.path.join("database", "True.csv")
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)
    
    fake_df['label'] = 0
    true_df['label'] = 1
    original_df = pd.concat([fake_df, true_df])
    original_df['content'] = original_df['title'].astype(str) + " " + original_df['text'].astype(str)
    
    # Load flagged articles from DB.
    conn = get_db_connection()
    flagged_df = pd.read_sql("SELECT * FROM flagged_articles", conn)
    conn.close()
    
    if not flagged_df.empty:
        # Use title and description for flagged articles.
        flagged_df['content'] = flagged_df['title'].astype(str) + " " + flagged_df['description'].astype(str)
        flagged_df['label'] = 0  # Flagged articles are assumed fake.
        # Combine original data with flagged articles.
        combined_df = pd.concat([original_df, flagged_df], ignore_index=True)
    else:
        combined_df = original_df
    
    # Preprocess the combined content.
    combined_df['processed_content'] = combined_df['content'].apply(preprocess_text)
    
    # Train the model on the combined dataset.
    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words='english', max_features=5000)),
        ('lda', LatentDirichletAllocation(n_components=10, random_state=42, max_iter=20)),
        ('clf', PassiveAggressiveClassifier(max_iter=50))
    ])
    
    pipeline.fit(combined_df['processed_content'], combined_df['label'])
    training_accuracy = pipeline.score(combined_df['processed_content'], combined_df['label'])
    print("Retraining Accuracy:", training_accuracy)
    
    # Clear the flagged_articles table after retraining.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flagged_articles")
    conn.commit()
    cursor.close()
    conn.close()
    
    return pipeline

# -------------------------------------------------------------------
# NewsAPI Integration and Database Update
# -------------------------------------------------------------------
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
            # Skip if already in DB.
            cursor.execute("SELECT * FROM news WHERE id = ?", (article_id,))
            result = cursor.fetchone()
            if result:
                continue

            # Preprocess and predict using the trained pipeline.
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

# -------------------------------------------------------------------
# Flask Endpoints
# -------------------------------------------------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data.get("content", "")
    if not content:
        return jsonify({"error": "No content provided"}), 400
    # Preprocess the input text.
    processed_content = preprocess_text(content)
    pred = model.predict([processed_content])[0]
    score = model.decision_function([processed_content])[0]
    probability = 1 / (1 + np.exp(-score))
    percentage = round(probability * 100, 2)
    classification = "True" if probability > 0.51 else "Fake"
    return jsonify({"prediction": classification, "percentage": percentage})

@app.route('/news', methods=['GET'])
def news():
    # Update the database with the latest news from the API.
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
    feedback_type = data.get("feedback")  # Expected to be "flag"

    if not article_id or feedback_type != "flag":
        return jsonify({"error": "Invalid input"}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM news WHERE id = ?", (article_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Article not found or already archived"}), 404

    # Increment flag count.
    flagged_count = row["flagged_count"] + 1
    cursor.execute("UPDATE news SET flagged_count = ? WHERE id = ?", (flagged_count, article_id))
    
    # If flag count reaches the threshold, archive the article.
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

    # Check if the total number of flagged articles reaches the retraining threshold.
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
