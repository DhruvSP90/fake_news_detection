# Fake News Detection System

## Overview
This project is a **Fake News Detection System** that fetches live news articles, classifies them as "Fake" or "True," and allows user feedback for continuous model improvement. It integrates **Machine Learning (Passive-Aggressive Classifier & LDA)** with **Flask**, **SQLite**, and **NewsAPI**.

## Features
- Fetches real-time news using **NewsAPI**
- Classifies news as **"Fake" or "True"**
- Allows users to flag potentially fake articles
- Automatically archives flagged articles after a threshold
- Retrains the model when enough flagged articles accumulate
- Provides a REST API for predictions and news retrieval

## Tech Stack
- **Backend**: Flask, SQLite, Pandas, NumPy, Requests
- **Machine Learning**: Passive-Aggressive Classifier, LDA, CountVectorizer (Scikit-Learn)
- **Data Processing**: NLTK (Stopwords, Tokenization)
- **Database**: SQLite

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fake-news-detection.git
   cd fake-news-detection
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Set up the database:
   ```bash
   python app.py
   ```
   This initializes the SQLite database.

4. Run the Flask server:
   ```bash
   python app.py
   ```
   The API will be available at `http://127.0.0.1:5000/`.

## API Endpoints
| Method | Endpoint | Description |
|--------|------------|----------------|
| `GET` | `/` | Returns the homepage |
| `POST` | `/predict` | Classifies input text as fake or true |
| `GET` | `/news` | Fetches latest news and classifications |
| `POST` | `/feedback` | Flags an article for potential retraining |
| `POST` | `/retrain` | Manually retrains the model |

## Model Training
- Uses **Fake News Detection Dataset** from Kaggle
- Combines **CountVectorizer** (text vectorization) with **LDA** (feature extraction)
- Uses **Passive-Aggressive Classifier** for fast, incremental learning
- Retrains using flagged articles for **adaptive learning**

## Usage Example
### Predicting Fake News
Send a `POST` request to `/predict` with the following JSON:
```json
{
  "text": "Breaking news: Scientists discover water on Mars."
}
```
Response:
```json
{
  "prediction": "True"
}
```

### Flagging Fake News
Send a `POST` request to `/feedback`:
```json
{
  "article_id": 123,
  "reason": "Suspicious source"
}
```
Response:
```json
{
  "message": "Feedback received. Article flagged."
}
```

## Notes
- Requires **NewsAPI Key** (replace `NEWS_API_KEY` in `app.py`)
- Adjust `FLAG_THRESHOLD` and `RETRAIN_FLAGGED_COUNT` as needed
