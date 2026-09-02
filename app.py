from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import time
import os

import nltk

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# NLTK SETUP
# ============================================================

def setup_nltk():

    packages = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for path, package in packages:

        try:
            nltk.data.find(path)

        except LookupError:
            try:
                nltk.download(package, quiet=False)
            except Exception:
                pass


setup_nltk()


# ============================================================
# LOAD DATA
# ============================================================

DOCUMENT_FILE = "documents.csv"
QUERY_FILE = "queries.csv"

documents = pd.read_csv(DOCUMENT_FILE)
queries = pd.read_csv(QUERY_FILE)

documents["text"] = (
    documents["title"].fillna("") + " " +
    documents["content"].fillna("")
)


# ============================================================
# NLP PREPROCESSING
# ============================================================

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = {
        "the", "is", "a", "an", "and", "or",
        "to", "of", "in", "for", "on", "with",
        "i", "want", "need", "about", "give",
        "me", "information"
    }


lemmatizer = WordNetLemmatizer()


def preprocess(text):

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in STOPWORDS
    ]

    # Lemmatization
    processed = []

    for word in tokens:

        try:
            word = lemmatizer.lemmatize(word)
        except Exception:
            pass

        processed.append(word)

    return " ".join(processed)


documents["processed"] = documents["text"].apply(preprocess)


# ============================================================
# TF-IDF MODEL
# ============================================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000
)

document_matrix = vectorizer.fit_transform(
    documents["processed"]
)


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(query):

    processed = preprocess(query)

    words = processed.split()

    keywords = []

    for word in words:

        if len(word) > 2 and word not in keywords:
            keywords.append(word)

    return keywords[:10]


# ============================================================
# ENTITY DETECTION
# ============================================================

KNOWN_ENTITIES = [
    "Python",
    "Java",
    "Apple",
    "Machine Learning",
    "Natural Language Processing",
    "Artificial Intelligence",
    "Data Science",
    "Deep Learning",
    "SQL",
    "HTML",
    "CSS",
    "JavaScript"
]


def detect_entities(query):

    found = []

    query_lower = query.lower()

    for entity in KNOWN_ENTITIES:

        if entity.lower() in query_lower:

            if entity not in found:
                found.append(entity)

    return found


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(query):

    q = query.lower()

    if any(word in q for word in [
        "learn",
        "study",
        "course",
        "tutorial",
        "beginner",
        "programming"
    ]):
        return "Learning / Educational Search"

    if any(word in q for word in [
        "information",
        "information about",
        "tell me",
        "details",
        "explain"
    ]):
        return "Information Seeking"

    if any(word in q for word in [
        "technology",
        "smartphone",
        "camera",
        "device"
    ]):
        return "Technology Search"

    if any(word in q for word in [
        "habitat",
        "animal",
        "snake",
        "wildlife"
    ]):
        return "General Knowledge Search"

    if any(word in q for word in [
        "nutrition",
        "health",
        "benefits",
        "food"
    ]):
        return "Health / Nutrition Search"

    return "General Search"


# ============================================================
# QUERY EXPANSION
# ============================================================

SEMANTIC_GROUPS = {

    "python": [
        "python",
        "programming",
        "coding",
        "beginner",
        "software"
    ],

    "machine": [
        "machine",
        "learning",
        "ml",
        "artificial",
        "intelligence"
    ],

    "natural": [
        "natural",
        "language",
        "processing",
        "nlp",
        "text"
    ],

    "java": [
        "java",
        "programming",
        "coding",
        "software"
    ],

    "apple": [
        "apple",
        "iphone",
        "smartphone",
        "technology",
        "nutrition",
        "fruit"
    ],

    "web": [
        "web",
        "html",
        "css",
        "javascript",
        "website"
    ],

    "data": [
        "data",
        "science",
        "analytics",
        "database",
        "information"
    ],

    "deep": [
        "deep",
        "learning",
        "neural",
        "network",
        "artificial"
    ],

    "sql": [
        "sql",
        "database",
        "management",
        "data"
    ]
}


def expand_query(query):

    words = preprocess(query).split()

    expanded = list(words)

    for word in words:

        if word in SEMANTIC_GROUPS:

            for related in SEMANTIC_GROUPS[word]:

                if related not in expanded:
                    expanded.append(related)

    return " ".join(expanded)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_documents(query, top_k=5):

    start_time = time.time()

    # ---------------------------------------------
    # Original TF-IDF search
    # ---------------------------------------------

    processed_query = preprocess(query)

    query_vector = vectorizer.transform(
        [processed_query]
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        document_matrix
    )[0]


    # ---------------------------------------------
    # Semantic-style expanded search
    # ---------------------------------------------

    expanded_query = expand_query(query)

    expanded_vector = vectorizer.transform(
        [expanded_query]
    )

    semantic_scores = cosine_similarity(
        expanded_vector,
        document_matrix
    )[0]


    # ---------------------------------------------
    # Hybrid score
    # ---------------------------------------------

    hybrid_scores = (
        0.4 * tfidf_scores +
        0.6 * semantic_scores
    )


    keywords = extract_keywords(query)
    entities = detect_entities(query)


    results = []

    for i in range(len(documents)):

        document = documents.iloc[i]

        matching_keywords = []

        document_words = preprocess(
            document["text"]
        ).split()

        for keyword in keywords:

            if keyword in document_words:

                matching_keywords.append(keyword)


        matching_entities = []

        for entity in entities:

            if entity.lower() in document["text"].lower():

                matching_entities.append(entity)


        explanation_parts = []

        if matching_keywords:

            explanation_parts.append(
                "Keyword match: " +
                ", ".join(matching_keywords[:5])
            )

        if matching_entities:

            explanation_parts.append(
                "Entity match: " +
                ", ".join(matching_entities)
            )

        if semantic_scores[i] > tfidf_scores[i]:

            explanation_parts.append(
                "Semantic expansion improved the match."
            )

        if not explanation_parts:

            explanation_parts.append(
                "Low similarity based on the current query."
            )


        results.append({

            "id": document["id"],

            "title": document["title"],

            "content": document["content"],

            "tfidf_score": round(
                float(tfidf_scores[i]), 4
            ),

            "semantic_score": round(
                float(semantic_scores[i]), 4
            ),

            "hybrid_score": round(
                float(hybrid_scores[i]), 4
            ),

            "matching_keywords":
                matching_keywords,

            "matching_entities":
                matching_entities,

            "explanation":
                " ".join(explanation_parts)
        })


    # ---------------------------------------------
    # Sort by hybrid score
    # ---------------------------------------------

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )


    execution_time = (
        time.time() - start_time
    ) * 1000


    return {
        "results": results[:top_k],

        "keywords": keywords,

        "entities": entities,

        "intent": detect_intent(query),

        "expanded_query": expanded_query,

        "execution_time": round(
            execution_time,
            2
        )
    }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# SEARCH API
# ============================================================

@app.route("/search", methods=["POST"])
def search():

    data = request.get_json()

    query = data.get(
        "query",
        ""
    ).strip()


    if not query:

        return jsonify({
            "error": "Please enter a search query."
        }), 400


    result = search_documents(
        query,
        top_k=5
    )


    return jsonify(result)


# ============================================================
# EVALUATION
# ============================================================

@app.route("/evaluate", methods=["POST"])
def evaluate():

    precision_values = []
    recall_values = []
    f1_values = []
    precision5_values = []
    reciprocal_rank_values = []
    response_times = []


    # =============================================
    # Evaluate every query
    # =============================================

    for _, row in queries.iterrows():

        query = row["query"]
        expected_id = row["expected_id"]


        start_time = time.time()


        search_result = search_documents(
            query,
            top_k=len(documents)
        )


        response_time = (
            time.time() - start_time
        ) * 1000


        ranked_results = search_result["results"]

        ranked_ids = [
            item["id"]
            for item in ranked_results
        ]


        # =========================================
        # Find correct result rank
        # =========================================

        if expected_id in ranked_ids:

            rank = (
                ranked_ids.index(expected_id)
                + 1
            )

        else:

            rank = 0


        # =========================================
        # Precision / Recall
        # =========================================

        if rank > 0:

            precision = 1.0
            recall = 1.0

        else:

            precision = 0.0
            recall = 0.0


        # =========================================
        # F1
        # =========================================

        if precision + recall > 0:

            f1 = (
                2 *
                precision *
                recall /
                (precision + recall)
            )

        else:

            f1 = 0.0


        # =========================================
        # Precision@5
        # =========================================

        top5_ids = ranked_ids[:5]

        if expected_id in top5_ids:

            precision5 = 1.0 / 5.0

        else:

            precision5 = 0.0


        # =========================================
        # MRR
        # =========================================

        if rank > 0:

            reciprocal_rank = 1.0 / rank

        else:

            reciprocal_rank = 0.0


        # =========================================
        # Store values
        # =========================================

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
            f1
        )

        precision5_values.append(
            precision5
        )

        reciprocal_rank_values.append(
            reciprocal_rank
        )

        response_times.append(
            response_time
        )


    # =============================================
    # Average metrics
    # =============================================

    precision = np.mean(
        precision_values
    )

    recall = np.mean(
        recall_values
    )

    f1 = np.mean(
        f1_values
    )

    precision5 = np.mean(
        precision5_values
    )

    mrr = np.mean(
        reciprocal_rank_values
    )

    response_time = np.mean(
        response_times
    )


    # =============================================
    # Evaluation graph
    # =============================================

    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "Precision@5",
        "MRR"
    ]


    metric_values = [
        precision,
        recall,
        f1,
        precision5,
        mrr
    ]


    # Make static folder
    os.makedirs(
        "static",
        exist_ok=True
    )


    graph_path = os.path.join(
        "static",
        "evaluation_metrics.png"
    )


    plt.figure(
        figsize=(10, 5)
    )


    plt.bar(
        metric_names,
        metric_values
    )


    plt.ylim(
        0,
        1.1
    )


    plt.ylabel(
        "Score"
    )


    plt.xlabel(
        "Evaluation Metric"
    )


    plt.title(
        "QueryMind-X Performance Evaluation"
    )


    for i, value in enumerate(
        metric_values
    ):

        plt.text(
            i,
            value + 0.03,
            f"{value:.2f}",
            ha="center"
        )


    plt.tight_layout()


    plt.savefig(
        graph_path,
        dpi=150
    )


    plt.close()


    # =============================================
    # Return JSON
    # =============================================

    return jsonify({

        "precision":
            round(float(precision), 4),

        "recall":
            round(float(recall), 4),

        "f1":
            round(float(f1), 4),

        "precision_at_5":
            round(float(precision5), 4),

        "mrr":
            round(float(mrr), 4),

        "response_time":
            round(float(response_time), 2),

        "total_queries":
            len(queries),

        "graph":
            "/static/evaluation_metrics.png"
    })


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("        QueryMind-X NLP Search System")
    print("==============================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("==============================================\n")


    app.run(
        debug=True
    )