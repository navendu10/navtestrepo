"""Precision@K / Recall@K evaluation for TF-IDF document retrieval on 20-newsgroups."""

import re
import string

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

newsgroups_train = fetch_20newsgroups(
    subset="train", remove=("headers", "footers", "quotes")
)

df = pd.DataFrame(
    {
        "text": newsgroups_train.data,
        "label": newsgroups_train.target,
    }
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def preprocess_text(text):
    """Lowercase, strip punctuation/digits, and collapse whitespace."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def top_k_greatest_indices(scores, k, exclude_index=None):
    """Return indices of the k largest values in `scores`, in descending order."""
    scores = np.asarray(scores, dtype=float)
    if exclude_index is not None:
        scores = scores.copy()
        scores[exclude_index] = -np.inf

    k = max(0, min(k, len(scores)))
    if k == 0:
        return np.array([], dtype=int)

    top_k_unsorted = np.argpartition(scores, -k)[-k:]
    return top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]


def precision_at_k(retrieved_labels, relevant_label, k):
    """Fraction of the top-k retrieved items that share the relevant label."""
    retrieved_labels = retrieved_labels[:k]
    if len(retrieved_labels) == 0:
        return 0.0
    num_relevant = sum(1 for label in retrieved_labels if label == relevant_label)
    return num_relevant / len(retrieved_labels)


def recall_at_k(retrieved_labels, relevant_label, total_relevant, k):
    """Fraction of all relevant items that appear in the top-k retrieved items."""
    if total_relevant == 0:
        return 0.0
    retrieved_labels = retrieved_labels[:k]
    num_relevant = sum(1 for label in retrieved_labels if label == relevant_label)
    return num_relevant / total_relevant


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(query_index, tfidf_matrix, df, k=10):
    """Compute Precision@K and Recall@K for a single query document.

    query_index  : row position of the query document in `df` / `tfidf_matrix`.
    tfidf_matrix : TF-IDF matrix aligned row-for-row with `df`.
    df           : DataFrame with a 'label' column for each document.
    k            : number of top results to evaluate.
    """
    query_vector = tfidf_matrix[query_index]
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_k_idx = top_k_greatest_indices(similarities, k, exclude_index=query_index)

    query_label = df.iloc[query_index]["label"]
    retrieved_labels = df.iloc[top_k_idx]["label"].tolist()

    # Exclude the query document itself from the relevant-document count.
    total_relevant = int((df["label"] == query_label).sum()) - 1

    precision = precision_at_k(retrieved_labels, query_label, k)
    recall = recall_at_k(retrieved_labels, query_label, total_relevant, k)

    return {"precision@k": precision, "recall@k": recall}


if __name__ == "__main__":
    df["clean_text"] = df["text"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])

    metrics = compute_metrics(query_index=0, tfidf_matrix=tfidf_matrix, df=df, k=10)
    print(metrics)
