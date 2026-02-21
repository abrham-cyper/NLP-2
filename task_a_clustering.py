"""
NLP Assignment - Task A: Corpus Clustering using Cosine Similarity
==================================================================
Clusters the Reuters corpus into a specified number of classes using:
- TF-IDF vectorization
- L2-normalized vectors (cosine similarity equivalent)
- K-Means clustering
"""

import sys
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Download required NLTK data
print("Downloading NLTK data (if not already downloaded)...")
nltk.download('reuters', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import reuters


def load_corpus():
    """Load all Reuters documents as raw text strings."""
    print("\nLoading Reuters corpus...")
    doc_ids = reuters.fileids()
    documents = []
    for doc_id in doc_ids:
        raw = reuters.raw(doc_id)
        documents.append(raw)
    print(f"  → Loaded {len(documents)} documents.")
    return doc_ids, documents


def build_tfidf_matrix(documents):
    """Build and L2-normalize a TF-IDF matrix from documents."""
    print("\nBuilding TF-IDF matrix...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        sublinear_tf=True,       # use log(tf) + 1 for smoother scaling
        min_df=3,                 # ignore very rare terms
        strip_accents='unicode',
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    # L2-normalize so dot product = cosine similarity
    tfidf_normalized = normalize(tfidf_matrix, norm='l2')
    print(f"  → TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_normalized, vectorizer


def cluster_corpus(tfidf_matrix, k):
    """Run K-Means clustering on normalized TF-IDF vectors."""
    print(f"\nClustering into {k} clusters (this may take a moment)...")
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    kmeans.fit(tfidf_matrix)
    return kmeans


def get_top_terms_per_cluster(kmeans, vectorizer, n_terms=10):
    """Return the top N terms for each cluster centroid."""
    feature_names = vectorizer.get_feature_names_out()
    clusters_terms = {}
    for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
        top_indices = centroid.argsort()[::-1][:n_terms]
        top_terms = [feature_names[i] for i in top_indices]
        clusters_terms[cluster_id] = top_terms
    return clusters_terms


def print_results(doc_ids, labels, top_terms, k, max_docs_shown=5):
    """Print cluster summary results."""
    print("\n" + "=" * 60)
    print(f"  CLUSTERING RESULTS  (K = {k})")
    print("=" * 60)

    for cluster_id in range(k):
        cluster_doc_ids = [doc_ids[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        print(f"\n📁 CLUSTER {cluster_id + 1}  ({len(cluster_doc_ids)} documents)")
        print(f"   Top terms : {', '.join(top_terms[cluster_id])}")
        shown = cluster_doc_ids[:max_docs_shown]
        print(f"   Sample docs: {', '.join(shown)}", end="")
        if len(cluster_doc_ids) > max_docs_shown:
            print(f"  ... (+{len(cluster_doc_ids) - max_docs_shown} more)")
        else:
            print()

    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("  NLP ASSIGNMENT — TASK A: CORPUS CLUSTERING")
    print("=" * 60)

    # Get K from user
    while True:
        try:
            k = int(input("\nEnter the number of clusters (K): ").strip())
            if k < 2:
                print("  Please enter a number >= 2.")
                continue
            break
        except ValueError:
            print("  Invalid input. Please enter an integer.")

    # Pipeline
    doc_ids, documents = load_corpus()
    tfidf_matrix, vectorizer = build_tfidf_matrix(documents)
    kmeans = cluster_corpus(tfidf_matrix, k)
    top_terms = get_top_terms_per_cluster(kmeans, vectorizer)
    print_results(doc_ids, kmeans.labels_, top_terms, k)

    print("\nTask A complete!")


if __name__ == "__main__":
    main()
