"""
NLP Assignment - Task B: Keyword TF-IDF Classification
=======================================================
Accepts a file of keywords and:
1. Computes the TF-IDF score for each keyword across the Reuters corpus
2. Classifies keywords into three classes using the 10-80-10 percentile rule:
   - TOP    (top 10%):    score >= 90th percentile
   - MEDIUM (middle 80%): 10th percentile <= score < 90th percentile
   - BOTTOM (bottom 10%): score < 10th percentile
"""

import os
import sys
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
print("Downloading NLTK data (if not already downloaded)...")
nltk.download('reuters', quiet=True)

from nltk.corpus import reuters


def load_corpus():
    """Load all Reuters documents as raw text strings."""
    print("\nLoading Reuters corpus...")
    doc_ids = reuters.fileids()
    documents = [reuters.raw(fid) for fid in doc_ids]
    print(f"  → Loaded {len(documents)} documents.")
    return documents


def build_tfidf_model(documents):
    """Build TF-IDF model on all corpus documents."""
    print("\nBuilding TF-IDF model...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=1,
        strip_accents='unicode',
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"  → Vocabulary size: {len(vectorizer.vocabulary_)} terms")
    print(f"  → TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def get_keyword_score(keyword, tfidf_matrix, vectorizer):
    """
    Compute the representative TF-IDF score for a keyword.
    Uses the mean of non-zero TF-IDF values across all documents
    (i.e., average score in the documents where the keyword appears).
    Returns 0 if keyword is not in vocabulary.
    """
    vocab = vectorizer.vocabulary_
    kw = keyword.lower().strip()
    if kw not in vocab:
        return 0.0
    col_index = vocab[kw]
    col = tfidf_matrix.getcol(col_index)
    nonzero_vals = col.data   # only non-zero entries
    if len(nonzero_vals) == 0:
        return 0.0
    return float(np.mean(nonzero_vals))


def load_keywords(filepath):
    """Read keywords from a file (one per line)."""
    if not os.path.isfile(filepath):
        print(f"\n  ERROR: File not found: '{filepath}'")
        sys.exit(1)
    with open(filepath, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]
    if not keywords:
        print("\n  ERROR: Keywords file is empty.")
        sys.exit(1)
    print(f"\n  → Loaded {len(keywords)} keywords from '{filepath}'")
    return keywords


def classify_keywords(keywords, scores):
    """
    Classify keywords using the 10-80-10 percentile distribution.
    Returns a list of (keyword, score, label) tuples.
    """
    score_values = list(scores.values())
    p10 = np.percentile(score_values, 10)
    p90 = np.percentile(score_values, 90)

    results = []
    for kw in keywords:
        score = scores[kw]
        if score >= p90:
            label = "TOP"
        elif score < p10:
            label = "BOTTOM"
        else:
            label = "MEDIUM"
        results.append((kw, score, label))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results, p10, p90


def print_results(results, p10, p90):
    """Print a formatted results table."""
    label_colors = {
        "TOP":    "⭐ TOP   ",
        "MEDIUM": "📊 MEDIUM",
        "BOTTOM": "🔻 BOTTOM",
    }

    print("\n" + "=" * 65)
    print("  KEYWORD TF-IDF CLASSIFICATION RESULTS")
    print("=" * 65)
    print(f"  Percentile thresholds  →  10th: {p10:.6f}  |  90th: {p90:.6f}")
    print("=" * 65)
    print(f"  {'KEYWORD':<20} {'TF-IDF SCORE':>14}   CLASS")
    print("-" * 65)

    for keyword, score, label in results:
        tag = label_colors[label]
        status = "(not in vocab)" if score == 0.0 else ""
        print(f"  {keyword:<20} {score:>14.6f}   {tag}  {status}")

    print("=" * 65)

    # Summary counts
    tops    = sum(1 for _, _, l in results if l == "TOP")
    middles = sum(1 for _, _, l in results if l == "MEDIUM")
    bottoms = sum(1 for _, _, l in results if l == "BOTTOM")
    print(f"\n  Summary: ⭐ TOP: {tops}  |  📊 MEDIUM: {middles}  |  🔻 BOTTOM: {bottoms}")
    print("=" * 65)


def main():
    print("=" * 65)
    print("  NLP ASSIGNMENT — TASK B: KEYWORD TF-IDF CLASSIFICATION")
    print("=" * 65)

    # Get keywords file path from user
    filepath = input(
        "\nEnter path to keywords file (one keyword per line)\n"
        "[Press Enter to use 'sample_keywords.txt']: "
    ).strip()
    if not filepath:
        filepath = "sample_keywords.txt"

    keywords = load_keywords(filepath)
    documents = load_corpus()
    tfidf_matrix, vectorizer = build_tfidf_model(documents)

    # Compute TF-IDF score for each keyword
    print("\nComputing TF-IDF scores for keywords...")
    scores = {}
    for kw in keywords:
        scores[kw] = get_keyword_score(kw, tfidf_matrix, vectorizer)

    results, p10, p90 = classify_keywords(keywords, scores)
    print_results(results, p10, p90)
    print("\nTask B complete!")


if __name__ == "__main__":
    main()
