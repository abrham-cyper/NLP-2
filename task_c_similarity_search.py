"""
NLP Assignment - Task C: Document Similarity Search
====================================================
Given a user-provided document and a match percentile threshold:
1. Builds a TF-IDF matrix of the Reuters corpus (NO stopword removal)
2. Transforms the user document using the same TF-IDF model
3. Computes cosine similarity between user doc and all corpus docs
4. Returns all corpus documents whose similarity score is above
   the user-specified percentile threshold
"""

import sys
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    return doc_ids, documents


def build_tfidf_model(documents):
    """
    Build a TF-IDF model.
    NOTE: As per assignment instructions, NO stopword elimination is applied.
    """
    print("\nBuilding TF-IDF matrix (no stopword removal)...")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
        # stop_words is intentionally NOT set
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    print(f"  → TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def get_user_document():
    """Prompt user to enter a document (multi-line input)."""
    print("\n" + "-" * 60)
    print("Enter your document text below.")
    print("Type/paste your text, then press Enter twice to finish:")
    print("-" * 60)
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
                lines.append(line)
        except EOFError:
            # Handle piped input: all text up to EOF is the document
            break
    document = " ".join(lines).strip()
    if not document:
        print("\n  ERROR: No document text entered.")
        sys.exit(1)
    return document


def get_percentile():
    """Prompt user to enter a match percentile threshold."""
    while True:
        try:
            raw = input(
                "\nEnter the match percentile threshold (0-100).\n"
                "  Example: 80 → return docs in top 20% similarity: "
            ).strip()
            pct = float(raw)
            if 0 <= pct <= 100:
                return pct
            else:
                print("  Please enter a value between 0 and 100.")
        except EOFError:
            print("  No input received. Using default percentile: 80")
            return 80.0
        except ValueError:
            print("  Invalid input. Please enter a number.")


def find_similar_documents(user_doc, tfidf_matrix, vectorizer, doc_ids, percentile):
    """
    Compute cosine similarity between user_doc and all corpus documents,
    then return those above the specified percentile threshold.
    """
    print("\nComputing cosine similarities...")
    user_vector = vectorizer.transform([user_doc])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Compute the percentile threshold on the similarity scores
    threshold = np.percentile(similarities, percentile)
    print(f"  → Similarity score threshold at {percentile}th percentile: {threshold:.6f}")

    # Filter docs above threshold
    matching_indices = np.where(similarities >= threshold)[0]
    results = [
        (doc_ids[i], float(similarities[i]))
        for i in matching_indices
    ]
    # Sort by similarity score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results, threshold


def print_results(results, percentile, threshold, max_shown=30):
    """Print the matching document results."""
    print("\n" + "=" * 65)
    print("  DOCUMENT SIMILARITY SEARCH RESULTS")
    print("=" * 65)
    print(f"  Percentile threshold : {percentile}%  (score >= {threshold:.6f})")
    print(f"  Matching documents   : {len(results)}")
    print("=" * 65)

    if not results:
        print("\n  No documents found above the specified threshold.")
    else:
        print(f"  {'RANK':<6} {'DOCUMENT ID':<35} {'SIMILARITY SCORE':>16}")
        print("-" * 65)
        display = results[:max_shown]
        for rank, (doc_id, score) in enumerate(display, 1):
            print(f"  {rank:<6} {doc_id:<35} {score:>16.6f}")

        if len(results) > max_shown:
            print(f"\n  ... and {len(results) - max_shown} more documents above threshold.")

    print("=" * 65)


def main():
    print("=" * 65)
    print("  NLP ASSIGNMENT — TASK C: DOCUMENT SIMILARITY SEARCH")
    print("=" * 65)

    doc_ids, documents = load_corpus()
    tfidf_matrix, vectorizer = build_tfidf_model(documents)

    user_doc = get_user_document()
    percentile = get_percentile()

    results, threshold = find_similar_documents(
        user_doc, tfidf_matrix, vectorizer, doc_ids, percentile
    )

    print_results(results, percentile, threshold)
    print("\nTask C complete!")


if __name__ == "__main__":
    main()
