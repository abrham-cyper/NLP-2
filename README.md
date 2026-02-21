# NLP-2

> **NLP Course Assignment** — General Tasks A, B & C  
> Python + NLTK + scikit-learn on the Reuters Corpus (10,788 news articles)

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `task_a_clustering.py` | 🔵 Task A — Cluster corpus into K groups (K-Means + Cosine Similarity) |
| `task_b_keyword_tfidf.py` | 🟡 Task B — Classify keywords by TF-IDF score (TOP / MEDIUM / BOTTOM) |
| `task_c_similarity_search.py` | 🟢 Task C — Find similar documents using Cosine Similarity |
| `NLP_Assignment.ipynb` | 📓 Google Colab notebook with all tasks + visualizations |
| `sample_keywords.txt` | Sample keywords for Task B |
| `requirements.txt` | Python dependencies |

---

## 🚀 How to Run

### Option 1 — Local (Terminal)
```bash
pip3 install -r requirements.txt

python3 task_a_clustering.py        # Enter K (e.g. 5)
python3 task_b_keyword_tfidf.py     # Press Enter for sample_keywords.txt
python3 task_c_similarity_search.py # Paste document + enter percentile
```

### Option 2 — Google Colab
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `NLP_Assignment.ipynb`
3. Click **Runtime → Run All**

---

## 📋 Task Descriptions

### 🔵 Task A — Corpus Clustering
Clusters all Reuters corpus documents into **K groups** based on cosine similarity.  
Uses TF-IDF vectorization + L2 normalization + K-Means clustering.

### 🟡 Task B — Keyword TF-IDF Classification
Reads a keyword file and assigns each word a TF-IDF score, then classifies using:
- ⭐ **TOP** — score ≥ 90th percentile
- 📊 **MEDIUM** — 10th ≤ score < 90th percentile
- 🔻 **BOTTOM** — score < 10th percentile

### 🟢 Task C — Document Similarity Search
Given a document and a percentile threshold, returns all corpus documents  
with cosine similarity above the threshold. No stopword removal (per spec).

---

## 🛠️ Tech Stack
- **Python 3** · **NLTK** · **scikit-learn** · **NumPy** · **Matplotlib**
- **Corpus:** `nltk.corpus.reuters` (10,788 news articles)
