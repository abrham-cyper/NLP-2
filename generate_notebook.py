"""
Generates NLP_Assignment.ipynb — a Google Colab notebook with
Tasks A, B, C + rich visualizations.
Run: python3 generate_notebook.py
"""
import json

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

cells = []

# ── TITLE ─────────────────────────────────────────────────────
cells.append(md(
"""# 🧠 NLP Course Assignment
## Tasks A · B · C — Corpus Analysis with Visualizations
> **Reuters Corpus** (10,788 news articles) &nbsp;|&nbsp; Python · NLTK · scikit-learn · matplotlib

---
| Task | What it does |
|------|-------------|
| 🔵 **A** | Cluster corpus into K groups using cosine similarity (K-Means) |
| 🟡 **B** | Score keywords with TF-IDF, classify as TOP / MEDIUM / BOTTOM |
| 🟢 **C** | Find documents similar to a query using cosine similarity |
"""))

# ── SETUP ─────────────────────────────────────────────────────
cells.append(code(
"""# ── INSTALL & IMPORTS ─────────────────────────────────────────
!pip install -q nltk scikit-learn matplotlib seaborn

import nltk
nltk.download('reuters', quiet=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import reuters
import warnings
warnings.filterwarnings('ignore')

# Dark GitHub-style theme
plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22',
    'text.color':'white','axes.labelcolor':'white',
    'xtick.color':'white','ytick.color':'white',
    'axes.edgecolor':'#30363d','grid.color':'#21262d',
    'axes.grid':True,'grid.alpha':0.3,
})
print('✅  Setup complete!')
"""))

# ── LOAD CORPUS ───────────────────────────────────────────────
cells.append(code(
r"""# ── LOAD REUTERS CORPUS + SHARED TF-IDF ──────────────────────
print('📚 Loading Reuters corpus...')
doc_ids   = reuters.fileids()
documents = [reuters.raw(fid) for fid in doc_ids]
print(f'  ✅  {len(documents):,} documents loaded')

print('\n🔧 Building shared TF-IDF model...')
VECTORIZER = TfidfVectorizer(
    max_features=10000, sublinear_tf=True, min_df=3,
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
TFIDF      = VECTORIZER.fit_transform(documents)
TFIDF_NORM = normalize(TFIDF, norm='l2')
FEATURES   = VECTORIZER.get_feature_names_out()
print(f'  ✅  Matrix shape: {TFIDF.shape}')
"""))

# ── TASK A HEADER ─────────────────────────────────────────────
cells.append(md(
"""---
## 🔵 Task A — Corpus Clustering
Groups all documents into **K clusters** by topic/vocabulary similarity using K-Means on normalized TF-IDF vectors (equivalent to cosine similarity clustering).

> ✏️ **Change `K` in the cell below to try different numbers of clusters.**
"""))

# ── TASK A CODE ───────────────────────────────────────────────
cells.append(code(
"""# ── TASK A: CLUSTERING ────────────────────────────────────────
K = 5        # ✏️ Change this!

print(f'🔵 Clustering {len(documents):,} docs into {K} groups...')
km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
km.fit(TFIDF_NORM)
labels = km.labels_

def top_terms(i, n=8):
    idx = km.cluster_centers_[i].argsort()[::-1][:n]
    return [FEATURES[j] for j in idx]

print(f'\\n  {"Cluster":<12} {"Docs":>6}   Top Keywords')
print('  ' + '─'*55)
for i in range(K):
    n   = (labels==i).sum()
    kws = ', '.join(top_terms(i, 6))
    print(f'  Cluster {i+1:<4}  {n:>6,}   {kws}')
print('\\n✅  Clustering done!')
"""))

# ── TASK A VISUALIZATION ──────────────────────────────────────
cells.append(code(
"""# ── TASK A: VISUALIZATIONS ────────────────────────────────────
COLORS = plt.cm.Set2(np.linspace(0, 1, K))
sizes  = [(labels==i).sum() for i in range(K)]

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#0d1117')
fig.suptitle(f'🔵 Task A — Corpus Clustering  (K={K})',
             fontsize=20, fontweight='bold', color='#58a6ff', y=1.01)

# 1. Bar chart of cluster sizes
ax1 = fig.add_subplot(2, 3, 1)
bars = ax1.barh([f'Cluster {i+1}' for i in range(K)], sizes, color=COLORS, edgecolor='none')
ax1.set_title('Cluster Sizes', color='#58a6ff', fontweight='bold')
ax1.set_xlabel('# Documents')
for b, s in zip(bars, sizes):
    ax1.text(b.get_width()+30, b.get_y()+b.get_height()/2, f'{s:,}', va='center', color='white', fontsize=9)

# 2. PCA 2-D scatter
ax2 = fig.add_subplot(2, 3, 2)
np.random.seed(42)
idx  = np.random.choice(len(documents), min(2000, len(documents)), replace=False)
pca  = PCA(n_components=2, random_state=42)
pts  = pca.fit_transform(TFIDF_NORM[idx].toarray())
for i in range(K):
    m = labels[idx]==i
    ax2.scatter(pts[m,0], pts[m,1], c=[COLORS[i]], label=f'C{i+1}', alpha=0.45, s=6)
ax2.set_title('PCA Cluster Map (2,000 sample)', color='#58a6ff', fontweight='bold')
ax2.legend(fontsize=8, markerscale=3, framealpha=0.3)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

# 3. Pie chart
ax3 = fig.add_subplot(2, 3, 3)
wedges,_,autos = ax3.pie(sizes, labels=[f'C{i+1}' for i in range(K)],
    colors=COLORS, autopct='%1.1f%%', startangle=90,
    textprops={'color':'white','fontsize':9},
    wedgeprops={'edgecolor':'#0d1117','linewidth':2})
[a.set_color('#0d1117') for a in autos]
ax3.set_title('Distribution', color='#58a6ff', fontweight='bold')

# 4-6. Top terms per cluster (first 3)
for i in range(min(K,3)):
    ax = fig.add_subplot(2, 3, 4+i)
    terms  = top_terms(i, 8)
    scores = [km.cluster_centers_[i][VECTORIZER.vocabulary_[t]] for t in terms]
    cmap   = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
    ax.barh(terms[::-1], scores[::-1], color=cmap[::-1], edgecolor='none')
    ax.set_title(f'Cluster {i+1} — Top Terms', color='#58a6ff', fontweight='bold', fontsize=10)
    ax.set_xlabel('Weight', fontsize=8)

plt.tight_layout()
plt.savefig('task_a_clusters.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('✅  Saved: task_a_clusters.png')
"""))

# ── TASK B HEADER ─────────────────────────────────────────────
cells.append(md(
"""---
## 🟡 Task B — Keyword TF-IDF Classification
Computes the **average TF-IDF score** for each keyword across the entire corpus, then classifies using the **10-80-10 percentile rule**.

| Class | Condition | Meaning |
|-------|-----------|---------|
| ⭐ **TOP** | score ≥ 90th percentile | Highly specific term |
| 📊 **MEDIUM** | 10th ≤ score < 90th | Average importance |
| 🔻 **BOTTOM** | score < 10th percentile | Very common / weak term |

> ✏️ **Edit `KEYWORDS` below. Press Enter to use `sample_keywords.txt`.**
"""))

# ── TASK B CODE ───────────────────────────────────────────────
cells.append(code(
r"""# ── TASK B: KEYWORD CLASSIFICATION ──────────────────────────
# ✏️ Edit or load from file:
KEYWORDS = ['oil','trade','market','dollar','bank',
            'government','price','stock','profit','export',
            'import','economy','debt','gold','energy']

# Optionally load from file:
# import os
# if os.path.exists('sample_keywords.txt'):
#     KEYWORDS = open('sample_keywords.txt').read().splitlines()

print('🟡 Computing TF-IDF for keywords...')
vec_b  = TfidfVectorizer(sublinear_tf=True, min_df=1,
                          token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat_b  = vec_b.fit_transform(documents)
voc_b  = vec_b.vocabulary_

def kw_score(w):
    w = w.lower().strip()
    if w not in voc_b: return 0.0
    col = mat_b.getcol(voc_b[w]).data
    return float(np.mean(col)) if len(col) else 0.0

scores_b = {kw: kw_score(kw) for kw in KEYWORDS}
vals_b   = list(scores_b.values())
P10, P90 = np.percentile(vals_b, 10), np.percentile(vals_b, 90)

def classify(s):
    return 'TOP' if s>=P90 else ('BOTTOM' if s<P10 else 'MEDIUM')

results_b = sorted([(kw, sc, classify(sc)) for kw,sc in scores_b.items()],
                   key=lambda x: x[1], reverse=True)

print(f'\n  10th pct: {P10:.4f}  |  90th pct: {P90:.4f}\n')
icons = {'TOP':'⭐','MEDIUM':'📊','BOTTOM':'🔻'}
for kw, sc, lbl in results_b:
    print(f'  {icons[lbl]} {kw:<15} {sc:.6f}   [{lbl}]')
print('\n✅  Task B done!')
"""))

# ── TASK B VISUALIZATION ──────────────────────────────────────
cells.append(code(
"""# ── TASK B: VISUALIZATIONS ────────────────────────────────────
CLR = {'TOP':'#FFD700','MEDIUM':'#4FC3F7','BOTTOM':'#EF5350'}
kw_names = [r[0] for r in results_b]
kw_vals  = [r[1] for r in results_b]
kw_cols  = [CLR[r[2]] for r in results_b]
kw_lbls  = [r[2] for r in results_b]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('🟡 Task B — Keyword TF-IDF Classification',
             fontsize=18, fontweight='bold', color='#FFD700')

# Bar chart
bars = ax1.barh(kw_names[::-1], kw_vals[::-1], color=kw_cols[::-1], edgecolor='none')
ax1.axvline(P10, color='#EF5350', ls='--', lw=1.5, label=f'10th pct ({P10:.4f})')
ax1.axvline(P90, color='#FFD700', ls='--', lw=1.5, label=f'90th pct ({P90:.4f})')
ax1.set_title('TF-IDF Score per Keyword', color='#FFD700', fontweight='bold')
ax1.set_xlabel('Mean TF-IDF Score')
patches = [mpatches.Patch(color=c, label=l) for l,c in CLR.items()]
ax1.legend(handles=patches + [
    plt.Line2D([0],[0], color='#EF5350', ls='--', label=f'10th ({P10:.4f})'),
    plt.Line2D([0],[0], color='#FFD700', ls='--', label=f'90th ({P90:.4f})')
], fontsize=8, loc='lower right', framealpha=0.3)
for b, v in zip(bars, kw_vals[::-1]):
    ax1.text(b.get_width()+0.001, b.get_y()+b.get_height()/2,
             f'{v:.4f}', va='center', color='white', fontsize=8)

# Donut chart
tops = kw_lbls.count('TOP')
meds = kw_lbls.count('MEDIUM')
bots = kw_lbls.count('BOTTOM')
wedges,_,autos = ax2.pie(
    [tops, meds, bots],
    labels=[f'⭐ TOP\\n({tops})', f'📊 MEDIUM\\n({meds})', f'🔻 BOTTOM\\n({bots})'],
    colors=['#FFD700','#4FC3F7','#EF5350'],
    autopct='%1.0f%%', startangle=90,
    textprops={'color':'white','fontsize':11},
    wedgeprops={'edgecolor':'#0d1117','linewidth':2,'width':0.6})
[a.set_color('#0d1117') for a in autos]
ax2.set_title('10 – 80 – 10 Distribution', color='#FFD700', fontweight='bold')

plt.tight_layout()
plt.savefig('task_b_keywords.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('✅  Saved: task_b_keywords.png')
"""))

# ── TASK C HEADER ─────────────────────────────────────────────
cells.append(md(
"""---
## 🟢 Task C — Document Similarity Search
Given a query document and a percentile threshold, returns all corpus documents **above that percentile** in cosine similarity.

> ⚠️ **No stopword removal** is applied, as required by the assignment.
>
> ✏️ **Edit `USER_DOC` and `PERCENTILE` below.**
"""))

# ── TASK C CODE ───────────────────────────────────────────────
TASK_C_SRC = (
'# ── TASK C: SIMILARITY SEARCH ────────────────────────────\n'
'USER_DOC = (\n'
'    "Oil prices surged today as OPEC agreed to cut production significantly. "\n'
'    "The crude market saw strong gains and the dollar weakened. "\n'
'    "Energy stocks rose sharply on the news."\n'
')\n'
'PERCENTILE = 80  # ✏️ Change: 80 = return top 20% most similar\n'
'\n'
'print("🟢 Task C: Document Similarity Search")\n'
r'print(f"  Query: \"{USER_DOC[:70]}...\"")' + '\n'
r'print(f"  Threshold: {PERCENTILE}th percentile\n")' + '\n'
'\n'
'# Build TF-IDF without stopword removal (per assignment spec)\n'
r"vec_c = TfidfVectorizer(sublinear_tf=True, min_df=2, token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')" + '\n'
'mat_c = vec_c.fit_transform(documents)\n'
'uvec  = vec_c.transform([USER_DOC])\n'
'sims  = cosine_similarity(uvec, mat_c).flatten()\n'
'\n'
'threshold = np.percentile(sims, PERCENTILE)\n'
'match_idx = np.where(sims >= threshold)[0]\n'
'results_c = sorted([(doc_ids[i], float(sims[i])) for i in match_idx],\n'
'                   key=lambda x: x[1], reverse=True)\n'
'ALL_SIMS  = sims\n'
'\n'
r'print(f"  Threshold : {threshold:.6f}")' + '\n'
r'print(f"  Matches   : {len(results_c):,}\n")' + '\n'
r'print(f"  {chr(82)+"ANK":<6} {chr(68)+"OCUMENT ID":<35} {chr(83)+"IMILARITY":>12}")' + '\n'
'print("  " + "─"*55)\n'
'for rank,(did,sc) in enumerate(results_c[:15], 1):\n'
r'    print(f"  {rank:<6} {did:<35} {sc:>12.6f}")' + '\n'
'if len(results_c)>15:\n'
r'    print(f"  ... +{len(results_c)-15} more")' + '\n'
'print("\\n✅  Task C done!")\n'
)
cells.append(code(TASK_C_SRC))

# ── TASK C VISUALIZATION ──────────────────────────────────────
cells.append(code(
"""# ── TASK C: VISUALIZATIONS ────────────────────────────────────
top_n     = min(20, len(results_c))
top_docs  = results_c[:top_n]
top_scores= [s for _,s in top_docs]
top_labels= [d.split('/')[-1] for d,_ in top_docs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('🟢 Task C — Document Similarity Search',
             fontsize=18, fontweight='bold', color='#3fb950')

# Histogram of ALL similarity scores
ax1.hist(ALL_SIMS, bins=80, color='#4FC3F7', alpha=0.75, edgecolor='none', label='All docs')
ylim = ax1.get_ylim()
ax1.axvline(threshold, color='#FF6B6B', lw=2.5,
            label=f'{PERCENTILE}th pct = {threshold:.4f}')
ax1.fill_betweenx([0, ylim[1]], threshold, ALL_SIMS.max(),
                   alpha=0.2, color='#3fb950', label=f'{len(results_c):,} matches')
ax1.set_ylim(ylim)
ax1.set_title('Cosine Similarity Distribution', color='#3fb950', fontweight='bold')
ax1.set_xlabel('Cosine Similarity Score')
ax1.set_ylabel('# Documents')
ax1.legend(fontsize=9, framealpha=0.3)

# Top-N matches bar chart
grad = plt.cm.YlGn(np.linspace(0.4, 1.0, top_n))
bars = ax2.barh(top_labels[::-1], top_scores[::-1], color=grad, edgecolor='none')
ax2.set_title(f'Top {top_n} Most Similar Documents', color='#3fb950', fontweight='bold')
ax2.set_xlabel('Cosine Similarity Score')
for b, sc in zip(bars, top_scores[::-1]):
    ax2.text(b.get_width()+0.001, b.get_y()+b.get_height()/2,
             f'{sc:.4f}', va='center', color='white', fontsize=7)

plt.tight_layout()
plt.savefig('task_c_similarity.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('✅  Saved: task_c_similarity.png')
"""))

# ── SUMMARY ───────────────────────────────────────────────────
cells.append(md(
"""---
## ✅ Summary

| Task | Method | Corpus | Output |
|------|--------|--------|--------|
| 🔵 A | K-Means on L2-normalized TF-IDF | Reuters 10,788 docs | K clusters with top terms + PCA map |
| 🟡 B | Mean TF-IDF score per keyword | Reuters 10,788 docs | TOP / MEDIUM / BOTTOM classification |
| 🟢 C | Cosine similarity (no stopword removal) | Reuters 10,788 docs | Ranked list of similar documents |

> 📁 Plots saved: `task_a_clusters.png`, `task_b_keywords.png`, `task_c_similarity.png`
"""))

# ── WRITE NOTEBOOK ────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "NLP_Assignment.ipynb"},
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.8.0"}
    },
    "cells": cells
}

with open('NLP_Assignment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("✅  NLP_Assignment.ipynb created successfully!")
