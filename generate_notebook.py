"""
generate_notebook_v2.py — Enhanced NLP Assignment Colab Notebook
New features: Word Clouds, t-SNE, Category Alignment, Similarity Heatmap
Run: python3 generate_notebook_v2.py
"""
import json

def md(src): return {"cell_type":"markdown","metadata":{},"source":src}
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

cells = []

# ── TITLE ─────────────────────────────────────────────────────
cells.append(md("""# 🧠 NLP Course Assignment — Enhanced Edition
## Tasks A · B · C with Advanced Visualizations
> **Reuters Corpus** (10,788 articles) | Python · NLTK · scikit-learn · WordCloud · Seaborn

| Task | Method | Visuals |
|------|--------|---------|
| 🔵 **A** | K-Means + Cosine Similarity Clustering | Word clouds, t-SNE map, Category alignment |
| 🟡 **B** | TF-IDF Keyword Classification (10-80-10) | Bar chart, Donut, Bubble chart |
| 🟢 **C** | Document Cosine Similarity Search | Score histogram, Top-N bar, Heatmap |
"""))

# ── SETUP ─────────────────────────────────────────────────────
cells.append(code(
"""!pip install -q nltk scikit-learn matplotlib seaborn wordcloud

import nltk
nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import reuters, stopwords
from wordcloud import WordCloud
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── PREMIUM DARK THEME ─────────────────────────────────────────
BG   = '#0d1117'
BG2  = '#161b22'
EDGE = '#30363d'
plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor':   BG2,
    'text.color':   'white',  'axes.labelcolor':  'white',
    'xtick.color':  'white',  'ytick.color':      'white',
    'axes.edgecolor': EDGE,   'grid.color':       '#21262d',
    'axes.grid': True,        'grid.alpha':       0.25,
    'legend.facecolor': BG2,  'legend.edgecolor': EDGE,
})
STOP = set(stopwords.words('english'))
print('✅  Setup complete!')
"""))

# ── LOAD CORPUS ───────────────────────────────────────────────
cells.append(code(r"""
print('📚 Loading Reuters corpus...')
doc_ids   = reuters.fileids()
documents = [reuters.raw(fid) for fid in doc_ids]
# Reuters categories per document
doc_cats  = {fid: reuters.categories(fid) for fid in doc_ids}
ALL_CATS  = sorted(set(c for cats in doc_cats.values() for c in cats))
print(f'  ✅  {len(documents):,} documents  |  {len(ALL_CATS)} categories')

print('\n🔧 Building TF-IDF model...')
VECTORIZER = TfidfVectorizer(
    max_features=10000, sublinear_tf=True, min_df=3,
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
TFIDF      = VECTORIZER.fit_transform(documents)
TFIDF_NORM = normalize(TFIDF, norm='l2')
FEATURES   = VECTORIZER.get_feature_names_out()
print(f'  ✅  Matrix: {TFIDF.shape}')
"""))

# ── TASK A HEADER ─────────────────────────────────────────────
cells.append(md("""---
## 🔵 Task A — Corpus Clustering
K-Means on L2-normalized TF-IDF vectors = cosine similarity clustering.

> ✏️ **Change `K` below.**
"""))

# ── TASK A CLUSTERING ─────────────────────────────────────────
cells.append(code("""
K = 5   # ✏️  Change this!

print(f'🔵 Clustering {len(documents):,} docs into {K} groups...')
km     = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
km.fit(TFIDF_NORM)
labels = km.labels_
COLORS = plt.cm.Set2(np.linspace(0, 1, K))

def top_terms(i, n=15):
    idx = km.cluster_centers_[i].argsort()[::-1][:n]
    return [FEATURES[j] for j in idx]

print(f'\\n  {"Cluster":<12} {"Docs":>6}   Top Keywords')
print('  ' + '─'*60)
for i in range(K):
    n   = (labels==i).sum()
    kws = ', '.join(top_terms(i, 6))
    print(f'  Cluster {i+1:<4}  {n:>6,}   {kws}')
print('\\n✅  Clustering done!')
"""))

# ── TASK A: WORD CLOUDS ───────────────────────────────────────
cells.append(code("""
# ═══ WORD CLOUDS PER CLUSTER ══════════════════════════════════
print('☁️  Generating word clouds...')
cols = min(K, 3)
rows = (K + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
fig.patch.set_facecolor(BG)
fig.suptitle('🔵 Task A — Word Clouds per Cluster',
             fontsize=18, fontweight='bold', color='#58a6ff', y=1.01)
axes = axes.flatten() if K > 1 else [axes]

PALETTES = ['Blues','Purples','Greens','Oranges','RdPu',
            'YlOrRd','BuGn','BuPu','copper','winter']

for i in range(K):
    # Build frequency dict from centroid weights
    freq = {FEATURES[j]: float(km.cluster_centers_[i][j])
            for j in km.cluster_centers_[i].argsort()[::-1][:100]}
    wc = WordCloud(width=600, height=400,
                   background_color='#161b22',
                   colormap=PALETTES[i % len(PALETTES)],
                   max_words=60, prefer_horizontal=0.85,
                   min_font_size=10).generate_from_frequencies(freq)
    axes[i].imshow(wc, interpolation='bilinear')
    axes[i].axis('off')
    n = (labels==i).sum()
    axes[i].set_title(f'Cluster {i+1}  ({n:,} docs)',
                      color='#58a6ff', fontweight='bold', fontsize=12, pad=8)

for j in range(K, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('task_a_wordclouds.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print('✅  Saved: task_a_wordclouds.png')
"""))

# ── TASK A: t-SNE + CATEGORY ALIGNMENT ────────────────────────
cells.append(code("""
# ═══ t-SNE CLUSTER MAP + CATEGORY ALIGNMENT ═══════════════════
np.random.seed(42)
N_SAMPLE = 1500
idx    = np.random.choice(len(documents), N_SAMPLE, replace=False)
sub    = TFIDF_NORM[idx]

print(f'⏳ Computing t-SNE on {N_SAMPLE} sample docs (takes ~60s)...')
tsne   = TSNE(n_components=2, perplexity=40, n_iter=800,
               random_state=42, init='pca')
pts    = tsne.fit_transform(sub.toarray())
slbls  = labels[idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)
fig.suptitle('🔵 Task A — Cluster Maps', fontsize=18,
             fontweight='bold', color='#58a6ff')

# t-SNE scatter
for i in range(K):
    m = slbls == i
    ax1.scatter(pts[m,0], pts[m,1], c=[COLORS[i]],
                label=f'C{i+1} ({m.sum()})', alpha=0.55, s=10, edgecolors='none')
ax1.set_title(f't-SNE Map ({N_SAMPLE} sample)', color='#58a6ff', fontweight='bold')
ax1.legend(fontsize=9, markerscale=2, framealpha=0.4)
ax1.set_xlabel('t-SNE dim 1')
ax1.set_ylabel('t-SNE dim 2')

# Category alignment heatmap
top_cats = [c for c,_ in Counter(
    c for fid in doc_ids for c in doc_cats[fid]
).most_common(10)]
align = np.zeros((K, len(top_cats)))
for i, fid in enumerate(doc_ids):
    cl  = labels[i]
    for c in doc_cats[fid]:
        if c in top_cats:
            align[cl, top_cats.index(c)] += 1
align_norm = align / (align.sum(axis=1, keepdims=True) + 1e-9)

sns.heatmap(align_norm,
            xticklabels=top_cats,
            yticklabels=[f'Cluster {i+1}' for i in range(K)],
            cmap='Blues', ax=ax2, annot=True, fmt='.2f',
            linewidths=0.5, linecolor=EDGE, cbar_kws={'label':'Proportion'})
ax2.set_title('Category Alignment per Cluster', color='#58a6ff', fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('task_a_tsne.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print('✅  Saved: task_a_tsne.png')
"""))

# ── TASK B HEADER ─────────────────────────────────────────────
cells.append(md("""---
## 🟡 Task B — Keyword TF-IDF Classification
**10-80-10 percentile rule** → ⭐ TOP / 📊 MEDIUM / 🔻 BOTTOM

> ✏️ **Edit `KEYWORDS` below.**
"""))

# ── TASK B CODE + VISUALIZATIONS ─────────────────────────────
cells.append(code(r"""
# ─── TASK B ───────────────────────────────────────────────────
KEYWORDS = ['oil','trade','market','dollar','bank',
            'government','price','stock','profit','export',
            'import','economy','debt','gold','energy',
            'merger','tax','rate','inflation','currency']

print('🟡 Computing TF-IDF scores...')
vec_b  = TfidfVectorizer(sublinear_tf=True, min_df=1,
                          token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat_b  = vec_b.fit_transform(documents)
voc_b  = vec_b.vocabulary_

def kw_score(w):
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

CLR  = {'TOP':'#FFD700','MEDIUM':'#4FC3F7','BOTTOM':'#EF5350'}
kw_names = [r[0] for r in results_b]
kw_vals  = [r[1] for r in results_b]
kw_cols  = [CLR[r[2]] for r in results_b]
kw_lbls  = [r[2] for r in results_b]

print(f'\n  P10={P10:.4f}  |  P90={P90:.4f}\n')
icons = {'TOP':'⭐','MEDIUM':'📊','BOTTOM':'🔻'}
for kw,sc,lbl in results_b:
    print(f'  {icons[lbl]} {kw:<15} {sc:.6f}   [{lbl}]')
print('\n✅  Task B done!')
"""))

cells.append(code("""
# ═══ TASK B VISUALIZATIONS ════════════════════════════════════
fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor(BG)
fig.suptitle('🟡 Task B — Keyword TF-IDF Classification',
             fontsize=18, fontweight='bold', color='#FFD700')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# 1. Horizontal bar chart
ax1 = fig.add_subplot(gs[0, :2])
bars = ax1.barh(kw_names[::-1], kw_vals[::-1], color=kw_cols[::-1], edgecolor='none', height=0.7)
ax1.axvline(P10, color='#EF5350', ls='--', lw=1.5, label=f'10th pct ({P10:.4f})')
ax1.axvline(P90, color='#FFD700', ls='--', lw=1.5, label=f'90th pct ({P90:.4f})')
ax1.set_title('TF-IDF Score per Keyword', color='#FFD700', fontweight='bold')
ax1.set_xlabel('Mean TF-IDF Score')
patches = [mpatches.Patch(color=c, label=l) for l,c in CLR.items()]
ax1.legend(handles=patches + [
    plt.Line2D([0],[0], color='#EF5350', ls='--', label=f'10th ({P10:.4f})'),
    plt.Line2D([0],[0], color='#FFD700', ls='--', label=f'90th ({P90:.4f})')
], fontsize=8, framealpha=0.3)
for b,v in zip(bars, kw_vals[::-1]):
    ax1.text(b.get_width()+0.0005, b.get_y()+b.get_height()/2,
             f'{v:.4f}', va='center', color='white', fontsize=7)

# 2. Bubble chart — size = TF-IDF score
ax2 = fig.add_subplot(gs[0, 2])
xs   = np.random.uniform(0.1, 0.9, len(kw_names))
ys   = np.random.uniform(0.1, 0.9, len(kw_names))
szs  = [v*5000 for v in kw_vals]
cbls = [CLR[l] for l in kw_lbls]
ax2.scatter(xs, ys, s=szs, c=cbls, alpha=0.7, edgecolors='white', linewidth=0.5)
for x, y, kw, v in zip(xs, ys, kw_names, kw_vals):
    ax2.text(x, y, kw, ha='center', va='center', fontsize=7,
             fontweight='bold', color='#0d1117')
ax2.set_title('Bubble Chart\\n(size = TF-IDF score)', color='#FFD700', fontweight='bold')
ax2.set_xticks([]); ax2.set_yticks([])

plt.savefig('task_b_visualization.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print('✅  Saved: task_b_visualization.png')
"""))

# ── TASK C HEADER ─────────────────────────────────────────────
cells.append(md("""---
## 🟢 Task C — Document Similarity Search
Cosine similarity search — **no stopword removal** (per assignment spec).

> ✏️ **Edit `USER_DOC` and `PERCENTILE` below.**
"""))

# ── TASK C CODE ───────────────────────────────────────────────
cells.append(code(
'USER_DOC = (\n'
'    "Oil prices surged today as OPEC agreed to cut production significantly. "\n'
'    "The crude market saw strong gains and the dollar weakened. "\n'
'    "Energy stocks rose sharply on the news."\n'
')\n'
'PERCENTILE = 80\n'
'\n'
'print("🟢 Task C: Document Similarity Search")\n'
'print(f"  Query     : {USER_DOC[:70]}...")\n'
'print(f"  Percentile: {PERCENTILE}th\\n")\n'
'\n'
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
'print(f"  Threshold : {threshold:.6f}")\n'
'print(f"  Matches   : {len(results_c):,}\\n")\n'
'print(f"  RANK   DOCUMENT ID                        SIMILARITY")\n'
'print("  " + "-"*55)\n'
'for rank,(did,sc) in enumerate(results_c[:15], 1):\n'
'    print(f"  {rank:<6} {did:<35} {sc:>10.6f}")\n'
'if len(results_c)>15:\n'
'    print(f"  ... +{len(results_c)-15} more")\n'
'print("\\n✅  Task C done!")\n'
))

# ── TASK C VISUALIZATIONS ────────────────────────────────────
cells.append(code("""
# ═══ TASK C VISUALIZATIONS ════════════════════════════════════
TOP_N = min(20, len(results_c))
top_docs   = results_c[:TOP_N]
top_scores = [s for _,s in top_docs]
top_labels = [d.split('/')[-1] for d,_ in top_docs]

fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('🟢 Task C — Document Similarity Search',
             fontsize=18, fontweight='bold', color='#3fb950')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# 1. Similarity score histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ALL_SIMS, bins=80, color='#4FC3F7', alpha=0.8, edgecolor='none')
ylim = ax1.get_ylim()
ax1.axvline(threshold, color='#FF6B6B', lw=2.5,
            label=f'{PERCENTILE}th pct = {threshold:.4f}')
ax1.fill_betweenx([0, ylim[1]], threshold, ALL_SIMS.max(),
                   alpha=0.2, color='#3fb950',
                   label=f'{len(results_c):,} matches')
ax1.set_ylim(ylim)
ax1.set_title('Similarity Score Distribution', color='#3fb950', fontweight='bold')
ax1.set_xlabel('Cosine Similarity'); ax1.set_ylabel('# Documents')
ax1.legend(fontsize=9, framealpha=0.3)

# 2. Top-N bar chart
ax2 = fig.add_subplot(gs[0, 1])
grad = plt.cm.YlGn(np.linspace(0.4, 1.0, TOP_N))
bars = ax2.barh(top_labels[::-1], top_scores[::-1], color=grad, edgecolor='none')
ax2.set_title(f'Top {TOP_N} Most Similar Documents', color='#3fb950', fontweight='bold')
ax2.set_xlabel('Cosine Similarity Score')
for b,sc in zip(bars, top_scores[::-1]):
    ax2.text(b.get_width()+0.0005, b.get_y()+b.get_height()/2,
             f'{sc:.4f}', va='center', color='white', fontsize=7)

# 3. Top-15 similarity heatmap (doc × doc cross-similarity)
ax3 = fig.add_subplot(gs[1, :])
heat_n   = min(15, len(results_c))
heat_ids = [results_c[i][0] for i in range(heat_n)]
heat_vec = vec_c.transform([reuters.raw(fid) for fid in heat_ids])
heat_mat = cosine_similarity(heat_vec, heat_vec)
short    = [h.split('/')[-1] for h in heat_ids]

mask = np.zeros_like(heat_mat, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(heat_mat, ax=ax3, cmap='YlGn', annot=True, fmt='.2f',
            xticklabels=short, yticklabels=short,
            linewidths=0.4, linecolor=EDGE, mask=False,
            cbar_kws={'label':'Cosine Similarity'})
ax3.set_title(f'Similarity Heatmap — Top {heat_n} Matching Documents',
              color='#3fb950', fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=35, ha='right', fontsize=8)

plt.savefig('task_c_visualization.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print('✅  Saved: task_c_visualization.png')
"""))

# ── SUMMARY ───────────────────────────────────────────────────
cells.append(md("""---
## ✅ Summary

| Task | Method | Key Visuals |
|------|--------|-------------|
| 🔵 A | K-Means cosine clustering | Word clouds · t-SNE map · Category heatmap |
| 🟡 B | TF-IDF 10-80-10 classification | Score bars · Bubble chart · Donut |
| 🟢 C | Cosine similarity search | Score histogram · Top-N bar · Similarity heatmap |

> 🖼️ Saved: `task_a_wordclouds.png` · `task_a_tsne.png` · `task_b_visualization.png` · `task_c_visualization.png`
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

print("✅  NLP_Assignment.ipynb (v2) created successfully!")
