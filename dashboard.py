"""
dashboard.py — Generates a beautiful standalone HTML report.
Run: python3 dashboard.py
Then: open report.html
"""
import base64, json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import nltk
nltk.download('reuters', quiet=True)
from nltk.corpus import reuters

# ── DATA ──────────────────────────────────────────────────────
print("Loading corpus & running models...")
doc_ids   = reuters.fileids()
documents = [reuters.raw(fid) for fid in doc_ids]
K = 5

vec = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=3,
                      token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat = normalize(vec.fit_transform(documents), norm='l2')
ft  = vec.get_feature_names_out()
km  = KMeans(n_clusters=K, n_init=10, random_state=42).fit(mat)
labels = km.labels_

clusters_data = []
for i in range(K):
    top_idx = km.cluster_centers_[i].argsort()[::-1][:12]
    clusters_data.append({
        "id": i+1,
        "docs": int((labels==i).sum()),
        "terms": [ft[j] for j in top_idx],
        "scores": [float(km.cluster_centers_[i][j]) for j in top_idx]
    })

KEYWORDS = ['oil','trade','market','dollar','bank','government',
            'price','stock','profit','export','import','economy',
            'debt','gold','energy','merger','tax','rate','inflation','currency']
vec_b = TfidfVectorizer(sublinear_tf=True, min_df=1,
                        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat_b = vec_b.fit_transform(documents)
voc_b = vec_b.vocabulary_

def ks(w):
    if w not in voc_b: return 0.0
    col = mat_b.getcol(voc_b[w]).data
    return float(np.mean(col)) if len(col) else 0.0

scores_b = {kw: ks(kw) for kw in KEYWORDS}
vals_b   = list(scores_b.values())
P10 = float(np.percentile(vals_b, 10))
P90 = float(np.percentile(vals_b, 90))

def classify(s):
    return 'TOP' if s >= P90 else ('BOTTOM' if s < P10 else 'MEDIUM')

kw_data = sorted(
    [{"kw": kw, "score": sc, "label": classify(sc)} for kw, sc in scores_b.items()],
    key=lambda x: x["score"], reverse=True
)

def img_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

img_a_wc = img_b64("task_a_wordclouds.png")
img_a_ts = img_b64("task_a_tsne.png")
img_b    = img_b64("task_b_visualization.png")
img_c    = img_b64("task_c_visualization.png")

# ── PRE-BUILD HTML FRAGMENTS ──────────────────────────────────
cluster_cards = ""
for c in clusters_data:
    cluster_cards += (
        f'<div class="card">'
        f'<div class="ctitle">Cluster {c["id"]}</div>'
        f'<div class="num">{c["docs"]:,}</div>'
        f'<div class="lbl">documents</div>'
        f'<div class="terms">{", ".join(c["terms"][:6])}</div>'
        f'</div>'
    )

kw_rows = ""
for d in kw_data:
    lbl = d["label"]
    pill = 'pill-top' if lbl == 'TOP' else ('pill-bottom' if lbl == 'BOTTOM' else 'pill-mid')
    icon = '⭐' if lbl == 'TOP' else ('🔻' if lbl == 'BOTTOM' else '📊')
    kw_rows += (
        f'<tr><td><b>{d["kw"]}</b></td>'
        f'<td>{d["score"]:.6f}</td>'
        f'<td><span class="{pill}">{icon} {lbl}</span></td></tr>\n'
    )

img_wc_html = f'<div class="chart-box"><h3>☁️ Word Clouds per Cluster</h3><img class="embed" src="{img_a_wc}" alt="Word Clouds"/></div>' if img_a_wc else '<div class="chart-box"><h3>☁️ Word Clouds</h3><p style="color:#8b949e">Run NLP_Assignment.ipynb in Colab first to generate charts.</p></div>'
img_ts_html = f'<div class="chart-box"><h3>🗺️ t-SNE Cluster Map + Category Heatmap</h3><img class="embed" src="{img_a_ts}" alt="t-SNE"/></div>' if img_a_ts else ''
img_b_html  = f'<div class="chart-box"><h3>📊 Task B Full Chart</h3><img class="embed" src="{img_b}" alt="Task B"/></div>' if img_b else ''
img_c_html  = f'<div class="chart-box"><h3>📈 Task C Similarity Analysis</h3><img class="embed" src="{img_c}" alt="Task C"/></div>' if img_c else ''

tops   = sum(1 for d in kw_data if d["label"] == "TOP")
mids   = sum(1 for d in kw_data if d["label"] == "MEDIUM")
bots   = sum(1 for d in kw_data if d["label"] == "BOTTOM")
bh     = max(300, len(KEYWORDS) * 22)
n_kw   = len(KEYWORDS)

# ── PLOTLY DATA ───────────────────────────────────────────────
pie_data   = json.dumps([{"labels": [f"Cluster {c['id']}" for c in clusters_data],
                           "values": [c["docs"] for c in clusters_data],
                           "type": "pie", "hole": 0.45,
                           "marker": {"colors": ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854"]}}])
bar_a_data = json.dumps([{"x": c["terms"], "y": c["scores"],
                           "type": "bar", "name": f"Cluster {c['id']}"} for c in clusters_data])
bar_b_data = json.dumps([{"x": [d["score"] for d in kw_data],
                           "y": [d["kw"] for d in kw_data],
                           "type": "bar", "orientation": "h",
                           "marker": {"color": ["#FFD700" if d["label"]=="TOP"
                                                else ("#4FC3F7" if d["label"]=="MEDIUM" else "#EF5350")
                                                for d in kw_data]}}])

# ── WRITE HTML ────────────────────────────────────────────────
html_parts = [
"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NLP Assignment — Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
header{background:linear-gradient(135deg,#161b22,#21262d);border-bottom:1px solid #30363d;padding:2rem 3rem}
header h1{font-size:2rem;font-weight:700;color:#58a6ff}
header p{color:#8b949e;margin-top:.4rem}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:12px;font-size:.78rem;font-weight:600;margin:.15rem}
.ba{background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb}
.bb{background:#9e6a0322;color:#FFD700;border:1px solid #9e6a03}
.bc{background:#1a7f3722;color:#3fb950;border:1px solid #2ea043}
nav{display:flex;gap:1rem;padding:1rem 3rem;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:100}
nav a{color:#8b949e;text-decoration:none;font-size:.9rem;padding:.4rem .8rem;border-radius:6px;transition:all .2s}
nav a:hover{background:#21262d;color:#e6edf3}
.con{max-width:1400px;margin:0 auto;padding:2rem 3rem}
.sec{margin-bottom:3rem}
.sh{display:flex;align-items:center;gap:.8rem;margin-bottom:1.5rem;padding-bottom:.8rem;border-bottom:1px solid #21262d}
.sh h2{font-size:1.4rem;font-weight:700}
.cards{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin-bottom:1.5rem}
.card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.2rem}
.card .ctitle{font-size:1.05rem;font-weight:700;color:#58a6ff}
.card .num{font-size:2rem;font-weight:700;color:#58a6ff}
.card .lbl{color:#8b949e;font-size:.85rem;margin-top:.2rem}
.card .terms{margin-top:.6rem;font-size:.76rem;color:#8b949e}
.cb{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.2rem;margin-bottom:1.2rem}
.cb h3{font-size:1rem;font-weight:600;color:#8b949e;margin-bottom:.8rem}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
.g3{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem}
img.embed{width:100%;border-radius:8px;border:1px solid #21262d}
.kwt{width:100%;border-collapse:collapse;font-size:.85rem}
.kwt th{background:#21262d;color:#8b949e;padding:.5rem .8rem;text-align:left;font-weight:600}
.kwt td{padding:.45rem .8rem;border-bottom:1px solid #21262d}
.kwt tr:hover td{background:#21262d44}
.pill-top{padding:.15rem .5rem;border-radius:8px;font-size:.75rem;font-weight:600;background:#9e6a0322;color:#FFD700;border:1px solid #9e6a03}
.pill-mid{padding:.15rem .5rem;border-radius:8px;font-size:.75rem;font-weight:600;background:#1f6feb22;color:#4FC3F7;border:1px solid #1f6feb}
.pill-bottom{padding:.15rem .5rem;border-radius:8px;font-size:.75rem;font-weight:600;background:#da363322;color:#EF5350;border:1px solid #da3633}
footer{text-align:center;padding:2rem;color:#484f58;font-size:.82rem;border-top:1px solid #21262d;margin-top:2rem}
</style></head><body>
""",
    f"""<header>
  <h1>🧠 NLP Assignment Dashboard</h1>
  <p>Reuters Corpus — 10,788 news articles &nbsp;|&nbsp; Python · NLTK · scikit-learn</p>
  <div style="margin-top:.8rem">
    <span class="badge ba">🔵 Task A — Clustering</span>
    <span class="badge bb">🟡 Task B — TF-IDF</span>
    <span class="badge bc">🟢 Task C — Similarity</span>
  </div>
</header>
<nav>
  <a href="#task-a">🔵 Task A</a>
  <a href="#task-b">🟡 Task B</a>
  <a href="#task-c">🟢 Task C</a>
</nav>
<div class="con">
<div class="cards" style="margin:2rem 0 2.5rem">
  <div class="card"><div class="num">10,788</div><div class="lbl">Reuters Documents</div></div>
  <div class="card"><div class="num" style="color:#3fb950">{K}</div><div class="lbl">Clusters (Task A)</div></div>
  <div class="card"><div class="num" style="color:#FFD700">{n_kw}</div><div class="lbl">Keywords (Task B)</div></div>
  <div class="card"><div class="num" style="color:#ff7b72">90%</div><div class="lbl">Default Threshold (Task C)</div></div>
</div>
<div class="sec" id="task-a">
  <div class="sh"><span style="font-size:1.8rem">🔵</span><h2 style="color:#58a6ff">Task A — Corpus Clustering</h2></div>
  <div class="g3" style="margin-bottom:1.2rem">{cluster_cards}</div>
  <div class="g2">
    <div class="cb"><h3>Cluster Distribution</h3><div id="pie_a"></div></div>
    <div class="cb"><h3>Top Terms per Cluster</h3><div id="bar_a"></div></div>
  </div>
  {img_wc_html}
  {img_ts_html}
</div>
<div class="sec" id="task-b">
  <div class="sh"><span style="font-size:1.8rem">🟡</span><h2 style="color:#FFD700">Task B — Keyword TF-IDF Classification</h2></div>
  <div class="cards" style="margin-bottom:1.2rem">
    <div class="card"><div class="num" style="color:#FFD700">{tops}</div><div class="lbl">⭐ TOP keywords</div></div>
    <div class="card"><div class="num" style="color:#4FC3F7">{mids}</div><div class="lbl">📊 MEDIUM keywords</div></div>
    <div class="card"><div class="num" style="color:#EF5350">{bots}</div><div class="lbl">🔻 BOTTOM keywords</div></div>
    <div class="card"><div class="num" style="color:#8b949e;font-size:1.2rem">{P10:.4f} / {P90:.4f}</div><div class="lbl">10th / 90th percentile</div></div>
  </div>
  <div class="g2">
    <div class="cb"><h3>TF-IDF Scores</h3><div id="bar_b"></div></div>
    <div class="cb" style="overflow:auto"><h3>Keyword Results</h3>
      <table class="kwt">
        <tr><th>Keyword</th><th>TF-IDF Score</th><th>Class</th></tr>
        {kw_rows}
      </table></div>
  </div>
  {img_b_html}
</div>
<div class="sec" id="task-c">
  <div class="sh"><span style="font-size:1.8rem">🟢</span><h2 style="color:#3fb950">Task C — Document Similarity Search</h2></div>
  <div class="cb" style="margin-bottom:1.2rem">
    <div class="lbl">Sample Query</div>
    <div style="margin-top:.5rem;font-style:italic;line-height:1.6;color:#e6edf3">
      "Oil prices surged today as OPEC agreed to cut production. The crude market saw strong gains and the dollar weakened..."
    </div>
  </div>
  {img_c_html}
</div>
</div>
<footer>NLP Course Assignment &nbsp;·&nbsp; Reuters Corpus &nbsp;·&nbsp; Python + NLTK + scikit-learn &nbsp;·&nbsp; 2026</footer>
""",
    f"""<script>
const dk = {{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',
  font:{{color:'white',family:'Inter'}},
  xaxis:{{gridcolor:'#21262d'}},yaxis:{{gridcolor:'#21262d'}}}};
Plotly.newPlot('pie_a',{pie_data},{{...dk,margin:{{t:10,b:10}},height:280}});
Plotly.newPlot('bar_a',{bar_a_data},{{...dk,barmode:'group',margin:{{t:10,b:80}},height:280,xaxis:{{tickangle:-35}}}});
Plotly.newPlot('bar_b',{bar_b_data},{{...dk,margin:{{l:90,t:10,b:40}},height:{bh}}});
</script></body></html>"""
]

with open("report.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html_parts))

print("✅  report.html generated successfully!")
print("    Open with: open report.html")
