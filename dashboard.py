"""
dashboard.py — Interactive NLP Assignment HTML Dashboard
Student: Abrham Assefa Habtamu | ID: VR548223
Run: python3 dashboard.py  →  open report.html
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
    top_idx = km.cluster_centers_[i].argsort()[::-1][:15]
    clusters_data.append({
        "id": i+1,
        "docs": int((labels==i).sum()),
        "terms": [ft[j] for j in top_idx],
        "scores": [float(km.cluster_centers_[i][j]) for j in top_idx]
    })

# Full vocab for interactive lookup
print("Building interactive vocabulary (top 4000 words)...")
vec_b = TfidfVectorizer(sublinear_tf=True, min_df=1,
                        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat_b = vec_b.fit_transform(documents)
voc_b = vec_b.vocabulary_

# Pre-compute TF-IDF for entire vocabulary for live lookup
print("Pre-computing scores...")
vocab_scores = {}
for w, idx in voc_b.items():
    col = mat_b.getcol(idx).data
    if len(col) > 0:
        vocab_scores[w] = round(float(np.mean(col)), 6)

# Sort by score and keep top 4000 for embedding
vocab_sorted = sorted(vocab_scores.items(), key=lambda x: x[1], reverse=True)[:4000]
vocab_dict   = dict(vocab_sorted)
all_scores   = list(vocab_dict.values())
VP10 = float(np.percentile(all_scores, 10))
VP90 = float(np.percentile(all_scores, 90))

# Standard keywords for table
KEYWORDS = ['oil','trade','market','dollar','bank','government',
            'price','stock','profit','export','import','economy',
            'debt','gold','energy','merger','tax','rate','inflation','currency']
def ks(w):
    return vocab_scores.get(w.lower(), 0.0)

kw_scores_b = {kw: ks(kw) for kw in KEYWORDS}
vals_b = list(kw_scores_b.values())
P10 = float(np.percentile(vals_b, 10))
P90 = float(np.percentile(vals_b, 90))
def classify(s, p10=P10, p90=P90):
    return 'TOP' if s >= p90 else ('BOTTOM' if s < p10 else 'MEDIUM')

kw_data = sorted(
    [{"kw": kw, "score": sc, "label": classify(sc)} for kw, sc in kw_scores_b.items()],
    key=lambda x: x["score"], reverse=True
)

# Embed images
def img_b64(path):
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

img_a_wc = img_b64("task_a_wordclouds.png")
img_a_ts = img_b64("task_a_tsne.png")
img_b_p  = img_b64("task_b_visualization.png")
img_c_p  = img_b64("task_c_visualization.png")

# ── PRE-BUILD HTML FRAGMENTS ──────────────────────────────────
cluster_cards = ""
CLUSTER_COLORS = ["#58a6ff","#3fb950","#FFD700","#ff7b72","#d2a8ff"]
for c in clusters_data:
    col = CLUSTER_COLORS[c["id"]-1]
    terms_html = " ".join(f'<span class="term-pill">{t}</span>' for t in c["terms"][:8])
    cluster_cards += f"""
    <div class="cluster-card" onclick="toggleCluster(this)" data-id="{c['id']}">
      <div class="cc-header">
        <div class="cc-badge" style="background:{col}22;color:{col};border:1px solid {col}">Cluster {c['id']}</div>
        <div class="cc-count">{c['docs']:,} <span>docs</span></div>
        <div class="cc-arrow">▼</div>
      </div>
      <div class="cc-terms">{terms_html}</div>
      <div class="cc-full" style="display:none">
        {"".join(f'<span class="term-pill full">{t} <small>({s:.4f})</small></span>' for t,s in zip(c["terms"], c["scores"]))}
      </div>
    </div>"""

kw_rows = ""
for d in kw_data:
    lbl = d["label"]
    pill = 'pill-top' if lbl=='TOP' else ('pill-bottom' if lbl=='BOTTOM' else 'pill-mid')
    icon = '⭐' if lbl=='TOP' else ('🔻' if lbl=='BOTTOM' else '📊')
    bar_w = int(d["score"] / max(v["score"] for v in kw_data) * 100)
    kw_rows += (
        f'<tr data-label="{lbl}" data-kw="{d["kw"]}">'
        f'<td><b>{d["kw"]}</b></td>'
        f'<td><div class="score-bar-wrap"><div class="score-bar" style="width:{bar_w}%"></div>'
        f'<span>{d["score"]:.6f}</span></div></td>'
        f'<td><span class="{pill}">{icon} {lbl}</span></td></tr>\n'
    )

img_wc_html = f'<div class="cb"><h3>☁️ Word Clouds per Cluster</h3><img class="embed" src="{img_a_wc}"/></div>' if img_a_wc else '<div class="cb"><p class="hint">Run NLP_Assignment.ipynb in Colab to generate charts here.</p></div>'
img_ts_html = f'<div class="cb"><h3>🗺️ t-SNE Map + Category Alignment</h3><img class="embed" src="{img_a_ts}"/></div>' if img_a_ts else ''
img_b_html  = f'<div class="cb"><h3>📊 Full Visualization</h3><img class="embed" src="{img_b_p}"/></div>' if img_b_p else ''
img_c_html  = f'<div class="cb"><h3>📈 Similarity Analysis</h3><img class="embed" src="{img_c_p}"/></div>' if img_c_p else ''

tops = sum(1 for d in kw_data if d["label"]=="TOP")
mids = sum(1 for d in kw_data if d["label"]=="MEDIUM")
bots = sum(1 for d in kw_data if d["label"]=="BOTTOM")
bh   = max(340, len(KEYWORDS)*22)
n_kw = len(KEYWORDS)

# Plotly data
pie_data   = json.dumps([{"labels":[f"Cluster {c['id']}" for c in clusters_data],
                           "values":[c["docs"] for c in clusters_data],"type":"pie","hole":0.42,
                           "marker":{"colors":["#58a6ff","#3fb950","#FFD700","#ff7b72","#d2a8ff"]}}])
bar_a_data = json.dumps([{"x":c["terms"][:8],"y":c["scores"][:8],"type":"bar",
                           "name":f"C{c['id']}","marker":{"color":CLUSTER_COLORS[c['id']-1]}} for c in clusters_data])
bar_b_data = json.dumps([{"x":[d["score"] for d in kw_data],
                           "y":[d["kw"] for d in kw_data],"type":"bar","orientation":"h",
                           "marker":{"color":["#FFD700" if d["label"]=="TOP" else
                                              ("#4FC3F7" if d["label"]=="MEDIUM" else "#EF5350")
                                              for d in kw_data]}}])

vocab_json = json.dumps(vocab_dict)
clusters_json = json.dumps([{"id":c["id"],"terms":c["terms"][:8]} for c in clusters_data])
max_score_val = round(float(max(vocab_scores.values())), 6)
vocab_count = len(vocab_dict)

# ── BUILD HTML ────────────────────────────────────────────────
parts = []

# CSS
parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NLP Assignment — Abrham Assefa Habtamu</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#21262d;--border:#30363d;
      --text:#e6edf3;--muted:#8b949e;--blue:#58a6ff;--green:#3fb950;
      --gold:#FFD700;--red:#ff7b72;--purple:#d2a8ff;--cyan:#4FC3F7}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}

/* HEADER */
header{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1c2128 100%);
  border-bottom:1px solid var(--border);padding:2rem 3rem;position:relative;overflow:hidden}
header::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;
  background:radial-gradient(circle,#58a6ff15 0%,transparent 70%);pointer-events:none}
header::after{content:'';position:absolute;bottom:-80px;left:30%;width:400px;height:400px;
  background:radial-gradient(circle,#3fb95010 0%,transparent 70%);pointer-events:none}

.student-card{display:flex;align-items:center;gap:1.2rem;margin-bottom:1.4rem;
  background:linear-gradient(135deg,#58a6ff12,#3fb95008);
  border:1px solid #58a6ff33;border-radius:14px;padding:1rem 1.5rem;
  max-width:520px;backdrop-filter:blur(8px)}
.student-avatar{width:54px;height:54px;border-radius:50%;
  background:linear-gradient(135deg,#58a6ff,#3fb950);
  display:flex;align-items:center;justify-content:center;
  font-size:1.4rem;font-weight:800;color:#0d1117;flex-shrink:0;
  box-shadow:0 0 20px #58a6ff44}
.student-info h3{font-size:1.05rem;font-weight:700;color:var(--text)}
.student-info p{font-size:.82rem;color:var(--muted);margin-top:.15rem}
.student-id{display:inline-block;background:#58a6ff18;border:1px solid #58a6ff44;
  color:var(--blue);padding:.15rem .55rem;border-radius:8px;font-size:.75rem;
  font-weight:600;font-family:monospace;margin-top:.3rem}

header h1{font-size:1.9rem;font-weight:800;color:var(--blue);letter-spacing:-.02em}
header p.sub{color:var(--muted);margin-top:.35rem;font-size:.9rem}
.badges{margin-top:.9rem;display:flex;gap:.4rem;flex-wrap:wrap}
.badge{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .75rem;
  border-radius:12px;font-size:.78rem;font-weight:600}
.ba{background:#1f6feb22;color:var(--blue);border:1px solid #1f6feb}
.bb{background:#9e6a0322;color:var(--gold);border:1px solid #9e6a03}
.bc{background:#1a7f3722;color:var(--green);border:1px solid #2ea043}
.bd{background:#6e40c922;color:var(--purple);border:1px solid #6e40c9}

/* NAV */
nav{display:flex;gap:.5rem;padding:.75rem 3rem;background:var(--bg2);
  border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;
  backdrop-filter:blur(10px)}
nav a{color:var(--muted);text-decoration:none;font-size:.88rem;padding:.4rem .9rem;
  border-radius:8px;transition:all .2s;font-weight:500}
nav a:hover{background:var(--bg3);color:var(--text)}
nav a.active{background:var(--bg3);color:var(--blue)}

/* LAYOUT */
.con{max-width:1400px;margin:0 auto;padding:2rem 3rem}
.sec{margin-bottom:3.5rem;animation:fadeIn .5s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.sh{display:flex;align-items:center;gap:.8rem;margin-bottom:1.5rem;
  padding-bottom:.8rem;border-bottom:1px solid var(--bg3)}
.sh h2{font-size:1.35rem;font-weight:700}
.sh .task-num{font-size:1.8rem}

/* STAT CARDS */
.cards{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-bottom:1.5rem}
.card{background:var(--bg2);border:1px solid var(--bg3);border-radius:12px;padding:1.2rem;
  transition:border-color .2s,transform .2s;cursor:default}
.card:hover{border-color:var(--border);transform:translateY(-2px)}
.card .num{font-size:2rem;font-weight:800;color:var(--blue);
  counter-reset:n var(--val);animation:countUp 1.5s ease}
.card .lbl{color:var(--muted);font-size:.84rem;margin-top:.2rem}
.cb{background:var(--bg2);border:1px solid var(--bg3);border-radius:12px;
  padding:1.4rem;margin-bottom:1.2rem}
.cb h3{font-size:1rem;font-weight:600;color:var(--muted);margin-bottom:.9rem;
  display:flex;align-items:center;gap:.5rem}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:768px){.g2{grid-template-columns:1fr}}
img.embed{width:100%;border-radius:8px;border:1px solid var(--bg3)}
.hint{color:var(--muted);font-style:italic;font-size:.88rem;padding:.5rem 0}

/* CLUSTER CARDS */
.cluster-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin-bottom:1.5rem}
.cluster-card{background:var(--bg2);border:1px solid var(--bg3);border-radius:12px;
  padding:1rem 1.2rem;cursor:pointer;transition:all .25s;user-select:none}
.cluster-card:hover{border-color:var(--border);transform:translateY(-2px)}
.cc-header{display:flex;align-items:center;gap:.7rem;margin-bottom:.7rem}
.cc-badge{padding:.2rem .65rem;border-radius:8px;font-size:.8rem;font-weight:700}
.cc-count{margin-left:auto;font-size:1.1rem;font-weight:700;color:var(--text)}
.cc-count span{font-size:.75rem;color:var(--muted);margin-left:.2rem}
.cc-arrow{color:var(--muted);font-size:.75rem;transition:transform .25s}
.cluster-card.open .cc-arrow{transform:rotate(180deg)}
.cc-terms{display:flex;flex-wrap:wrap;gap:.35rem}
.cc-full{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.7rem;
  padding-top:.7rem;border-top:1px solid var(--bg3);animation:fadeIn .3s ease}
.term-pill{background:var(--bg3);color:var(--text);padding:.15rem .5rem;
  border-radius:6px;font-size:.75rem;font-family:monospace}
.term-pill.full{background:#58a6ff15;color:var(--blue);border:1px solid #58a6ff33}
.term-pill small{color:var(--muted)}

/* INTERACTIVE ANALYZER */
.analyzer{background:linear-gradient(135deg,var(--bg2),#1c2128);
  border:1px solid #58a6ff33;border-radius:14px;padding:1.5rem;margin-bottom:1.5rem}
.analyzer h3{color:var(--blue);font-weight:700;margin-bottom:.3rem;font-size:1.1rem}
.analyzer .sub{color:var(--muted);font-size:.84rem;margin-bottom:1rem}
.input-row{display:flex;gap:.7rem;align-items:center;flex-wrap:wrap}
.kw-input{flex:1;min-width:200px;background:var(--bg);border:1px solid var(--border);
  color:var(--text);padding:.65rem 1rem;border-radius:8px;font-family:'Inter',sans-serif;
  font-size:.95rem;outline:none;transition:border-color .2s}
.kw-input:focus{border-color:var(--blue)}
.kw-input::placeholder{color:var(--muted)}
.btn{padding:.65rem 1.3rem;border-radius:8px;border:none;font-family:'Inter',sans-serif;
  font-size:.88rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn-blue{background:#1f6feb;color:white} .btn-blue:hover{background:#388bfd}
.btn-ghost{background:var(--bg3);color:var(--text);border:1px solid var(--border)}
.btn-ghost:hover{background:var(--border)}
.result-box{margin-top:1rem;padding:1rem 1.2rem;border-radius:10px;
  display:none;animation:fadeIn .3s ease;border:1px solid var(--bg3)}
.result-box.show{display:flex;align-items:center;gap:1rem}
.result-icon{font-size:2rem}
.result-word{font-size:1.1rem;font-weight:700;color:var(--text)}
.result-score{font-size:.85rem;color:var(--muted);margin-top:.2rem}
.not-found{color:var(--muted);font-style:italic}

/* KEYWORD TABLE */
.filter-row{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center}
.filter-btn{padding:.35rem .9rem;border-radius:20px;border:1px solid var(--border);
  background:var(--bg3);color:var(--muted);font-size:.82rem;font-weight:600;
  cursor:pointer;transition:all .2s}
.filter-btn:hover,.filter-btn.active{background:var(--blue);color:white;border-color:var(--blue)}
.filter-btn.f-top.active{background:#9e6a03;border-color:var(--gold);color:var(--gold)}
.filter-btn.f-mid.active{background:#1f6feb44;border-color:var(--blue);color:var(--blue)}
.filter-btn.f-bot.active{background:#da363344;border-color:var(--red);color:var(--red)}
.search-input{padding:.38rem .8rem;border-radius:8px;border:1px solid var(--border);
  background:var(--bg);color:var(--text);font-size:.84rem;outline:none;
  width:160px;transition:border-color .2s}
.search-input:focus{border-color:var(--blue)}
.kwt{width:100%;border-collapse:collapse;font-size:.86rem}
.kwt th{background:var(--bg3);color:var(--muted);padding:.5rem .8rem;text-align:left;font-weight:600;position:sticky;top:0}
.kwt td{padding:.45rem .8rem;border-bottom:1px solid var(--bg3)}
.kwt tr:hover td{background:#21262d55}
.kwt tr[style*="none"]{display:none}
.pill-top{padding:.18rem .55rem;border-radius:8px;font-size:.74rem;font-weight:700;
  background:#9e6a0322;color:var(--gold);border:1px solid #9e6a0355}
.pill-mid{padding:.18rem .55rem;border-radius:8px;font-size:.74rem;font-weight:700;
  background:#1f6feb22;color:var(--cyan);border:1px solid #1f6feb55}
.pill-bottom{padding:.18rem .55rem;border-radius:8px;font-size:.74rem;font-weight:700;
  background:#da363322;color:var(--red);border:1px solid #da363355}
.score-bar-wrap{display:flex;align-items:center;gap:.6rem}
.score-bar{height:6px;background:linear-gradient(90deg,#58a6ff,#3fb950);
  border-radius:3px;min-width:4px;transition:width .5s ease}

/* FOOTER */
footer{text-align:center;padding:2rem;color:var(--muted);font-size:.82rem;
  border-top:1px solid var(--bg3);margin-top:2rem;line-height:1.8}
footer .student-footer{color:var(--blue);font-weight:600}
</style></head><body>
""")

# HEADER
parts.append(f"""
<header>
  <div class="student-card">
    <div class="student-avatar">AH</div>
    <div class="student-info">
      <h3>Abrham Assefa Habtamu</h3>
      <p>NLP Course — General Assignment (Tasks A, B &amp; C)</p>
      <div class="student-id">🪪 ID: VR548223</div>
    </div>
  </div>
  <h1>🧠 NLP Assignment Dashboard</h1>
  <p class="sub">Reuters Corpus (10,788 articles) · Python · NLTK · scikit-learn</p>
  <div class="badges">
    <span class="badge ba">🔵 Task A — Clustering</span>
    <span class="badge bb">🟡 Task B — TF-IDF</span>
    <span class="badge bc">🟢 Task C — Similarity</span>
    <span class="badge bd">🔤 Reuters Corpus</span>
  </div>
</header>

<nav>
  <a href="#overview" class="active">📊 Overview</a>
  <a href="#task-a">🔵 Task A</a>
  <a href="#task-b">🟡 Task B</a>
  <a href="#task-c">🟢 Task C</a>
  <a href="#try-it">⚡ Try It</a>
</nav>

<div class="con">

<!-- OVERVIEW STATS -->
<div class="sec" id="overview">
  <div class="cards" style="margin-top:1rem">
    <div class="card"><div class="num" id="cnt1" style="color:var(--blue)">0</div><div class="lbl">📰 Reuters Documents</div></div>
    <div class="card"><div class="num" id="cnt2" style="color:var(--green)">{K}</div><div class="lbl">🔵 Clusters (Task A)</div></div>
    <div class="card"><div class="num" id="cnt3" style="color:var(--gold)">{n_kw}</div><div class="lbl">🟡 Keywords Analyzed</div></div>
    <div class="card"><div class="num" id="cnt4" style="color:var(--purple)">0</div><div class="lbl">🔤 Vocabulary Size</div></div>
  </div>
</div>

<!-- TASK A -->
<div class="sec" id="task-a">
  <div class="sh"><span class="task-num">🔵</span><h2 style="color:var(--blue)">Task A — Corpus Clustering</h2></div>
  <p style="color:var(--muted);margin-bottom:1.2rem;font-size:.9rem">
    K-Means on L2-normalized TF-IDF vectors = cosine similarity clustering. Click a cluster card to expand its top terms.
  </p>
  <div class="cluster-grid">{cluster_cards}</div>
  <div class="g2">
    <div class="cb"><h3>📊 Cluster Distribution</h3><div id="pie_a"></div></div>
    <div class="cb"><h3>📈 Top Terms per Cluster</h3><div id="bar_a"></div></div>
  </div>
  {img_wc_html}
  {img_ts_html}
</div>

<!-- TASK B -->
<div class="sec" id="task-b">
  <div class="sh"><span class="task-num">🟡</span><h2 style="color:var(--gold)">Task B — Keyword TF-IDF Classification</h2></div>
  <p style="color:var(--muted);margin-bottom:1.2rem;font-size:.9rem">
    10-80-10 percentile rule: ⭐ TOP (≥90th) · 📊 MEDIUM (10th–90th) · 🔻 BOTTOM (&lt;10th).
    Use the filters and search below to explore.
  </p>
  <div class="cards" style="margin-bottom:1.5rem">
    <div class="card"><div class="num" style="color:var(--gold)">{tops}</div><div class="lbl">⭐ TOP keywords</div></div>
    <div class="card"><div class="num" style="color:var(--cyan)">{mids}</div><div class="lbl">📊 MEDIUM keywords</div></div>
    <div class="card"><div class="num" style="color:var(--red)">{bots}</div><div class="lbl">🔻 BOTTOM keywords</div></div>
    <div class="card"><div class="num" style="color:var(--muted);font-size:1.1rem">{P10:.4f} / {P90:.4f}</div><div class="lbl">P10 / P90 thresholds</div></div>
  </div>
  <div class="g2">
    <div class="cb"><h3>📊 TF-IDF Scores</h3><div id="bar_b"></div></div>
    <div class="cb">
      <h3>🔍 Keyword Results</h3>
      <div class="filter-row">
        <button class="filter-btn active" onclick="filterKw('ALL',this)">All</button>
        <button class="filter-btn f-top" onclick="filterKw('TOP',this)">⭐ TOP</button>
        <button class="filter-btn f-mid" onclick="filterKw('MEDIUM',this)">📊 MEDIUM</button>
        <button class="filter-btn f-bot" onclick="filterKw('BOTTOM',this)">🔻 BOTTOM</button>
        <input class="search-input" id="kwSearch" placeholder="🔍 Search..." oninput="searchKw(this.value)"/>
      </div>
      <div style="overflow:auto;max-height:320px">
        <table class="kwt" id="kwTable">
          <tr><th>Keyword</th><th>TF-IDF Score</th><th>Class</th></tr>
          {kw_rows}
        </table>
      </div>
    </div>
  </div>
  {img_b_html}
</div>

<!-- TASK C -->
<div class="sec" id="task-c">
  <div class="sh"><span class="task-num">🟢</span><h2 style="color:var(--green)">Task C — Document Similarity Search</h2></div>
  <p style="color:var(--muted);margin-bottom:1.2rem;font-size:.9rem">
    Cosine similarity search on the full Reuters corpus — <b>no stopword removal</b> (per assignment spec).
    Given a document + percentile, returns all corpus docs above that similarity threshold.
  </p>
  <div class="cb" style="border-left:3px solid var(--green)">
    <div style="color:var(--muted);font-size:.82rem;font-weight:600;margin-bottom:.4rem">SAMPLE QUERY</div>
    <div style="font-style:italic;line-height:1.7;color:var(--text)">
      "Oil prices surged today as OPEC agreed to cut production significantly.
       The crude market saw strong gains and the dollar weakened against major currencies.
       Energy stocks rose sharply on the news."
    </div>
  </div>
  {img_c_html}
</div>

<!-- INTERACTIVE TRY IT -->
<div class="sec" id="try-it">
  <div class="sh"><span class="task-num">⚡</span><h2 style="color:var(--purple)">Try It Yourself</h2></div>

  <div class="analyzer">
    <h3>🔤 Live Keyword Analyzer</h3>
    <p class="sub">Type any English word and instantly see its TF-IDF score and classification on the Reuters corpus.</p>
    <div class="input-row">
      <input class="kw-input" id="liveInput" placeholder="e.g. inflation, revenue, acquisition..."
             onkeydown="if(event.key==='Enter') analyzeLive()"/>
      <button class="btn btn-blue" onclick="analyzeLive()">Analyze →</button>
      <button class="btn btn-ghost" onclick="randomWord()">🎲 Random</button>
    </div>
    <div class="result-box" id="liveResult">
      <div class="result-icon" id="resIcon"></div>
      <div>
        <div class="result-word" id="resWord"></div>
        <div class="result-score" id="resScore"></div>
        <div id="resBar" style="margin-top:.5rem;height:8px;border-radius:4px;width:0;transition:width .6s ease"></div>
      </div>
    </div>
  </div>

  <div class="analyzer" style="border-color:#3fb95033">
    <h3 style="color:var(--green)">📡 Cluster Predictor</h3>
    <p class="sub">Enter some text and see which cluster it most resembles based on top term overlap.</p>
    <div class="input-row" style="align-items:flex-start">
      <textarea class="kw-input" id="clusterInput" rows="3" style="resize:vertical"
        placeholder="Paste any text here...">Oil prices rose as OPEC cut production and energy markets reacted.</textarea>
      <button class="btn btn-blue" onclick="predictCluster()" style="align-self:flex-end">Predict Cluster →</button>
    </div>
    <div class="result-box" id="clusterResult" style="align-items:flex-start;flex-direction:column;gap:.6rem">
      <div id="clusterOut"></div>
    </div>
  </div>
</div>

</div><!-- /con -->

<footer>
  <div class="student-footer">Abrham Assefa Habtamu &nbsp;·&nbsp; Student ID: VR548223</div>
  NLP Course Assignment — General Tasks A, B &amp; C<br/>
  Reuters Corpus · Python · NLTK · scikit-learn · matplotlib<br/>
  <a href="https://github.com/abrham-cyper/NLP-2" target="_blank"
     style="color:var(--blue);text-decoration:none">📁 GitHub Repository</a>
</footer>
""")

# JAVASCRIPT
parts.append(f"""
<script>
// ── EMBEDDED DATA ──────────────────────────────────────────
const VOCAB = {vocab_json};
const VP10  = {VP10};
const VP90  = {VP90};

const CLUSTERS = {clusters_json};

const MAX_SCORE = {max_score_val};

// ── ANIMATED COUNTERS ──────────────────────────────────────
function animateCount(el, target, duration=1500){{
  let start = 0, step = target / (duration / 16);
  const t = setInterval(()=>{{
    start = Math.min(start + step, target);
    el.textContent = Math.floor(start).toLocaleString();
    if(start >= target) clearInterval(t);
  }}, 16);
}}
window.addEventListener('DOMContentLoaded', ()=>{{
  animateCount(document.getElementById('cnt1'), 10788);
  animateCount(document.getElementById('cnt4'), {vocab_count});
}});

// ── CLUSTER CARDS TOGGLE ───────────────────────────────────
function toggleCluster(card){{
  const full = card.querySelector('.cc-full');
  card.classList.toggle('open');
  full.style.display = card.classList.contains('open') ? 'flex' : 'none';
}}

// ── KEYWORD TABLE FILTER ───────────────────────────────────
let currentFilter = 'ALL';
let currentSearch = '';
function filterKw(label, btn){{
  currentFilter = label;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  applyFilters();
}}
function searchKw(val){{
  currentSearch = val.toLowerCase();
  applyFilters();
}}
function applyFilters(){{
  document.querySelectorAll('#kwTable tr[data-label]').forEach(row=>{{
    const matchLabel = currentFilter === 'ALL' || row.dataset.label === currentFilter;
    const matchSearch = !currentSearch || row.dataset.kw.includes(currentSearch);
    row.style.display = (matchLabel && matchSearch) ? '' : 'none';
  }});
}}

// ── LIVE KEYWORD ANALYZER ─────────────────────────────────
function analyzeLive(){{
  const word = document.getElementById('liveInput').value.trim().toLowerCase();
  const box  = document.getElementById('liveResult');
  box.className = 'result-box show';

  if(!word){{ box.className='result-box'; return; }}

  const score = VOCAB[word];
  const icon  = document.getElementById('resIcon');
  const wEl   = document.getElementById('resWord');
  const sEl   = document.getElementById('resScore');
  const bar   = document.getElementById('resBar');

  if(score === undefined){{
    icon.textContent = '❓';
    wEl.textContent  = '"' + word + '" — not in corpus vocabulary';
    wEl.style.color  = 'var(--muted)';
    sEl.textContent  = 'Try a different word (must appear in Reuters corpus)';
    bar.style.width  = '0'; bar.style.background = 'var(--bg3)';
    box.style.background = 'var(--bg2)'; box.style.borderColor = 'var(--bg3)';
  }} else {{
    let label, color, boxBg, boxBorder;
    if(score >= VP90)      {{ label='TOP';    color='#FFD700'; boxBg='#9e6a0312'; boxBorder='#9e6a0355'; }}
    else if(score < VP10)  {{ label='BOTTOM'; color='#EF5350'; boxBg='#da363312'; boxBorder='#da363355'; }}
    else                  {{ label='MEDIUM'; color='#4FC3F7'; boxBg='#1f6feb12'; boxBorder='#1f6feb55'; }}
    const icons = {{'TOP':'⭐','MEDIUM':'📊','BOTTOM':'🔻'}};
    icon.textContent = icons[label];
    wEl.textContent  = word;
    wEl.style.color  = color;
    sEl.textContent  = 'TF-IDF score: ' + score.toFixed(6) + ' · Class: ' + label;
    const pct = Math.min(100, Math.round(score / MAX_SCORE * 100));
    bar.style.width  = pct + '%';
    bar.style.background = 'linear-gradient(90deg,' + color + '88,' + color + ')';
    box.style.background = boxBg; box.style.borderColor = boxBorder;
  }}
}}

// ── RANDOM WORD ────────────────────────────────────────────
const ALL_WORDS = Object.keys(VOCAB);
function randomWord(){{
  const w = ALL_WORDS[Math.floor(Math.random() * ALL_WORDS.length)];
  document.getElementById('liveInput').value = w;
  analyzeLive();
}}

// ── CLUSTER PREDICTOR ─────────────────────────────────────
function predictCluster(){{
  const text = document.getElementById('clusterInput').value.toLowerCase();
  const words = text.match(/[a-zA-Z]{{2,}}/g) || [];
  const box   = document.getElementById('clusterResult');
  const out   = document.getElementById('clusterOut');
  box.className = 'result-box show';
  box.style.background='var(--bg2)'; box.style.borderColor='var(--bg3)';

  if(!words.length){{ out.innerHTML='<span style="color:var(--muted)">Enter some text first.</span>'; return; }}

  // Score each cluster by term overlap
  const scores = CLUSTERS.map(c=>{{
    let s = 0;
    words.forEach(w => {{ if(c.terms.includes(w)) s += (VOCAB[w] || 0); }});
    return {{id: c.id, score: s, terms: c.terms}};
  }}).sort((a,b)=>b.score-a.score);

  const best  = scores[0];
  const COLORS = ['#58a6ff','#3fb950','#FFD700','#ff7b72','#d2a8ff'];
  const col   = COLORS[best.id - 1];
  const matched = words.filter(w => best.terms.includes(w));

  out.innerHTML = `
    <div style="font-size:1rem;font-weight:700;color:${{col}}">→ Most likely Cluster ${{best.id}}</div>
    <div style="color:var(--muted);font-size:.84rem;margin-top:.3rem">
      Matched terms: ${{matched.length ? matched.map(w=>'<code style="background:var(--bg3);padding:0 .3rem;border-radius:4px">'+w+'</code>').join(' ') : 'none'}}
    </div>
    <div style="margin-top:.6rem;display:flex;gap:.5rem">
      ${{scores.map(s=>`<div style="text-align:center">
        <div style="font-size:.72rem;color:var(--muted)">C${{s.id}}</div>
        <div style="width:36px;background:${{COLORS[s.id-1]}}${{s.id===best.id?'':'44'}};
          height:${{Math.max(4,Math.round(s.score/(best.score||1)*40))}}px;
          border-radius:3px;margin:2px auto;transition:height .4s"></div>
      </div>`).join('')}}
    </div>`;
}}

// ── NAV ACTIVE STATE ───────────────────────────────────────
const observer = new IntersectionObserver(entries=>{{
  entries.forEach(e=>{{
    if(e.isIntersecting){{
      document.querySelectorAll('nav a').forEach(a=>a.classList.remove('active'));
      const id = e.target.id;
      const link = document.querySelector('nav a[href="#'+id+'"]');
      if(link) link.classList.add('active');
    }}
  }});
}}, {{threshold:0.3}});
document.querySelectorAll('.sec').forEach(s=>observer.observe(s));

// ── PLOTLY CHARTS ─────────────────────────────────────────
const dk={{paper_bgcolor:'#0d1117',plot_bgcolor:'#161b22',
  font:{{color:'#e6edf3',family:'Inter'}},
  xaxis:{{gridcolor:'#21262d',linecolor:'#30363d'}},
  yaxis:{{gridcolor:'#21262d',linecolor:'#30363d'}},
  margin:{{t:10,b:50,l:50,r:20}}}};

Plotly.newPlot('pie_a',{pie_data},{{...dk,showlegend:true,height:280,margin:{{t:10,b:10,l:10,r:10}}}});
Plotly.newPlot('bar_a',{bar_a_data},{{...dk,barmode:'group',height:280,xaxis:{{tickangle:-30}}}});
Plotly.newPlot('bar_b',{bar_b_data},{{...dk,height:{bh},margin:{{l:90,t:10,b:40,r:20}}}});
</script></body></html>""")

with open("report.html", "w", encoding="utf-8") as f:
    f.write("\n".join(parts))

print("✅  report.html generated!")
print("    Open with: open report.html")
