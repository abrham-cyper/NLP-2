"""
generate_report.py — Beautiful PDF Report for NLP Assignment
Student: Abrham Assefa Habtamu | ID: VR548223
Run: python3 generate_report.py  →  NLP_Project_Report.pdf
"""
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics.charts.legends import Legend
from reportlab.pdfgen import canvas
from datetime import datetime
import os, sys

# ── PALETTE ──────────────────────────────────────────────────
DARK       = colors.HexColor('#0d1117')
DARK2      = colors.HexColor('#161b22')
DARK3      = colors.HexColor('#21262d')
BLUE       = colors.HexColor('#58a6ff')
BLUE_D     = colors.HexColor('#1f6feb')
GREEN      = colors.HexColor('#3fb950')
GOLD       = colors.HexColor('#e3a008')
RED        = colors.HexColor('#f85149')
PURPLE     = colors.HexColor('#bc8cff')
CYAN       = colors.HexColor('#39d353')
WHITE      = colors.white
LIGHT_GRAY = colors.HexColor('#f6f8fa')
MED_GRAY   = colors.HexColor('#8b949e')
BORDER     = colors.HexColor('#30363d')

W, H = A4   # 595.27 x 841.89 pt
M = 2*cm    # margin

# ── CANVAS CALLBACKS ─────────────────────────────────────────
def cover_bg(canvas, doc):
    """Full-page dark gradient background for cover."""
    canvas.saveState()
    # Background
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Decorative top-right glow
    canvas.setFillColorRGB(0.35, 0.65, 1, 0.08)
    canvas.circle(W, H, 280, fill=1, stroke=0)
    # Decorative bottom-left glow
    canvas.setFillColorRGB(0.25, 0.72, 0.31, 0.06)
    canvas.circle(0, 0, 240, fill=1, stroke=0)
    # Top accent line
    canvas.setStrokeColor(BLUE_D)
    canvas.setLineWidth(3)
    canvas.line(0, H - 4*mm, W, H - 4*mm)
    # Bottom accent line
    canvas.setStrokeColor(GREEN)
    canvas.setLineWidth(2)
    canvas.line(0, 4*mm, W, 4*mm)
    canvas.restoreState()

def inner_bg(canvas, doc):
    """White background with subtle header/footer for inner pages."""
    canvas.saveState()
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # Header bar
    canvas.setFillColor(colors.HexColor('#f6f8fa'))
    canvas.rect(0, H - 2*cm, W, 2*cm, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor('#d0d7de'))
    canvas.setLineWidth(0.5)
    canvas.line(0, H - 2*cm, W, H - 2*cm)
    # Header text
    canvas.setFillColor(colors.HexColor('#57606a'))
    canvas.setFont('Helvetica', 8)
    canvas.drawString(M, H - 1.3*cm, 'NLP Course Assignment — Abrham Assefa Habtamu')
    canvas.drawRightString(W - M, H - 1.3*cm, f'Page {doc.page}')
    # Footer bar
    canvas.setFillColor(colors.HexColor('#f6f8fa'))
    canvas.rect(0, 0, W, 1.5*cm, fill=1, stroke=0)
    canvas.setStrokeColor(colors.HexColor('#d0d7de'))
    canvas.line(0, 1.5*cm, W, 1.5*cm)
    canvas.setFillColor(colors.HexColor('#57606a'))
    canvas.drawCentredString(W/2, 0.55*cm, 'Student ID: VR548223  ·  Reuters Corpus  ·  Python + NLTK + scikit-learn')
    # Left accent stripe
    canvas.setFillColor(BLUE_D)
    canvas.rect(0, 1.5*cm, 3, H - 3.5*cm, fill=1, stroke=0)
    canvas.restoreState()

# ── STYLES ───────────────────────────────────────────────────
styles = getSampleStyleSheet()

def sty(name, **kw):
    return ParagraphStyle(name, **kw)

# Cover styles (dark bg)
S_COVER_TAG    = sty('ct', fontName='Helvetica', fontSize=10, textColor=BLUE,
                     alignment=TA_CENTER, spaceAfter=4)
S_COVER_TITLE  = sty('ctitle', fontName='Helvetica-Bold', fontSize=34,
                     textColor=WHITE, alignment=TA_CENTER, leading=42, spaceAfter=8)
S_COVER_SUB    = sty('csub', fontName='Helvetica', fontSize=13,
                     textColor=MED_GRAY, alignment=TA_CENTER, spaceAfter=6)
S_COVER_NAME   = sty('cname', fontName='Helvetica-Bold', fontSize=16,
                     textColor=WHITE, alignment=TA_CENTER)
S_COVER_ID     = sty('cid', fontName='Helvetica', fontSize=12,
                     textColor=MED_GRAY, alignment=TA_CENTER)
S_COVER_DATE   = sty('cdate', fontName='Helvetica', fontSize=10,
                     textColor=colors.HexColor('#484f58'), alignment=TA_CENTER)

# Inner styles
S_H1 = sty('h1', fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor('#0969da'),
           spaceBefore=18, spaceAfter=8, borderPad=4)
S_H2 = sty('h2', fontName='Helvetica-Bold', fontSize=14, textColor=colors.HexColor('#1a7f37'),
           spaceBefore=14, spaceAfter=6)
S_H3 = sty('h3', fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#6e40c9'),
           spaceBefore=10, spaceAfter=4)
S_BODY = sty('body', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#1f2328'),
             leading=16, spaceAfter=6, alignment=TA_JUSTIFY)
S_MONO = sty('mono', fontName='Courier', fontSize=9, textColor=colors.HexColor('#0550ae'),
             backColor=colors.HexColor('#ddf4ff'), leading=14, spaceAfter=4,
             leftIndent=8, rightIndent=8, borderPad=4)
S_CAPTION = sty('cap', fontName='Helvetica-Oblique', fontSize=8.5,
                textColor=MED_GRAY, alignment=TA_CENTER, spaceAfter=10)
S_LABEL  = sty('lbl', fontName='Helvetica-Bold', fontSize=9,
               textColor=colors.HexColor('#57606a'))
S_TOC_H  = sty('toc_h', fontName='Helvetica-Bold', fontSize=11,
               textColor=colors.HexColor('#0969da'), spaceBefore=2, spaceAfter=2)
S_TOC_L  = sty('toc_l', fontName='Helvetica', fontSize=10,
               textColor=colors.HexColor('#57606a'), leftIndent=16, spaceAfter=1)
S_BULLET = sty('bul', fontName='Helvetica', fontSize=10,
               textColor=colors.HexColor('#1f2328'), leading=16,
               leftIndent=16, spaceAfter=3)

# ── HELPERS ──────────────────────────────────────────────────
def divider(color=colors.HexColor('#d0d7de'), thickness=0.5):
    return HRFlowable(width='100%', thickness=thickness, color=color,
                      spaceAfter=8, spaceBefore=4)

def spacer(h=0.3):
    return Spacer(1, h*cm)

def colored_table(data, col_widths, header_color=BLUE_D, row_colors=None, font_size=9):
    """Build a styled table with colored header."""
    style = [
        ('BACKGROUND',  (0,0), (-1,0), header_color),
        ('TEXTCOLOR',   (0,0), (-1,0), WHITE),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), font_size),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('ALIGN',       (1,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f6f8fa'), WHITE]),
        ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#d0d7de')),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING',(0,0), (-1,-1), 8),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0),(-1,-1), 5),
        ('FONTNAME',    (0,1), (-1,-1), 'Helvetica'),
        ('ROUNDEDCORNERS', [4]),
    ]
    if row_colors:
        for i, c in enumerate(row_colors, 1):
            style.append(('TEXTCOLOR', (2,i), (2,i), c))
            style.append(('FONTNAME',  (2,i), (2,i), 'Helvetica-Bold'))
    return Table(data, colWidths=col_widths, style=TableStyle(style),
                 hAlign='LEFT', repeatRows=1)

def info_box(text, bg=colors.HexColor('#ddf4ff'), border=colors.HexColor('#0969da')):
    """A colored callout box."""
    t = Table([[Paragraph(text, sty('ib', fontName='Helvetica', fontSize=9.5,
                textColor=colors.HexColor('#0550ae'), leading=15))]],
              colWidths=[W - 2*M - 0.5*cm],
              style=TableStyle([
                  ('BACKGROUND', (0,0), (-1,-1), bg),
                  ('LEFTPADDING', (0,0), (-1,-1), 10),
                  ('RIGHTPADDING', (0,0), (-1,-1), 10),
                  ('TOPPADDING', (0,0), (-1,-1), 8),
                  ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                  ('LINEBEFORE', (0,0), (0,-1), 3, border),
                  ('ROUNDEDCORNERS', [4]),
              ]))
    return t

# ── RUN MODELS ───────────────────────────────────────────────
print("Loading corpus & running models for report data...")
import nltk
nltk.download('reuters', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import reuters, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

doc_ids   = reuters.fileids()
documents = [reuters.raw(fid) for fid in doc_ids]
print(f"  Loaded {len(documents):,} documents")

K = 5
CLUSTER_COLORS = [BLUE, GREEN, GOLD, RED, PURPLE]
vec = TfidfVectorizer(max_features=10000, sublinear_tf=True, min_df=3,
                      token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat  = normalize(vec.fit_transform(documents), norm='l2')
ft   = vec.get_feature_names_out()
km   = KMeans(n_clusters=K, n_init=10, random_state=42)
km.fit(mat)
labels = km.labels_
clusters = []
for i in range(K):
    top = km.cluster_centers_[i].argsort()[::-1][:10]
    clusters.append({"id":i+1,"docs":int((labels==i).sum()),
                     "terms":[ft[j] for j in top]})
print("  Clustering done")

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
def classify(s): return 'TOP' if s>=P90 else ('BOTTOM' if s<P10 else 'MEDIUM')
kw_data = sorted([(kw,sc,classify(sc)) for kw,sc in scores_b.items()],
                 key=lambda x:x[1], reverse=True)
print("  TF-IDF done")

USER_DOC = ("Oil prices surged today as OPEC agreed to cut production. "
            "The crude market saw strong gains and the dollar weakened against major currencies. "
            "Energy stocks rose sharply on the news.")
PERCENTILE = 80
vec_c = TfidfVectorizer(sublinear_tf=True, min_df=2,
                        token_pattern=r'(?u)\b[a-zA-Z]{2,}\b')
mat_c = vec_c.fit_transform(documents)
uvec  = vec_c.transform([USER_DOC])
sims  = cosine_similarity(uvec, mat_c).flatten()
threshold = float(np.percentile(sims, PERCENTILE))
matches   = int((sims >= threshold).sum())
top_docs  = sorted([(doc_ids[i], float(sims[i])) for i in range(len(sims)) if sims[i]>=threshold],
                   key=lambda x:x[1], reverse=True)[:10]
print("  Similarity done")

# ── CHART HELPERS ────────────────────────────────────────────
def make_cluster_bar():
    """Cluster size bar chart."""
    d  = Drawing(360, 160)
    bc = VerticalBarChart()
    bc.x, bc.y = 40, 20
    bc.width, bc.height = 300, 120
    bc.data = [[c["docs"] for c in clusters]]
    bc.bars[0].fillColor = BLUE_D
    for i in range(K):
        bc.bars[(0,i)].fillColor = CLUSTER_COLORS[i]
    bc.valueAxis.valueMin  = 0
    bc.valueAxis.valueMax  = max(c["docs"] for c in clusters) * 1.15
    bc.valueAxis.valueStep = 500
    bc.valueAxis.labels.fontName = 'Helvetica'
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.gridStrokeColor = colors.HexColor('#e1e4e8')
    bc.categoryAxis.labels.fontName = 'Helvetica'
    bc.categoryAxis.labels.fontSize = 8
    bc.categoryAxis.categoryNames = [f'C{c["id"]}' for c in clusters]
    bc.categoryAxis.labels.angle  = 0
    bc.groupSpacing = 8
    d.insert(0, Rect(0, 0, 360, 160, fillColor=colors.HexColor('#f6f8fa'), strokeColor=None))
    d.add(bc)
    return d

def make_tfidf_bar():
    """TF-IDF horizontal bar chart."""
    n  = len(kw_data)
    h  = max(140, n*14)
    d  = Drawing(360, h)
    bc = HorizontalBarChart()
    bc.x, bc.y = 80, 10
    bc.width, bc.height = 260, h - 20
    bc.data = [[kw[1] for kw in kw_data[::-1]]]
    bc.bars[0].fillColor = BLUE
    for i, kw in enumerate(kw_data[::-1]):
        if kw[2] == 'TOP':    bc.bars[(0,i)].fillColor = GOLD
        elif kw[2]=='BOTTOM': bc.bars[(0,i)].fillColor = RED
        else:                  bc.bars[(0,i)].fillColor = BLUE
    bc.valueAxis.valueMin  = 0
    bc.valueAxis.valueMax  = max(v[1] for v in kw_data)*1.15
    bc.valueAxis.labels.fontName = 'Helvetica'
    bc.valueAxis.labels.fontSize = 7
    bc.categoryAxis.labels.fontName = 'Helvetica'
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.categoryNames = [kw[0] for kw in kw_data[::-1]]
    bc.categoryAxis.labels.dx = -4
    d.insert(0, Rect(0, 0, 360, h, fillColor=colors.HexColor('#f6f8fa'), strokeColor=None))
    d.add(bc)
    return d

def make_pie():
    """Cluster size pie chart."""
    d = Drawing(200, 160)
    p = Pie()
    p.x, p.y = 30, 20
    p.width = p.height = 120
    p.data  = [c["docs"] for c in clusters]
    p.labels = [f'C{c["id"]}' for c in clusters]
    for i in range(K):
        p.slices[i].fillColor = CLUSTER_COLORS[i]
        p.slices[i].strokeColor = WHITE
        p.slices[i].strokeWidth = 1.5
        p.slices[i].labelRadius = 1.25
        p.slices[i].fontSize    = 8
    d.insert(0, Rect(0, 0, 200, 160, fillColor=colors.HexColor('#f6f8fa'), strokeColor=None))
    d.add(p)
    return d

def make_sim_bar():
    """Top-10 similarity bar chart."""
    d  = Drawing(360, 160)
    bc = VerticalBarChart()
    bc.x, bc.y = 50, 25
    bc.width, bc.height = 290, 115
    bc.data = [[s for _,s in top_docs]]
    for i in range(len(top_docs)):
        alpha = 1.0 - i*0.06
        bc.bars[(0,i)].fillColor = colors.HexColor(
            f'#{int(63*alpha):02x}{int(211*alpha):02x}{int(83*alpha):02x}')
    bc.valueAxis.valueMin  = min(s for _,s in top_docs)*0.9
    bc.valueAxis.valueMax  = top_docs[0][1]*1.08
    bc.valueAxis.labels.fontName = 'Helvetica'
    bc.valueAxis.labels.fontSize = 7
    bc.categoryAxis.labels.fontName = 'Helvetica'
    bc.categoryAxis.labels.fontSize = 6
    bc.categoryAxis.categoryNames = [d.split('/')[-1] for d,_ in top_docs]
    bc.categoryAxis.labels.angle  = 25
    d.insert(0, Rect(0, 0, 360, 160, fillColor=colors.HexColor('#f6f8fa'), strokeColor=None))
    d.add(bc)
    return d

# ── SECTION TITLE FLOWABLE ────────────────────────────────────
def section_title(num, emoji, text, color=BLUE_D):
    """Colored block section header."""
    data = [[Paragraph(f'{emoji}  {num}. {text}',
             sty('sh', fontName='Helvetica-Bold', fontSize=16,
                 textColor=WHITE, leading=20))]]
    t = Table(data, colWidths=[W - 2*M],
              style=TableStyle([
                  ('BACKGROUND',    (0,0),(-1,-1), color),
                  ('LEFTPADDING',   (0,0),(-1,-1), 12),
                  ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                  ('TOPPADDING',    (0,0),(-1,-1), 10),
                  ('BOTTOMPADDING', (0,0),(-1,-1), 10),
                  ('ROUNDEDCORNERS',[6]),
              ]))
    return t

def sub_header(text, color=colors.HexColor('#0969da')):
    return Paragraph(text, sty('sbh', fontName='Helvetica-Bold', fontSize=12,
                               textColor=color, spaceBefore=12, spaceAfter=5))

# ── BUILD DOCUMENT ────────────────────────────────────────────
print("Building PDF...")
OUT = "NLP_Project_Report.pdf"

story_cover  = []
story_inner  = []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COVER PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cover = []
cover.append(spacer(5.5))

# Top badge
badge = Table([[Paragraph('NLP COURSE — GENERAL ASSIGNMENT',
                sty('b1',fontName='Helvetica-Bold',fontSize=9,
                    textColor=BLUE,letterSpacing=2))]],
              colWidths=[10*cm],
              style=TableStyle([
                  ('BACKGROUND',   (0,0),(-1,-1),colors.HexColor('#0d2135')),
                  ('ALIGN',        (0,0),(-1,-1),'CENTER'),
                  ('LEFTPADDING',  (0,0),(-1,-1),16),
                  ('RIGHTPADDING', (0,0),(-1,-1),16),
                  ('TOPPADDING',   (0,0),(-1,-1),7),
                  ('BOTTOMPADDING',(0,0),(-1,-1),7),
                  ('ROUNDEDCORNERS',[12]),
                  ('BOX', (0,0),(-1,-1),1,colors.HexColor('#1f6feb')),
              ]))
cover.append(Table([[badge]], colWidths=[W-2*M],
             style=TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')])))
cover.append(spacer(1.2))

# Main title
cover.append(Paragraph('🧠 NLP Assignment', sty('mt', fontName='Helvetica-Bold',
             fontSize=42, textColor=WHITE, alignment=TA_CENTER, leading=50)))
cover.append(Paragraph('Project Report', sty('mt2', fontName='Helvetica',
             fontSize=28, textColor=BLUE, alignment=TA_CENTER, leading=36)))
cover.append(spacer(0.6))

# Subtitle
cover.append(Paragraph('Reuters Corpus · Tasks A, B &amp; C · Python + NLTK + scikit-learn',
             sty('cs', fontName='Helvetica', fontSize=12, textColor=MED_GRAY,
                 alignment=TA_CENTER)))
cover.append(spacer(2.5))

# Student card
card = Table([
    [Paragraph('👤', sty('av', fontName='Helvetica', fontSize=28, textColor=BLUE, alignment=TA_CENTER)),
     [Paragraph('Student Name', sty('sl',fontName='Helvetica',fontSize=9,textColor=MED_GRAY)),
      Paragraph('Abrham Assefa Habtamu', sty('sn',fontName='Helvetica-Bold',fontSize=16,textColor=WHITE,leading=20)),
      Paragraph('🪪 Student ID: VR548223', sty('si',fontName='Courier',fontSize=11,textColor=BLUE))]],
], colWidths=[2*cm, 11*cm],
   style=TableStyle([
       ('BACKGROUND',   (0,0),(-1,-1), colors.HexColor('#0d2135')),
       ('LEFTPADDING',  (0,0),(-1,-1), 18),
       ('RIGHTPADDING', (0,0),(-1,-1), 18),
       ('TOPPADDING',   (0,0),(-1,-1), 16),
       ('BOTTOMPADDING',(0,0),(-1,-1), 16),
       ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
       ('BOX', (0,0),(-1,-1), 1, colors.HexColor('#1f6feb')),
       ('ROUNDEDCORNERS',[10]),
   ]))
cover.append(Table([[card]], colWidths=[W-2*M],
             style=TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')])))
cover.append(spacer(2.0))

# Stats row on cover
def stat_cell(value, label, color):
    return [Paragraph(str(value),
                sty(f's{label}',fontName='Helvetica-Bold',fontSize=22,
                    textColor=color,alignment=TA_CENTER)),
            Paragraph(label,
                sty(f'sl{label}',fontName='Helvetica',fontSize=9,
                    textColor=MED_GRAY,alignment=TA_CENTER))]

stats = Table([
    [Table([stat_cell('10,788','Documents',BLUE)],   colWidths=[3.5*cm], style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0d2135')),('ALIGN',(0,0),(-1,-1),'CENTER'),('ROUNDEDCORNERS',[8]),('BOX',(0,0),(-1,-1),1,colors.HexColor('#1f3858')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)])),
     Table([stat_cell(K,'Clusters (A)',GREEN)],      colWidths=[3.5*cm], style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0d2135')),('ALIGN',(0,0),(-1,-1),'CENTER'),('ROUNDEDCORNERS',[8]),('BOX',(0,0),(-1,-1),1,colors.HexColor('#1a3d25')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)])),
     Table([stat_cell(len(KEYWORDS),'Keywords (B)',GOLD)], colWidths=[3.5*cm], style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0d2135')),('ALIGN',(0,0),(-1,-1),'CENTER'),('ROUNDEDCORNERS',[8]),('BOX',(0,0),(-1,-1),1,colors.HexColor('#3d2d00')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)])),
     Table([stat_cell(f'{matches:,}',f'Matches {PERCENTILE}th%',GREEN)], colWidths=[3.5*cm], style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#0d2135')),('ALIGN',(0,0),(-1,-1),'CENTER'),('ROUNDEDCORNERS',[8]),('BOX',(0,0),(-1,-1),1,colors.HexColor('#1a3d25')),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)])),
    ]
], colWidths=[3.5*cm,3.5*cm,3.5*cm,3.5*cm], hAlign='CENTER',
   style=TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
cover.append(Table([[stats]], colWidths=[W-2*M],
             style=TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')])))
cover.append(spacer(1.5))

# Date
cover.append(Paragraph(f'February 2026  ·  Academic Year 2025-26',
             sty('dt', fontName='Helvetica', fontSize=10,
                 textColor=colors.HexColor('#484f58'), alignment=TA_CENTER)))
cover.append(PageBreak())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INNER PAGES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
inner = []

# ── TABLE OF CONTENTS ─────────────────────────────────────────
inner.append(spacer(0.5))
inner.append(Paragraph('Table of Contents', S_H1))
inner.append(divider(BLUE_D, 1.5))
toc_items = [
    ('1.', 'Executive Summary', '3'),
    ('2.', 'Technical Stack', '3'),
    ('3.', 'Task A — Corpus Clustering', '4'),
    ('',   '3.1 Methodology', '4'),
    ('',   '3.2 Results', '4'),
    ('4.', 'Task B — Keyword TF-IDF Classification', '6'),
    ('',   '4.1 Methodology', '6'),
    ('',   '4.2 Results Table', '6'),
    ('5.', 'Task C — Document Similarity Search', '8'),
    ('',   '5.1 Methodology', '8'),
    ('',   '5.2 Results', '8'),
    ('6.', 'Conclusion', '9'),
]
for num, title, page in toc_items:
    row_style = S_TOC_H if num else S_TOC_L
    inner.append(Table([[
        Paragraph(f'{num} {title}', row_style),
        Paragraph(page, sty('pg', fontName='Helvetica', fontSize=10,
                            textColor=MED_GRAY, alignment=TA_RIGHT))
    ]], colWidths=[W-2*M-1.5*cm, 1.5*cm],
        style=TableStyle([
            ('LINEBELOW', (0,0),(-1,-1), 0.3, colors.HexColor('#e1e4e8')),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
        ])))
inner.append(spacer(0.5))
inner.append(PageBreak())

# ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('1', '📋', 'Executive Summary', BLUE_D))
inner.append(spacer(0.4))
inner.append(Paragraph(
    'This report presents the implementation of the <b>NLP Course General Assignment</b>, '
    'covering all three required tasks (A, B, and C). The implementation uses '
    '<b>Python 3</b> with <b>NLTK</b> and <b>scikit-learn</b> libraries, '
    'operating on the <b>Reuters-21578 corpus</b> — a benchmark dataset '
    'of 10,788 newswire articles from Reuters distributed across a wide range of financial '
    'and economic topics.', S_BODY))
inner.append(spacer(0.3))
inner.append(info_box(
    '📌 All three tasks are implemented as independent Python scripts plus a unified '
    'Jupyter Notebook (NLP_Assignment.ipynb) usable in Google Colab. '
    'An interactive HTML dashboard (report.html) with live keyword analyzer and '
    'cluster predictor is also included.'
))
inner.append(spacer(0.3))

# Quick stats table
inner.append(colored_table(
    [['Task', 'Method', 'Key Libraries', 'Result'],
     ['A — Clustering',        'K-Means + Cosine Similarity', 'sklearn, NLTK', f'{K} clusters on 10,788 docs'],
     ['B — TF-IDF Keywords',   'TF-IDF + Percentile Split',   'sklearn, numpy',f'{len(KEYWORDS)} keywords classified'],
     ['C — Similarity Search', 'Cosine Similarity (no stopwords)','sklearn',   f'{matches:,} docs above {PERCENTILE}th pct'],
    ],
    col_widths=[3.5*cm, 5.5*cm, 3.5*cm, 4.5*cm],
    font_size=8.5
))
inner.append(spacer(0.3))

# ── 2. TECHNICAL STACK ────────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('2', '🔧', 'Technical Stack', colors.HexColor('#6e40c9')))
inner.append(spacer(0.4))
tech = [
    ['Component', 'Technology', 'Purpose'],
    ['Language',        'Python 3.9',                 'Primary implementation language'],
    ['NLP Library',     'NLTK (Natural Language Toolkit)', 'Corpus access & tokenization'],
    ['ML Library',      'scikit-learn 1.x',           'TF-IDF, K-Means, cosine similarity'],
    ['Numerics',        'NumPy',                       'Array operations & percentiles'],
    ['Visualization',   'matplotlib, seaborn, WordCloud','Charts & word cloud generation'],
    ['Corpus',          'Reuters-21578 (nltk.corpus.reuters)','10,788 newswire articles'],
    ['Notebook',        'Jupyter / Google Colab',      'Interactive execution environment'],
    ['Dashboard',       'Plotly (JS) + custom HTML/CSS','Interactive web report'],
]
inner.append(colored_table(tech, [3*cm, 5.5*cm, 8.5*cm],
             header_color=colors.HexColor('#6e40c9'), font_size=8.5))
inner.append(PageBreak())

# ── 3. TASK A ────────────────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('3', '🔵', 'Task A — Corpus Clustering', BLUE_D))
inner.append(spacer(0.4))

inner.append(sub_header('3.1 Methodology'))
inner.append(Paragraph(
    'Task A requires clustering the Reuters corpus into a specified number of classes '
    'based on <b>cosine similarity</b>. The implementation uses the following pipeline:', S_BODY))
steps = [
    ('Step 1 — TF-IDF Vectorization',
     'All 10,788 Reuters documents are converted to TF-IDF vectors using sklearn\'s '
     'TfidfVectorizer (max 10,000 features, sublinear TF scaling, min_df=3).'),
    ('Step 2 — L2 Normalization',
     'All TF-IDF vectors are normalized to unit length using L2 norm. '
     'This means the dot product between any two vectors equals their cosine similarity.'),
    ('Step 3 — K-Means Clustering',
     'K-Means (k-means++ initialization, n_init=10) is applied on the normalized vectors. '
     'Minimizing Euclidean distance on L2-normalized vectors is mathematically equivalent '
     'to maximizing cosine similarity — satisfying the assignment requirement.'),
]
for title, desc in steps:
    inner.append(Table([[
        Paragraph(f'<b>{title}</b><br/>{desc}',
                  sty('step', fontName='Helvetica', fontSize=9.5,
                      textColor=colors.HexColor('#1f2328'), leading=15))
    ]], colWidths=[W-2*M-0.5*cm],
        style=TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#f0f6ff')),
            ('LEFTPADDING',   (0,0),(-1,-1), 12),
            ('RIGHTPADDING',  (0,0),(-1,-1), 12),
            ('TOPPADDING',    (0,0),(-1,-1), 7),
            ('BOTTOMPADDING', (0,0),(-1,-1), 7),
            ('LINEBEFORE', (0,0),(0,-1), 3, BLUE_D),
            ('ROUNDEDCORNERS',[4]),
        ])))
    inner.append(spacer(0.2))

inner.append(Paragraph(
    f'<b>Configuration used:</b>  K = {K} clusters, 10,788 documents, '
    f'TF-IDF vocabulary size = 10,000 features.', S_BODY))
inner.append(spacer(0.3))

inner.append(sub_header('3.2 Results'))
# Cluster results table
cdata = [['Cluster', '# Documents', '% of Corpus', 'Top Keywords']]
for c in clusters:
    pct = c["docs"] / len(documents) * 100
    cdata.append([
        f'Cluster {c["id"]}',
        f'{c["docs"]:,}',
        f'{pct:.1f}%',
        ', '.join(c["terms"][:6])
    ])
inner.append(colored_table(cdata, [2.5*cm, 3*cm, 3*cm, 8.5*cm], font_size=8.5))
inner.append(Paragraph('Table 1: Cluster composition and top TF-IDF terms per cluster.', S_CAPTION))

# Charts side-by-side
inner.append(spacer(0.3))
charts_row = Table([[make_pie(), make_cluster_bar()]],
                   colWidths=[6.5*cm, 10.5*cm],
                   style=TableStyle([
                       ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f6f8fa')),
                       ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#d0d7de')),
                       ('ROUNDEDCORNERS',[6]),
                       ('ALIGN',(0,0),(-1,-1),'CENTER'),
                       ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                   ]))
inner.append(charts_row)
inner.append(Paragraph('Figure 1: Cluster size distribution (pie) and document counts per cluster (bar).', S_CAPTION))
inner.append(PageBreak())

# ── 4. TASK B ────────────────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('4', '🟡', 'Task B — Keyword TF-IDF Classification', colors.HexColor('#9e6a03')))
inner.append(spacer(0.4))

inner.append(sub_header('4.1 Methodology'))
inner.append(Paragraph(
    'Task B accepts a file of keywords and computes the <b>mean TF-IDF score</b> '
    'for each keyword across all documents in the Reuters corpus. '
    'Keywords are then classified into three tiers using a <b>10-80-10 percentile split</b>:', S_BODY))
inner.append(spacer(0.2))

tier_table = Table([
    ['Tier', 'Condition', 'Meaning', 'Icon'],
    ['TOP',    f'Score ≥ {P90:.5f}  (90th percentile)', 'High discriminative power in corpus', '⭐'],
    ['MEDIUM', f'{P10:.5f} ≤ Score < {P90:.5f}',        'Moderate presence across documents',  '📊'],
    ['BOTTOM', f'Score < {P10:.5f}  (10th percentile)', 'Low or rare occurrence in corpus',    '🔻'],
], colWidths=[2.5*cm, 6.5*cm, 6.5*cm, 1.5*cm],
   style=TableStyle([
       ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#9e6a03')),
       ('BACKGROUND',  (0,1),(-1,1), colors.HexColor('#fff8e1')),
       ('BACKGROUND',  (0,2),(-1,2), colors.HexColor('#f0f6ff')),
       ('BACKGROUND',  (0,3),(-1,3), colors.HexColor('#fff0f0')),
       ('TEXTCOLOR',   (0,0),(-1,0), WHITE),
       ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
       ('FONTNAME',    (0,1),(-1,-1),'Helvetica'),
       ('FONTSIZE',    (0,0),(-1,-1), 9),
       ('GRID',        (0,0),(-1,-1), 0.4, colors.HexColor('#d0d7de')),
       ('LEFTPADDING', (0,0),(-1,-1), 8),
       ('RIGHTPADDING',(0,0),(-1,-1), 8),
       ('TOPPADDING',  (0,0),(-1,-1), 5),
       ('BOTTOMPADDING',(0,0),(-1,-1), 5),
       ('ALIGN',       (3,0),(-1,-1), 'CENTER'),
       ('ROUNDEDCORNERS',[4]),
   ]))
inner.append(tier_table)
inner.append(spacer(0.3))

inner.append(sub_header('4.2 Results Table'))
lbl_colors = {
    'TOP':    colors.HexColor('#6f4800'),
    'MEDIUM': colors.HexColor('#0550ae'),
    'BOTTOM': colors.HexColor('#82071e'),
}
kw_rows_data = [['Rank', 'Keyword', 'Mean TF-IDF Score', 'Class']]
for i,(kw,sc,lb) in enumerate(kw_data, 1):
    kw_rows_data.append([str(i), kw, f'{sc:.6f}', lb])
kw_tbl_style = TableStyle([
    ('BACKGROUND',  (0,0),(-1,0), colors.HexColor('#9e6a03')),
    ('TEXTCOLOR',   (0,0),(-1,0), WHITE),
    ('FONTNAME',    (0,0),(-1,0), 'Helvetica-Bold'),
    ('FONTNAME',    (0,1),(-1,-1),'Helvetica'),
    ('FONTSIZE',    (0,0),(-1,-1), 9),
    ('ALIGN',       (0,0),(-1,-1), 'LEFT'),
    ('ALIGN',       (2,0),(-1,-1), 'CENTER'),
    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#fffdf0'),WHITE]),
    ('GRID',        (0,0),(-1,-1), 0.4, colors.HexColor('#d0d7de')),
    ('LEFTPADDING', (0,0),(-1,-1), 8),
    ('RIGHTPADDING',(0,0),(-1,-1), 8),
    ('TOPPADDING',  (0,0),(-1,-1), 5),
    ('BOTTOMPADDING',(0,0),(-1,-1), 5),
    ('ROUNDEDCORNERS',[4]),
])
for i,(kw,sc,lb) in enumerate(kw_data, 1):
    kw_tbl_style.add('TEXTCOLOR', (3,i),(3,i), lbl_colors[lb])
    kw_tbl_style.add('FONTNAME',  (3,i),(3,i), 'Helvetica-Bold')
    if lb=='TOP':
        kw_tbl_style.add('BACKGROUND',(0,i),(-1,i),colors.HexColor('#fffde0'))
    elif lb=='BOTTOM':
        kw_tbl_style.add('BACKGROUND',(0,i),(-1,i),colors.HexColor('#fff5f5'))
kw_table = Table(kw_rows_data, colWidths=[1.5*cm, 4*cm, 5.5*cm, 3*cm],
                 style=kw_tbl_style, hAlign='LEFT')
inner.append(kw_table)
inner.append(Paragraph('Table 2: TF-IDF scores and classification for all 20 keywords.', S_CAPTION))
inner.append(spacer(0.3))
inner.append(make_tfidf_bar())
inner.append(Paragraph('Figure 2: TF-IDF score per keyword. Gold = TOP, Blue = MEDIUM, Red = BOTTOM.', S_CAPTION))
inner.append(PageBreak())

# ── 5. TASK C ────────────────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('5', '🟢', 'Task C — Document Similarity Search', colors.HexColor('#1a7f37')))
inner.append(spacer(0.4))

inner.append(sub_header('5.1 Methodology'))
inner.append(Paragraph(
    'Task C accepts a query document and a match percentile threshold, '
    'then retrieves all corpus documents whose <b>cosine similarity</b> '
    'with the query exceeds the score at that percentile. '
    '<b>No stopword removal</b> is applied, as specified in the assignment.', S_BODY))
inner.append(spacer(0.2))

inner.append(info_box(
    f'⚙️  Configuration: Percentile = {PERCENTILE}  ·  '
    f'Threshold = {threshold:.6f}  ·  Documents above threshold = {matches:,}',
    bg=colors.HexColor('#dafbe1'), border=GREEN
))
inner.append(spacer(0.3))

# Query document display
inner.append(Table([[
    Paragraph('<b>Query Document (input):</b><br/>' + USER_DOC,
              sty('qd', fontName='Helvetica-Oblique', fontSize=9.5,
                  textColor=colors.HexColor('#1f2328'), leading=15))
]], colWidths=[W-2*M-0.5*cm],
    style=TableStyle([
        ('BACKGROUND',   (0,0),(-1,-1), colors.HexColor('#f1f8ff')),
        ('LEFTPADDING',  (0,0),(-1,-1), 12),
        ('RIGHTPADDING', (0,0),(-1,-1), 12),
        ('TOPPADDING',   (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1), 8),
        ('LINEBEFORE', (0,0),(0,-1), 3, GREEN),
        ('ROUNDEDCORNERS',[4]),
    ])))
inner.append(spacer(0.3))

inner.append(sub_header('5.2 Results', color=colors.HexColor('#1a7f37')))
sim_data = [['Rank', 'Document ID', 'Cosine Similarity', 'Category']]
for i, (did, sc) in enumerate(top_docs, 1):
    cats = ', '.join(reuters.categories(did)[:3])
    sim_data.append([str(i), did, f'{sc:.6f}', cats or '—'])
inner.append(colored_table(sim_data, [1.5*cm, 5.5*cm, 4.5*cm, 5.5*cm],
             header_color=colors.HexColor('#1a7f37'), font_size=8.5))
inner.append(Paragraph('Table 3: Top-10 most similar documents with cosine similarity scores.', S_CAPTION))
inner.append(spacer(0.3))
inner.append(make_sim_bar())
inner.append(Paragraph('Figure 3: Top-10 document similarity scores (descending). Green intensity reflects rank.', S_CAPTION))
inner.append(PageBreak())

# ── 6. CONCLUSION ────────────────────────────────────────────
inner.append(spacer(0.3))
inner.append(section_title('6', '✅', 'Conclusion', colors.HexColor('#6e40c9')))
inner.append(spacer(0.4))
inner.append(Paragraph(
    'All three general assignment tasks have been successfully implemented '
    'using Python, NLTK, and scikit-learn on the Reuters-21578 benchmark corpus. '
    'Each script is self-contained, interactive, and well-documented.', S_BODY))
inner.append(spacer(0.3))

summary_data = [
    ['✅', 'Task A', f'Successfully clustered 10,788 Reuters documents into {K} groups using K-Means on normalized TF-IDF vectors (cosine similarity equivalent).'],
    ['✅', 'Task B', f'Classified {len(KEYWORDS)} keywords into TOP/MEDIUM/BOTTOM tiers (10-80-10 percentile) based on mean TF-IDF scores across corpus.'],
    ['✅', 'Task C', f'Identified {matches:,} documents above the {PERCENTILE}th percentile similarity threshold for a sample query document.'],
]
inner.append(Table(summary_data, colWidths=[1.2*cm, 2.5*cm, 13.3*cm],
    style=TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), colors.HexColor('#f0fff4')),
        ('FONTNAME',   (0,0),(-1,-1),'Helvetica'),
        ('FONTNAME',   (1,0),(1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 9.5),
        ('TEXTCOLOR',  (0,0),(0,-1), GREEN),
        ('TEXTCOLOR',  (1,0),(1,-1), colors.HexColor('#1a7f37')),
        ('LINEBELOW',  (0,0),(-1,-2), 0.5, colors.HexColor('#d0d7de')),
        ('LEFTPADDING',(0,0),(-1,-1), 10),
        ('RIGHTPADDING',(0,0),(-1,-1),10),
        ('TOPPADDING', (0,0),(-1,-1), 8),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('VALIGN',     (0,0),(-1,-1),'TOP'),
        ('ROUNDEDCORNERS',[6]),
        ('BOX', (0,0),(-1,-1), 1, colors.HexColor('#2ea043')),
    ])))
inner.append(spacer(0.5))

# Deliverables
inner.append(sub_header('📁 Project Deliverables'))
files = [
    ('task_a_clustering.py',      'Task A standalone script — interactive K-Means clustering'),
    ('task_b_keyword_tfidf.py',   'Task B standalone script — keyword TF-IDF classification'),
    ('task_c_similarity_search.py','Task C standalone script — document similarity search'),
    ('NLP_Assignment.ipynb',      'Unified Colab notebook with advanced visualizations (word clouds, t-SNE, heatmaps)'),
    ('dashboard.py',              'Generates interactive HTML dashboard with live keyword analyzer'),
    ('report.html',               'Self-contained interactive web dashboard'),
    ('NLP_Project_Report.pdf',    'This project report document'),
]
for fname, desc in files:
    inner.append(Paragraph(f'• <font name="Courier" size="9" color="#0550ae">{fname}</font>'
                           f'  — {desc}', S_BULLET))
inner.append(spacer(0.5))

# GitHub
inner.append(info_box(
    '🔗 GitHub Repository: <font name="Courier">https://github.com/abrham-cyper/NLP-2</font>  |  '
    'Online Dashboard: <font name="Courier">https://abrham-cyper.github.io/NLP-2/</font>',
    bg=colors.HexColor('#ddf4ff'), border=BLUE_D
))

# ── ASSEMBLE DOCUMENT ─────────────────────────────────────────
print("Assembling and rendering PDF...")

class TwoTemplatePDF(BaseDocTemplate):
    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        cover_frame = Frame(0, 0, W, H, leftPadding=M, rightPadding=M,
                            topPadding=0, bottomPadding=0, id='cover')
        inner_frame = Frame(M + 3, 1.8*cm, W - 2*M - 3, H - 3.8*cm,
                            id='inner')
        self.addPageTemplates([
            PageTemplate(id='Cover', frames=[cover_frame], onPage=cover_bg),
            PageTemplate(id='Inner', frames=[inner_frame], onPage=inner_bg),
        ])

doc = TwoTemplatePDF(OUT, pagesize=A4, title='NLP Assignment Report',
                     author='Abrham Assefa Habtamu',
                     subject='NLP Course General Assignment')

from reportlab.platypus import NextPageTemplate
full_story = (
    [NextPageTemplate('Cover')] + cover +
    [NextPageTemplate('Inner')] + inner
)
doc.build(full_story)

size_kb = os.path.getsize(OUT) // 1024
print(f"\n✅  {OUT} created ({size_kb} KB)")
print(f"    Open with: open '{OUT}'")
