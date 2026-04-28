# -*- coding: utf-8 -*-
from datetime import datetime
import io
import os
import uuid
import warnings
import pickle
import gc
import traceback

from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


def df_to_summary(df):
    """Generates a summary dictionary from a dataframe."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    null_info = []

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        col_type = 'numerical' if col in num_cols else 'categorical'
        unique = int(df[col].nunique())

        null_info.append({
            'column': col,
            'type': col_type,
            'dtype': str(df[col].dtype),
            'null_count': null_count,
            'null_pct': null_pct,
            'unique': unique,
            'fill_options': ['mean', 'median'] if col_type == 'numerical' else ['mode'],
            'recommended': 'median' if col_type == 'numerical' else 'mode'
        })

    desc = df[num_cols].describe().round(4).to_dict() if num_cols else {}

    return {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'null_info': null_info,
        'numerical_cols': num_cols,
        'categorical_cols': cat_cols,
        'total_nulls': int(df.isnull().sum().sum()),
        'describe': desc,
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'sample': df.head(5).fillna('').to_dict(orient='records')
    }


def _get_df(sid):
    """Loads a dataframe from disk for a given session ID."""
    path = os.path.join(UPLOAD_FOLDER, f'{sid}.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _save_df(df, sid):
    """Saves a dataframe to disk."""
    path = os.path.join(UPLOAD_FOLDER, f'{sid}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(df, f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV supported'}), 400
    try:
        df = pd.read_csv(f)
        session_id = str(uuid.uuid4())
        _save_df(df, session_id)
        summary = df_to_summary(df)
        del df
        gc.collect()
        return jsonify({'session_id': session_id, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/fill_nulls', methods=['POST'])
def fill_nulls():
    data = request.json
    sid = data.get('session_id')
    strategy = data.get('strategy', {})
    df = _get_df(sid)

    if df is None:
        return jsonify({'error': 'Session expired. Please upload again.'}), 400

    try:
        applied = []
        for col, method in strategy.items():
            if col not in df.columns:
                continue
            before = int(df[col].isnull().sum())
            if before == 0:
                continue

            if method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'mode':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
            elif method == 'drop':
                df = df.dropna(subset=[col])

            after = int(df[col].isnull().sum())
            applied.append({'col': col, 'method': method, 'before': before, 'after': after})

        _save_df(df, sid)
        summary = df_to_summary(df)
        del df
        gc.collect()
        return jsonify({'applied': applied, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard', methods=['POST'])
def dashboard():
    try:
        data = request.json
        sid = data.get('session_id')
        df = _get_df(sid)
        if df is None:
            return jsonify({'error': 'Session expired. Please upload again.'}), 400

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        charts = []
        dark = {
            'paper_bgcolor': '#0d1117',
            'plot_bgcolor': '#161b22',
            'font': {'color': '#e6edf3'}
        }

        # 1. Distribution: Violin Plots
        try:
            for col in num_cols[:3]:
                fig = px.violin(df, y=col, box=True, points='all',
                                title=f'Distribution: {col}',
                                color_discrete_sequence=['#ff7f0e'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': f'Violin: {col}', 'json': fig.to_json()})
        except Exception:
            pass

        # 2. Relationship: Scatter Plots
        try:
            if len(num_cols) >= 2:
                try:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], trendline="ols",
                                     title=f'Relationship: {num_cols[0]} vs {num_cols[1]}')
                except Exception:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                                     title=f'Relationship: {num_cols[0]} vs {num_cols[1]}')
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Scatter Analysis', 'json': fig.to_json()})
        except Exception:
            pass

        # 3. Proportion: Enhanced Pie Chart
        try:
            if cat_cols:
                col = cat_cols[0]
                vc = df[col].value_counts().head(10)
                fig = px.pie(values=vc.values, names=vc.index.astype(str), hole=0.5,
                             title=f'Proportion: {col}',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(**dark, height=400)
                charts.append({'title': f'Pie: {col}', 'json': fig.to_json()})
        except Exception:
            pass

        # 4. Correlation Heatmap
        try:
            if len(num_cols) >= 3:
                corr = df[num_cols].corr().round(2)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                                title='Interactive Correlation Matrix')
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Correlations', 'json': fig.to_json()})
        except Exception:
            pass

        # 5. Data Hierarchy: Sunburst
        try:
            if len(cat_cols) >= 2:
                fig = px.sunburst(df, path=cat_cols[:2], title='Categorical Hierarchy',
                                  color_discrete_sequence=px.colors.qualitative.Prism)
                fig.update_layout(**dark, height=500)
                charts.append({'title': 'Data Hierarchy', 'json': fig.to_json()})
        except Exception:
            pass

        # 6. Outlier Analysis
        try:
            if num_cols:
                fig = px.box(df[num_cols[:6]], title='Outlier Detection (Box Plots)',
                             color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Outliers', 'json': fig.to_json()})
        except Exception:
            pass

        # 7. Histogram with Rug
        try:
            if num_cols:
                col = num_cols[0]
                fig = px.histogram(df, x=col, marginal="rug", title=f'Frequency: {col}',
                                   color_discrete_sequence=['#2ca02c'])
                fig.update_layout(**dark, height=350)
                charts.append({'title': 'Hist + Rug', 'json': fig.to_json()})
        except Exception:
            pass

        del df
        gc.collect()
        return jsonify({'charts': charts})
    except Exception:
        print("DASHBOARD ERROR:", traceback.format_exc())
        return jsonify({'error': 'Failed to generate dashboard'}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        sid = data.get('session_id')
        df = _get_df(sid)
        if df is None:
            return jsonify({'error': 'Session expired. Please upload again.'}), 400
        filename = data.get('filename', 'dataset')

        report_path = os.path.join(REPORT_FOLDER, f'report_{sid}.pdf')
        _build_pdf_report(df, report_path, filename)
        del df
        gc.collect()
        return jsonify({'report_url': f'/download_report/{sid}'})
    except Exception:
        print("REPORT ERROR:", traceback.format_exc())
        return jsonify({'error': 'Failed to generate report'}), 500


@app.route('/download_report/<sid>')
def download_report(sid):
    path = os.path.join(REPORT_FOLDER, f'report_{sid}.pdf')
    if not os.path.exists(path):
        return jsonify({'error': 'Report not found'}), 404
    return send_file(path, as_attachment=True, download_name='DataZen_Report.pdf')


@app.route('/download_clean/<sid>')
def download_clean(sid):
    df = _get_df(sid)
    if df is None:
        return jsonify({'error': 'Session expired'}), 400
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        as_attachment=True,
        download_name='cleaned_dataset.csv',
        mimetype='text/csv'
    )


def _mpl_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _build_pdf_report(df, path, filename):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    doc = SimpleDocTemplate(path, pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    BG = colors.HexColor('#0d1117')
    ACCENT = colors.HexColor('#58a6ff')
    TEXT = colors.HexColor('#e6edf3')
    MUTED = colors.HexColor('#8b949e')
    CARD = colors.HexColor('#161b22')

    h2 = ParagraphStyle('H2', fontSize=15, textColor=ACCENT,
                        spaceAfter=4, spaceBefore=10, fontName='Helvetica-Bold')
    h3 = ParagraphStyle('H3', fontSize=11, textColor=TEXT,
                        spaceAfter=3, spaceBefore=6, fontName='Helvetica-Bold')
    body = ParagraphStyle('Body', fontSize=9, textColor=TEXT,
                          spaceAfter=2, fontName='Helvetica', leading=14)
    muted_style = ParagraphStyle('Muted', fontSize=8, textColor=MUTED,
                                 spaceAfter=2, fontName='Helvetica')

    def page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, A4[1]-3, A4[0], 3, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.setFont('Helvetica', 7)
        canvas.drawString(1.8*cm, 0.8*cm, f'DataZen Report | {filename} | Page {doc.page}')
        canvas.drawRightString(A4[0]-1.8*cm, 0.8*cm, datetime.now().strftime('%B %d, %Y'))
        canvas.restoreState()

    story = []

    # ---- PAGE 1: Cover ----
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("DATAZEN", ParagraphStyle(
        'Cover', fontSize=40, textColor=ACCENT, fontName='Helvetica-Bold', alignment=TA_CENTER)))
    story.append(Paragraph("Dataset Analysis Report", ParagraphStyle(
        'Sub', fontSize=16, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=4)))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="60%", thickness=1, color=ACCENT, hAlign='CENTER'))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"File: <b>{filename}.csv</b>", ParagraphStyle(
        'CoverInfo', fontSize=11, textColor=TEXT, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", ParagraphStyle(
        'CoverDate', fontSize=10, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*inch))

    kpi_data = [
        ['Rows', 'Columns', 'Numerical', 'Categorical', 'Null Values'],
        [str(df.shape[0]), str(df.shape[1]), str(len(num_cols)),
         str(len(cat_cols)), str(int(df.isnull().sum().sum()))]
    ]
    kpi_table = Table(kpi_data, colWidths=[2.8*cm]*5)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#21262d')),
        ('TEXTCOLOR', (0, 0), (-1, 0), MUTED),
        ('TEXTCOLOR', (0, 1), (-1, 1), ACCENT),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, ACCENT),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#30363d')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Contents: Overview  -  Data Quality  -  Stats  -  Visuals", muted_style))
    story.append(PageBreak())

    # ---- PAGE 2: Dataset Overview ----
    story.append(Paragraph("1. Dataset Overview", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(
        f"Shape: <b>{df.shape[0]} rows x {df.shape[1]} columns</b>", body))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Column Information", h3))

    col_data = [['Column', 'Type', 'Dtype', 'Unique', 'Nulls']]
    for col in df.columns[:25]:  # Limit to avoid overflow
        ctype = 'Numerical' if col in num_cols else 'Categorical'
        col_data.append([col[:20], ctype, str(df[col].dtype),
                        str(df[col].nunique()), str(int(df[col].isnull().sum()))])

    col_table = Table(col_data, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), CARD),
        ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), TEXT),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
    ]))
    story.append(col_table)
    story.append(PageBreak())

    # ---- PAGE 3: Visuals ----
    story.append(Paragraph("2. Visual Insights", h2))
    if num_cols:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        df[num_cols[0]].dropna().hist(ax=ax, color='#58a6ff', bins=30)
        ax.set_title(f"Distribution of {num_cols[0]}", color=TEXT)
        ax.tick_params(colors=MUTED)
        img_bytes = _mpl_fig_to_bytes(fig)
        plt.close(fig)
        story.append(RLImage(io.BytesIO(img_bytes), width=14*cm, height=7*cm))

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
