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
        
        color_palettes = {
            'vivid': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'],
            'pastel': px.colors.qualitative.Pastel,
            'bold': px.colors.qualitative.Bold,
            'plotly': px.colors.qualitative.Plotly,
            'set1': px.colors.qualitative.Set1,
            'set2': px.colors.qualitative.Set2,
            'set3': px.colors.qualitative.Set3,
        }

        # 1. Histogram with Rug - First numerical column
        try:
            if len(num_cols) >= 1:
                col = num_cols[0]
                fig = px.histogram(df, x=col, marginal="rug", title=f'📊 Frequency Distribution: {col}',
                                   nbins=40, color_discrete_sequence=['#FF6B6B'])
                fig.update_layout(**dark, height=400, bargap=0.1)
                charts.append({'title': 'Histogram + Rug', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Histogram: {e}")

        # 2. Violin Plots for multiple numerical columns
        try:
            if len(num_cols) >= 1:
                plot_cols = num_cols[:min(4, len(num_cols))]
                fig = px.violin(df, y=plot_cols, box=True, points='all',
                                title='🎻 Distribution & Density (Violin)',
                                color_discrete_sequence=color_palettes['vivid'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Violin Plots', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Violin: {e}")

        # 3. Scatter Plot with Trendline
        try:
            if len(num_cols) >= 2:
                try:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1], trendline="ols",
                                     title=f'📈 Correlation: {num_cols[0]} vs {num_cols[1]}',
                                     color_discrete_sequence=['#4ECDC4'])
                    fig.update_traces(marker=dict(size=8, opacity=0.7))
                except Exception:
                    fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                                     title=f'📈 Scatter: {num_cols[0]} vs {num_cols[1]}',
                                     color_discrete_sequence=['#4ECDC4'])
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Scatter Plot', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Scatter: {e}")

        # 4. Box Plot for Outlier Detection
        try:
            if len(num_cols) >= 1:
                # Use different columns if available
                box_cols = num_cols[1:5] if len(num_cols) > 2 else num_cols[:2]
                fig = px.box(df[box_cols], title='📦 Outlier Detection (Box Plots)',
                             color_discrete_sequence=color_palettes['pastel'])
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Box Plot Analysis', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Box: {e}")

        # 5. Area Chart 
        try:
            if len(num_cols) >= 2:
                col1 = num_cols[0]
                col2 = num_cols[1]
                fig = px.area(df.head(50), x=df.index[:50], y=[col1, col2], 
                              title=f'⛰️ Area Chart (First 50 Rows): {col1} & {col2}',
                              color_discrete_sequence=color_palettes['set2'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Area Chart', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Area: {e}")

        # 6. Bar Chart for Categorical Data
        try:
            if len(cat_cols) >= 1:
                col = cat_cols[0]
                vc = df[col].value_counts().head(12)
                fig = px.bar(x=vc.index.astype(str), y=vc.values,
                             title=f'📊 Category Distribution: {col}',
                             color_discrete_sequence=['#BB8FCE'])
                fig.update_layout(**dark, height=400, xaxis_tickangle=45)
                charts.append({'title': f'Bar Chart: {col}', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Bar: {e}")

        # 7. Donut Chart (Pie with hole)
        try:
            if len(cat_cols) >= 1:
                # Try to use a different categorical column
                col = cat_cols[1] if len(cat_cols) > 1 else cat_cols[0]
                vc = df[col].value_counts().head(8)
                fig = px.pie(values=vc.values, names=vc.index.astype(str), hole=0.5,
                             title=f'🍩 Proportion (Donut): {col}',
                             color_discrete_sequence=color_palettes['bold'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Donut Chart', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Donut: {e}")

        # 8. Standard Pie Chart
        try:
            if len(cat_cols) >= 1:
                # Try to use a different categorical column
                col = cat_cols[2] if len(cat_cols) > 2 else cat_cols[0]
                vc = df[col].value_counts().head(8)
                fig = px.pie(values=vc.values, names=vc.index.astype(str),
                             title=f'🍕 Proportion (Pie): {col}',
                             color_discrete_sequence=color_palettes['set3'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Pie Chart', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Pie: {e}")

        # 9. Correlation Heatmap
        try:
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().round(2)
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                title='🔥 Correlation Heatmap',
                                color_continuous_midpoint=0)
                fig.update_layout(**dark, height=450)
                charts.append({'title': 'Correlation Matrix', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Heatmap: {e}")

        # 10. Line Chart
        try:
            if len(num_cols) >= 1:
                col = num_cols[-1] # Use the last num col for variety
                fig = px.line(df.head(100), y=col, title=f'📈 Line Chart (First 100 Rows): {col}',
                              color_discrete_sequence=['#F7DC6F'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Line Chart', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Line: {e}")

        # 11. Funnel Chart (if categorical data exists)
        try:
            if len(cat_cols) >= 1:
                col = cat_cols[0]
                vc = df[col].value_counts().head(6).reset_index()
                vc.columns = [col, 'count']
                fig = px.funnel(vc, x='count', y=col, title=f'🔽 Funnel Chart: {col}',
                                color_discrete_sequence=color_palettes['vivid'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Funnel Chart', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Funnel: {e}")

        # 12. 2D Density Heatmap
        try:
            if len(num_cols) >= 2:
                fig = px.density_heatmap(df, x=num_cols[0], y=num_cols[1],
                                         title=f'🔥 2D Density: {num_cols[0]} vs {num_cols[1]}',
                                         nbinsx=25, nbinsy=25, color_continuous_scale='Viridis')
                fig.update_layout(**dark, height=450)
                charts.append({'title': '2D Density Heatmap', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error 2D Density: {e}")

        # 13. Strip Plot (Scatter for categories)
        try:
            if len(num_cols) >= 1 and len(cat_cols) >= 1:
                num_col = num_cols[0]
                cat_col = cat_cols[0]
                fig = px.strip(df, x=cat_col, y=num_col,
                               title=f'⚡ Strip Plot: {num_col} by {cat_col}',
                               color_discrete_sequence=color_palettes['set1'])
                fig.update_layout(**dark, height=400)
                charts.append({'title': 'Strip Plot', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error Strip: {e}")

        # 14. ECDF (Empirical Cumulative Distribution Function)
        try:
            if len(num_cols) >= 1:
                col = num_cols[0]
                fig = px.ecdf(df, x=col, title=f'📉 ECDF Curve: {col}',
                              color_discrete_sequence=['#85C1E2'])
                fig.update_layout(**dark, height=350)
                charts.append({'title': 'ECDF Curve', 'json': fig.to_json()})
        except Exception as e:
            print(f"Error ECDF: {e}")

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
    try:
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
        h4 = ParagraphStyle('H4', fontSize=9, textColor=TEXT,
                            spaceAfter=2, spaceBefore=4, fontName='Helvetica-Bold')
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
        story.append(Paragraph("Comprehensive Dataset Analysis Report", ParagraphStyle(
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
            ['Rows', 'Columns', 'Numerical', 'Categorical', 'Null Values', 'Memory'],
            [str(df.shape[0]), str(df.shape[1]), str(len(num_cols)),
             str(len(cat_cols)), str(int(df.isnull().sum().sum())), 
             f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB"]
        ]
        kpi_table = Table(kpi_data, colWidths=[2.2*cm]*6)
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), CARD),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#21262d')),
            ('TEXTCOLOR', (0, 0), (-1, 0), MUTED),
            ('TEXTCOLOR', (0, 1), (-1, 1), ACCENT),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, 1), 14),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 0.5, ACCENT),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#30363d')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Contents: Dataset Details  -  Column Overview  -  Statistics  -  Visual Insights  -  Correlations  -  Summary", 
                              ParagraphStyle('Contents', fontSize=7, textColor=MUTED, fontName='Helvetica', leading=10)))
        story.append(PageBreak())

        # ---- PAGE 2: Detailed Dataset Info ----
        story.append(Paragraph("1. Detailed Dataset Information", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("Dataset Dimensions & Composition", h3))
        story.append(Paragraph(f"<b>Total Records:</b> {df.shape[0]:,} rows", body))
        story.append(Paragraph(f"<b>Total Features:</b> {df.shape[1]} columns", body))
        story.append(Paragraph(f"<b>Memory Usage:</b> {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB", body))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("Column Breakdown", h3))
        story.append(Paragraph(f"<b>Numerical Columns:</b> {len(num_cols)}", body))
        if num_cols:
            story.append(Paragraph(f"{', '.join(num_cols[:10])}", 
                                  ParagraphStyle('Small', fontSize=8, textColor=MUTED, fontName='Helvetica')))
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph(f"<b>Categorical Columns:</b> {len(cat_cols)}", body))
        if cat_cols:
            story.append(Paragraph(f"{', '.join(cat_cols[:10])}", 
                                  ParagraphStyle('Small', fontSize=8, textColor=MUTED, fontName='Helvetica')))
        story.append(Spacer(1, 0.12*inch))
        
        story.append(Paragraph("Data Quality Metrics", h3))
        total_cells = df.shape[0] * df.shape[1]
        null_cells = int(df.isnull().sum().sum())
        completeness = ((total_cells - null_cells) / total_cells * 100)
        story.append(Paragraph(f"<b>Total Cells:</b> {total_cells:,}", body))
        story.append(Paragraph(f"<b>Missing Values:</b> {null_cells:,} ({100 - completeness:.2f}%)", body))
        story.append(Paragraph(f"<b>Completeness:</b> {completeness:.2f}%", body))
        story.append(Spacer(1, 0.12*inch))
        
        story.append(Paragraph("Null Values Distribution", h3))
        null_data = [['Column', 'Nulls', 'Percentage']]
        null_cols = df.columns[df.isnull().sum() > 0].tolist()
        for col in null_cols[:15]:
            null_count = int(df[col].isnull().sum())
            null_pct = (null_count / len(df)) * 100
            null_data.append([col[:20], str(null_count), f'{null_pct:.1f}%'])
        
        if null_data != [['Column', 'Nulls', 'Percentage']]:
            null_table = Table(null_data, colWidths=[5*cm, 2.5*cm, 2.5*cm])
            null_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), CARD),
                ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
                ('TEXTCOLOR', (0, 1), (-1, -1), TEXT),
                ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
            ]))
            story.append(null_table)
        else:
            story.append(Paragraph("No missing values detected!", ParagraphStyle('Success', fontSize=9, textColor=colors.HexColor('#7ee787'), fontName='Helvetica')))
        
        story.append(PageBreak())

        # ---- PAGE 3: Column Overview ----
        story.append(Paragraph("2. Column Overview", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))

        col_data = [['Column', 'Type', 'Dtype', 'Unique', 'Nulls', 'Min/Max/Mode']]
        for col in df.columns[:25]:
            ctype = 'Numerical' if col in num_cols else 'Categorical'
            if col in num_cols:
                min_val = f"{df[col].min():.2f}"
                max_val = f"{df[col].max():.2f}"
                info = f"{min_val} to {max_val}"
            else:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'N/A'
                info = str(mode_val)[:15]
            
            col_data.append([col[:20], ctype, str(df[col].dtype),
                            str(df[col].nunique()), str(int(df[col].isnull().sum())), info])

        col_table = Table(col_data, colWidths=[3.5*cm, 2*cm, 2*cm, 1.8*cm, 1.5*cm, 2.7*cm])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), CARD),
            ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
            ('TEXTCOLOR', (0, 1), (-1, -1), TEXT),
            ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        story.append(col_table)
        story.append(PageBreak())

        # ---- PAGE 4: Statistical Summary ----
        story.append(Paragraph("3. Statistical Summary", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))
        
        if num_cols:
            story.append(Paragraph("Numerical Statistics", h3))
            desc = df[num_cols].describe().round(3)
            desc_data = [['Metric'] + num_cols[:7]]
            for idx in desc.index:
                row = [str(idx)]
                for col in num_cols[:7]:
                    row.append(str(desc.loc[idx, col]))
                desc_data.append(row)
            
            desc_table = Table(desc_data, colWidths=[1.5*cm] + [1.7*cm]*7)
            desc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), CARD),
                ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
                ('TEXTCOLOR', (0, 1), (-1, -1), TEXT),
                ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(desc_table)
            story.append(Spacer(1, 0.2*inch))

        if cat_cols:
            story.append(Paragraph("Categorical Statistics", h3))
            for col in cat_cols[:3]:
                vc = df[col].value_counts().head(5)
                story.append(Paragraph(f"<b>{col}</b>", h4))
                cat_stat_data = [['Value', 'Count', 'Percentage']]
                for val, count in vc.items():
                    pct = (count / len(df)) * 100
                    cat_stat_data.append([str(val)[:25], str(count), f'{pct:.1f}%'])
                
                cat_table = Table(cat_stat_data, colWidths=[5*cm, 2.5*cm, 2.5*cm])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), CARD),
                    ('TEXTCOLOR', (0, 0), (-1, 0), ACCENT),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
                    ('TEXTCOLOR', (0, 1), (-1, -1), TEXT),
                    ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
                ]))
                story.append(cat_table)
                story.append(Spacer(1, 0.1*inch))
        
        story.append(PageBreak())

        # ---- PAGE 5: Visual Insights ----
        story.append(Paragraph("4. Visual Insights", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))
        
        # Histogram
        if num_cols:
            try:
                fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0d1117')
                ax.set_facecolor('#161b22')
                df[num_cols[0]].dropna().hist(ax=ax, color='#FF6B6B', bins=40, edgecolor='#30363d')
                ax.set_title(f"Distribution of {num_cols[0]}", color='#e6edf3', fontsize=11, fontweight='bold')
                ax.set_xlabel(num_cols[0], color='#8b949e', fontsize=9)
                ax.set_ylabel('Frequency', color='#8b949e', fontsize=9)
                ax.tick_params(colors='#8b949e', labelsize=8)
                ax.spines['bottom'].set_color('#30363d')
                ax.spines['left'].set_color('#30363d')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                img_bytes = _mpl_fig_to_bytes(fig)
                plt.close(fig)
                story.append(Paragraph("Histogram", h3))
                story.append(RLImage(io.BytesIO(img_bytes), width=13*cm, height=5.5*cm))
                story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                print(f"Error creating histogram: {e}")
        
        # Box plot for multiple columns
        if len(num_cols) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0d1117')
                ax.set_facecolor('#161b22')
                cols_to_plot = num_cols[:5]
                df[cols_to_plot].boxplot(ax=ax, patch_artist=True)
                ax.set_title("Outlier Detection (Box Plots)", color='#e6edf3', fontsize=11, fontweight='bold')
                ax.set_ylabel('Value', color='#8b949e', fontsize=9)
                ax.tick_params(colors='#8b949e', labelsize=8)
                ax.spines['bottom'].set_color('#30363d')
                ax.spines['left'].set_color('#30363d')
                for patch in ax.artists:
                    patch.set_facecolor('#4ECDC4')
                img_bytes = _mpl_fig_to_bytes(fig)
                plt.close(fig)
                story.append(Paragraph("Box Plots", h3))
                story.append(RLImage(io.BytesIO(img_bytes), width=13*cm, height=5.5*cm))
            except Exception as e:
                print(f"Error creating box plot: {e}")
        
        story.append(PageBreak())

        # ---- PAGE 6: Correlations & Advanced Insights ----
        story.append(Paragraph("5. Correlation Analysis", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))
        
        if len(num_cols) >= 2:
            try:
                corr = df[num_cols].corr().round(2)
                
                # Scatter plot for top 2 numerical columns
                fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0d1117')
                ax.set_facecolor('#161b22')
                ax.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6, color='#45B7D1', s=30, edgecolors='#30363d', linewidth=0.5)
                ax.set_xlabel(num_cols[0], color='#8b949e', fontsize=9)
                ax.set_ylabel(num_cols[1], color='#8b949e', fontsize=9)
                ax.set_title(f"Relationship: {num_cols[0]} vs {num_cols[1]}", color='#e6edf3', fontsize=11, fontweight='bold')
                ax.tick_params(colors='#8b949e', labelsize=8)
                ax.spines['bottom'].set_color('#30363d')
                ax.spines['left'].set_color('#30363d')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.1, color='#30363d')
                img_bytes = _mpl_fig_to_bytes(fig)
                plt.close(fig)
                story.append(Paragraph("Scatter Plot Analysis", h3))
                story.append(RLImage(io.BytesIO(img_bytes), width=13*cm, height=6*cm))
                story.append(Spacer(1, 0.15*inch))
                
                # Correlation matrix
                story.append(Paragraph("Correlation Matrix", h3))
                corr_data = [[''] + num_cols[:8]]
                for idx, row_name in enumerate(num_cols[:8]):
                    row = [row_name]
                    for col_name in num_cols[:8]:
                        row.append(f"{corr.loc[row_name, col_name]:.2f}")
                    corr_data.append(row)
                
                corr_table = Table(corr_data, colWidths=[1.8*cm]*9)
                corr_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), CARD),
                    ('BACKGROUND', (0, 0), (0, -1), CARD),
                    ('TEXTCOLOR', (0, 0), (-1, -1), ACCENT),
                    ('FONTSIZE', (0, 0), (-1, -1), 6),
                    ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
                    ('TEXTCOLOR', (1, 1), (-1, -1), TEXT),
                    ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#21262d')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ]))
                story.append(corr_table)
            except Exception as e:
                print(f"Error creating correlation analysis: {e}")
        
        story.append(PageBreak())

        # ---- PAGE 7: Summary & Recommendations ----
        story.append(Paragraph("6. Summary & Key Findings", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph("<b>Dataset Overview</b>", h3))
        story.append(Paragraph(f"Your dataset contains {df.shape[0]:,} records with {df.shape[1]} features spanning both numerical and categorical data types.", body))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Data Quality Assessment</b>", h3))
        if completeness >= 95:
            story.append(Paragraph(f"✓ Excellent data quality with {completeness:.1f}% completeness.", 
                                  ParagraphStyle('Good', fontSize=9, textColor=colors.HexColor('#7ee787'), fontName='Helvetica')))
        elif completeness >= 80:
            story.append(Paragraph(f"⚠ Good data quality with {completeness:.1f}% completeness. Some missing values detected.", 
                                  ParagraphStyle('Warn', fontSize=9, textColor=colors.HexColor('#d29922'), fontName='Helvetica')))
        else:
            story.append(Paragraph(f"✗ Data quality needs attention. Only {completeness:.1f}% completeness.", 
                                  ParagraphStyle('Bad', fontSize=9, textColor=colors.HexColor('#f85149'), fontName='Helvetica')))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Column Statistics</b>", h3))
        story.append(Paragraph(f"• Numerical columns: {len(num_cols)}", body))
        story.append(Paragraph(f"• Categorical columns: {len(cat_cols)}", body))
        story.append(Paragraph(f"• Unique values range: {df.nunique().min()} to {df.nunique().max()}", body))
        story.append(Spacer(1, 0.1*inch))
        
        story.append(Paragraph("<b>Data Distribution</b>", h3))
        if num_cols:
            story.append(Paragraph(f"Primary numerical variable '{num_cols[0]}' shows ", body))
            skewness = df[num_cols[0]].skew()
            if abs(skewness) < 0.5:
                story.append(Paragraph("a relatively symmetric distribution.", body))
            elif skewness > 0:
                story.append(Paragraph("a right-skewed distribution.", body))
            else:
                story.append(Paragraph("a left-skewed distribution.", body))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Report generated by DataZen • " + datetime.now().strftime('%B %d, %Y at %H:%M'), muted_style))

        doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
    except Exception as e:
        print(f"PDF Build Error: {str(e)}")
        raise


if __name__ == '__main__':
    app.run(debug=True, port=5000)
