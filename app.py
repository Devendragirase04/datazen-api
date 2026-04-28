from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io, os, json, base64, uuid, warnings
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, HRFlowable
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import tempfile, traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

stored_dfs = {}

def df_to_summary(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    null_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / len(df) * 100, 2)
        dtype = str(df[col].dtype)
        col_type = 'numerical' if col in num_cols else 'categorical'
        unique = int(df[col].nunique())
        null_info.append({
            'column': col,
            'type': col_type,
            'dtype': dtype,
            'null_count': null_count,
            'null_pct': null_pct,
            'unique': unique,
            'fill_options': ['mean', 'median'] if col_type == 'numerical' else ['mode'],
            'recommended': 'median' if col_type == 'numerical' else 'mode'
        })
    
    desc = {}
    if num_cols:
        d = df[num_cols].describe().round(4)
        desc = d.to_dict()
    
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
        stored_dfs[session_id] = df.copy()
        summary = df_to_summary(df)
        return jsonify({'session_id': session_id, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fill_nulls', methods=['POST'])
def fill_nulls():
    data = request.json
    sid = data.get('session_id')
    strategy = data.get('strategy', {})  # {col: method}
    if sid not in stored_dfs:
        return jsonify({'error': 'Session expired'}), 404
    df = stored_dfs[sid].copy()
    applied = []
    for col, method in strategy.items():
        if col not in df.columns:
            continue
        before = int(df[col].isnull().sum())
        if before == 0:
            continue
        if method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
        elif method == 'drop':
            df.dropna(subset=[col], inplace=True)
        after = int(df[col].isnull().sum())
        applied.append({'col': col, 'method': method, 'before': before, 'after': after})
    stored_dfs[sid] = df
    summary = df_to_summary(df)
    return jsonify({'applied': applied, 'summary': summary})

@app.route('/dashboard', methods=['POST'])
def dashboard():
    try:
        data = request.json
        sid = data.get('session_id')
        if sid not in stored_dfs:
            return jsonify({'error': 'Session expired. Please upload again.'}), 400
        df = stored_dfs[sid]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        charts = []

        dark = {'paper_bgcolor': '#0d1117', 'plot_bgcolor': '#161b22', 'font': {'color': '#e6edf3'}}

        # 1. Null heatmap
        null_df = df.isnull().astype(int)
        if null_df.sum().sum() > 0:
            fig = px.imshow(null_df.T, color_continuous_scale=['#161b22', '#f85149'],
                            title='Missing Values Heatmap', aspect='auto',
                            labels={'color': 'Is Null'})
            fig.update_layout(**dark, title_font_size=14, height=350, coloraxis_showscale=False)
            charts.append({'title': 'Missing Values Heatmap', 'json': fig.to_json()})

        # 2. Distribution of numerical cols
        for col in num_cols[:6]:
            fig = px.histogram(df, x=col, nbins=30, title=f'Distribution: {col}',
                               color_discrete_sequence=['#58a6ff'])
            fig.update_layout(**dark, title_font_size=13, height=320, bargap=0.05)
            fig.update_traces(marker_line_color='#21262d', marker_line_width=1)
            charts.append({'title': f'Distribution: {col}', 'json': fig.to_json()})

        # 3. Box plots
        if num_cols:
            fig = px.box(df[num_cols[:8]], title='Box Plots – Numerical Columns',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**dark, title_font_size=14, height=400)
            charts.append({'title': 'Box Plots', 'json': fig.to_json()})

        # 4. Correlation heatmap
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                            title='Correlation Heatmap', zmin=-1, zmax=1)
            fig.update_layout(**dark, title_font_size=14, height=420)
            charts.append({'title': 'Correlation Heatmap', 'json': fig.to_json()})

        # 5. Categorical bar charts
        for col in cat_cols[:4]:
            vc = df[col].value_counts().head(15)
            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f'Value Counts: {col}',
                         labels={'x': col, 'y': 'Count'},
                         color=vc.values, color_continuous_scale='Blues')
            fig.update_layout(**dark, title_font_size=13, height=340, coloraxis_showscale=False)
            charts.append({'title': f'Value Counts: {col}', 'json': fig.to_json()})

        # 6. Pie chart for top categorical
        if cat_cols:
            col = cat_cols[0]
            vc = df[col].value_counts().head(8)
            fig = px.pie(values=vc.values, names=vc.index.astype(str),
                         title=f'Pie Chart: {col}', hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(**dark, title_font_size=13, height=380)
            charts.append({'title': f'Pie: {col}', 'json': fig.to_json()})

        # 7. Scatter matrix
        if len(num_cols) >= 2:
            cols_to_use = num_cols[:4]
            fig = px.scatter_matrix(df[cols_to_use].dropna(), title='Scatter Matrix',
                                    color_discrete_sequence=['#58a6ff'])
            fig.update_layout(**dark, title_font_size=14, height=500)
            charts.append({'title': 'Scatter Matrix', 'json': fig.to_json()})

        # 8. Outlier detection using IQR
        if num_cols:
            outlier_counts = {}
            for col in num_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]
                outlier_counts[col] = len(outliers)
            fig = px.bar(x=list(outlier_counts.keys()), y=list(outlier_counts.values()),
                         title='Outlier Count per Column (IQR Method)',
                         labels={'x': 'Column', 'y': 'Outlier Count'},
                         color=list(outlier_counts.values()), color_continuous_scale='Reds')
            fig.update_layout(**dark, title_font_size=14, height=350, coloraxis_showscale=False)
            charts.append({'title': 'Outlier Counts', 'json': fig.to_json()})

        return jsonify({'charts': charts})
    except Exception as e:
        print("DASHBOARD ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        data = request.json
        sid = data.get('session_id')
        if sid not in stored_dfs:
            return jsonify({'error': 'Session expired. Please upload again.'}), 400
        df = stored_dfs[sid]
        filename = data.get('filename', 'dataset')
        
        report_path = os.path.join(REPORT_FOLDER, f'report_{sid}.pdf')
        _build_pdf_report(df, report_path, filename)
        return jsonify({'report_url': f'/download_report/{sid}'})
    except Exception as e:
        print("REPORT ERROR:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<sid>')
def download_report(sid):
    path = os.path.join(REPORT_FOLDER, f'report_{sid}.pdf')
    if not os.path.exists(path):
        return jsonify({'error': 'Report not found'}), 404
    return send_file(path, as_attachment=True, download_name='DataZen_Report.pdf', mimetype='application/pdf')

@app.route('/download_clean/<sid>')
def download_clean(sid):
    if sid not in stored_dfs:
        return jsonify({'error': 'Session expired'}), 404
    df = stored_dfs[sid]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), as_attachment=True,
                     download_name='cleaned_dataset.csv', mimetype='text/csv')

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
    RED = colors.HexColor('#f85149')
    GREEN = colors.HexColor('#3fb950')

    h1 = ParagraphStyle('H1', fontSize=22, textColor=ACCENT, spaceAfter=6, spaceBefore=4, fontName='Helvetica-Bold', alignment=TA_CENTER)
    h2 = ParagraphStyle('H2', fontSize=15, textColor=ACCENT, spaceAfter=4, spaceBefore=10, fontName='Helvetica-Bold')
    h3 = ParagraphStyle('H3', fontSize=11, textColor=TEXT, spaceAfter=3, spaceBefore=6, fontName='Helvetica-Bold')
    body = ParagraphStyle('Body', fontSize=9, textColor=TEXT, spaceAfter=2, fontName='Helvetica', leading=14)
    muted_style = ParagraphStyle('Muted', fontSize=8, textColor=MUTED, spaceAfter=2, fontName='Helvetica')
    center_style = ParagraphStyle('Center', fontSize=9, textColor=TEXT, alignment=TA_CENTER, fontName='Helvetica')

    def page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, A4[1]-3, A4[0], 3, fill=1, stroke=0)
        canvas.setFillColor(MUTED)
        canvas.setFont('Helvetica', 7)
        canvas.drawString(1.8*cm, 0.8*cm, f'DataZen Report  •  {filename}  •  Page {doc.page}')
        canvas.drawRightString(A4[0]-1.8*cm, 0.8*cm, datetime.now().strftime('%B %d, %Y'))
        canvas.restoreState()

    story = []

    # ---- PAGE 1: Cover ----
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("DATAZEN", ParagraphStyle('Cover', fontSize=40, textColor=ACCENT, fontName='Helvetica-Bold', alignment=TA_CENTER)))
    story.append(Paragraph("Dataset Analysis Report", ParagraphStyle('Sub', fontSize=16, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER, spaceAfter=4)))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="60%", thickness=1, color=ACCENT, hAlign='CENTER'))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"File: <b>{filename}.csv</b>", ParagraphStyle('CoverInfo', fontSize=11, textColor=TEXT, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", ParagraphStyle('CoverDate', fontSize=10, textColor=MUTED, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*inch))
    
    kpi_data = [
        ['Rows', 'Columns', 'Numerical', 'Categorical', 'Null Values'],
        [str(df.shape[0]), str(df.shape[1]), str(len(num_cols)), str(len(cat_cols)), str(int(df.isnull().sum().sum()))]
    ]
    kpi_table = Table(kpi_data, colWidths=[2.8*cm]*5)
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#21262d')),
        ('TEXTCOLOR', (0,0), (-1,0), MUTED),
        ('TEXTCOLOR', (0,1), (-1,1), ACCENT),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('FONTSIZE', (0,1), (-1,1), 16),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [CARD, colors.HexColor('#21262d')]),
        ('BOX', (0,0), (-1,-1), 0.5, ACCENT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Contents: Overview  •  Data Quality  •  Statistical Analysis  •  Visualizations  •  Insights", muted_style))
    story.append(PageBreak())

    # ---- PAGE 2: Dataset Overview ----
    story.append(Paragraph("1. Dataset Overview", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph(f"Shape: <b>{df.shape[0]} rows × {df.shape[1]} columns</b>  |  Memory: <b>{df.memory_usage(deep=True).sum()/1024:.1f} KB</b>", body))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Column Information", h3))
    
    col_data = [['Column', 'Type', 'Dtype', 'Unique', 'Nulls', 'Null %']]
    for col in df.columns:
        null_c = int(df[col].isnull().sum())
        null_p = f"{null_c/len(df)*100:.1f}%"
        ctype = 'Numerical' if col in num_cols else 'Categorical'
        col_data.append([col[:22], ctype, str(df[col].dtype), str(df[col].nunique()), str(null_c), null_p])
    
    col_table = Table(col_data, colWidths=[4.5*cm, 2.5*cm, 2.2*cm, 1.8*cm, 1.5*cm, 1.8*cm])
    col_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD),
        ('TEXTCOLOR', (0,0), (-1,0), ACCENT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(col_table)
    story.append(Spacer(1, 0.2*inch))

    # Sample rows
    story.append(Paragraph("Sample Data (First 5 Rows)", h3))
    sample = df.head(5)
    cols_show = df.columns[:7].tolist()
    sdata = [[c[:14] for c in cols_show]]
    for _, row in sample[cols_show].iterrows():
        sdata.append([str(v)[:14] if pd.notna(v) else 'NaN' for v in row])
    sw = [14.4*cm / len(cols_show)] * len(cols_show)
    stable = Table(sdata, colWidths=sw)
    stable.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD),
        ('TEXTCOLOR', (0,0), (-1,0), ACCENT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.5),
        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(stable)
    story.append(PageBreak())

    # ---- PAGE 3: Data Quality ----
    story.append(Paragraph("2. Data Quality Analysis", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.1*inch))

    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    completeness = round((1 - total_nulls/total_cells)*100, 2) if total_cells > 0 else 100
    dup_rows = int(df.duplicated().sum())

    q_data = [['Metric', 'Value', 'Status']]
    q_data.append(['Total Cells', str(total_cells), '✓'])
    q_data.append(['Missing Values', str(total_nulls), '✓' if total_nulls == 0 else '⚠'])
    q_data.append(['Completeness', f'{completeness}%', '✓' if completeness >= 95 else '⚠'])
    q_data.append(['Duplicate Rows', str(dup_rows), '✓' if dup_rows == 0 else '⚠'])
    q_data.append(['Numerical Columns', str(len(num_cols)), '✓'])
    q_data.append(['Categorical Columns', str(len(cat_cols)), '✓'])

    qt = Table(q_data, colWidths=[6*cm, 5*cm, 3.4*cm])
    qt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CARD),
        ('TEXTCOLOR', (0,0), (-1,0), ACCENT),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
        ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 7),
    ]))
    story.append(qt)
    story.append(Spacer(1, 0.2*inch))

    # Missing values chart
    null_series = df.isnull().sum()
    null_series = null_series[null_series > 0]
    if len(null_series) > 0:
        story.append(Paragraph("Missing Values by Column", h3))
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        bars = ax.barh(null_series.index.tolist(), null_series.values, color='#f85149', alpha=0.85)
        ax.set_xlabel('Null Count', color='#8b949e')
        ax.tick_params(colors='#e6edf3', labelsize=8)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#30363d')
        for bar, val in zip(bars, null_series.values):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, str(val), va='center', color='#e6edf3', fontsize=8)
        plt.tight_layout()
        img_bytes = _mpl_fig_to_bytes(fig)
        plt.close(fig)
        story.append(RLImage(io.BytesIO(img_bytes), width=14*cm, height=5.5*cm))
    story.append(PageBreak())

    # ---- PAGE 4: Statistical Analysis ----
    story.append(Paragraph("3. Statistical Analysis", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.1*inch))

    if num_cols:
        desc = df[num_cols].describe().round(4)
        story.append(Paragraph("Descriptive Statistics – Numerical Columns", h3))
        stat_rows = [['Stat'] + [c[:12] for c in num_cols[:7]]]
        for idx in desc.index:
            row = [idx] + [str(desc.loc[idx, c]) if c in desc.columns else '-' for c in num_cols[:7]]
            stat_rows.append(row)
        cw = [2.5*cm] + [12*cm/min(len(num_cols), 7)] * min(len(num_cols), 7)
        dt = Table(stat_rows, colWidths=cw)
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), CARD),
            ('TEXTCOLOR', (0,0), (-1,0), ACCENT),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 7.5),
            ('TEXTCOLOR', (0,1), (-1,-1), TEXT),
            ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0,1), (0,-1), MUTED),
            ('FONTNAME', (1,1), (-1,-1), 'Helvetica'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
            ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(dt)
        story.append(Spacer(1, 0.15*inch))

    # Correlation
    if len(num_cols) >= 2:
        story.append(Paragraph("Correlation Matrix", h3))
        corr = df[num_cols[:7]].corr().round(3)
        corr_rows = [[''] + [c[:10] for c in corr.columns]]
        for idx in corr.index:
            corr_rows.append([idx[:10]] + [str(corr.loc[idx, c]) for c in corr.columns])
        cw2 = [2.3*cm] * (len(corr.columns)+1)
        ct = Table(corr_rows, colWidths=cw2)
        ct.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), CARD),
            ('BACKGROUND', (0,0), (0,-1), CARD),
            ('TEXTCOLOR', (0,0), (-1,0), ACCENT),
            ('TEXTCOLOR', (0,0), (0,-1), ACCENT),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('TEXTCOLOR', (1,1), (-1,-1), TEXT),
            ('ROWBACKGROUNDS', (1,1), (-1,-1), [colors.HexColor('#161b22'), colors.HexColor('#0d1117')]),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#30363d')),
            ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(ct)
    story.append(PageBreak())

    # ---- PAGE 5: Visualizations & Insights ----
    story.append(Paragraph("4. Visual Insights", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.1*inch))

    # Distribution plots for up to 4 numerical cols
    if num_cols:
        cols_plot = num_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor='#0d1117')
        axes = axes.flatten()
        for i, col in enumerate(cols_plot):
            ax = axes[i]
            ax.set_facecolor('#161b22')
            data_clean = df[col].dropna()
            ax.hist(data_clean, bins=25, color='#58a6ff', alpha=0.8, edgecolor='#21262d')
            ax.axvline(data_clean.mean(), color='#f85149', linestyle='--', linewidth=1.2, label='Mean')
            ax.axvline(data_clean.median(), color='#3fb950', linestyle='--', linewidth=1.2, label='Median')
            ax.set_title(col, color='#e6edf3', fontsize=9, pad=6)
            ax.tick_params(colors='#8b949e', labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#30363d')
            ax.legend(fontsize=7, labelcolor='#e6edf3', facecolor='#0d1117', edgecolor='#30363d')
        for j in range(len(cols_plot), 4):
            axes[j].set_visible(False)
        plt.tight_layout(pad=1.5)
        img_bytes = _mpl_fig_to_bytes(fig)
        plt.close(fig)
        story.append(Paragraph("Distribution Plots", h3))
        story.append(RLImage(io.BytesIO(img_bytes), width=14*cm, height=9.5*cm))
        story.append(Spacer(1, 0.15*inch))

    # Insights
    story.append(Paragraph("5. Key Insights & Recommendations", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#30363d')))
    story.append(Spacer(1, 0.1*inch))
    insights = []
    insights.append(f"• Dataset has <b>{df.shape[0]:,} rows</b> and <b>{df.shape[1]} columns</b> with {completeness}% completeness.")
    if total_nulls > 0:
        insights.append(f"• <b>{total_nulls} missing values</b> detected across {(df.isnull().sum()>0).sum()} columns — consider imputation or removal.")
    else:
        insights.append("• <b>No missing values</b> found — dataset is complete.")
    if dup_rows > 0:
        insights.append(f"• <b>{dup_rows} duplicate rows</b> found. Consider deduplication before modeling.")
    if num_cols:
        skewed = [c for c in num_cols if abs(df[c].skew()) > 1]
        if skewed:
            insights.append(f"• Highly skewed columns: <b>{', '.join(skewed[:5])}</b>. Consider log transformation.")
    if len(num_cols) >= 2:
        corr_m = df[num_cols].corr().abs()
        np.fill_diagonal(corr_m.values, 0)
        if corr_m.max().max() > 0.85:
            insights.append("• <b>High correlation detected</b> between some numerical features — check for multicollinearity.")
    for ins in insights:
        story.append(Paragraph(ins, body))
        story.append(Spacer(1, 0.06*inch))

    doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
