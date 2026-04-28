# DataZen — CSV Analytics Platform

A full-stack web app for CSV dataset analysis with:
- Auto null detection & imputation (mean/median/mode)
- Interactive Plotly dashboard with 8+ chart types
- 5-page PDF report with visualizations
- Cleaned CSV download

## Tech Stack
- **Backend**: Flask (Python), Pandas, NumPy, Plotly Express, Matplotlib, ReportLab
- **Frontend**: HTML5, CSS3 (custom dark theme), Vanilla JS, Plotly.js

---

## 🚀 LOCAL DEVELOPMENT

### 1. Install dependencies
```bash
cd datazen
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open browser
```
http://localhost:5000
```

---

## 🌐 DEPLOY ON RENDER (RECOMMENDED — Free tier)

Netlify is for static sites. Since this is a Python Flask backend, **Render.com** is better.

### Steps:
1. Push code to GitHub
   ```bash
   git init
   git add .
   git commit -m "initial"
   git remote add origin https://github.com/YOUR_USER/datazen.git
   git push -u origin main
   ```

2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click **Deploy**
6. Your app will be live at `https://datazen-xxxx.onrender.com`

---

## 🌐 DEPLOY ON RAILWAY (Also great)

1. Install Railway CLI: `npm i -g @railway/cli`
2. `railway login`
3. `cd datazen && railway init`
4. `railway up`
5. `railway open`

---

## 🌐 DEPLOY ON NETLIFY (Static Frontend + Separate API)

Netlify only hosts static files. For the Python backend, you need a separate service.

### Option A: Netlify Frontend + Render Backend
1. Deploy backend to Render (steps above)
2. In `static/js/app.js`, change line 1:
   ```js
   const API = 'https://your-render-url.onrender.com';
   ```
3. Deploy frontend to Netlify:
   ```bash
   netlify deploy --prod --dir=templates
   ```

### Option B: Convert to Netlify Functions (Python)
Add `netlify/functions/` folder with serverless handlers.
Note: file size limits apply (50MB max, 10s timeout).

---

## 📁 PROJECT STRUCTURE

```
datazen/
├── app.py              # Flask backend + all API routes
├── requirements.txt    # Python dependencies
├── Procfile            # For Gunicorn/Heroku-style deploy
├── runtime.txt         # Python version
├── netlify.toml        # Netlify config
├── templates/
│   └── index.html      # Single-page app
├── static/
│   ├── css/style.css   # Dark industrial UI theme
│   └── js/app.js       # Frontend logic
├── uploads/            # Temp CSV storage (auto-created)
└── reports/            # Generated PDFs (auto-created)
```

---

## 🔧 ENVIRONMENT VARIABLES (for production)

```env
SECRET_KEY=your-secret-key
MAX_CONTENT_LENGTH=52428800
```

---

## 📊 FEATURES

| Feature | Details |
|---|---|
| Upload | CSV up to 50MB, drag & drop |
| Null Analysis | Per-column null count, %, type detection |
| Imputation | Mean/Median (numerical), Mode (categorical), Drop rows |
| Dashboard | 8+ Plotly charts: histograms, box plots, correlation heatmap, scatter matrix, pie, bar, outlier detection |
| PDF Report | 5-page ReportLab PDF with cover, overview, quality, stats, insights |
| Download | Cleaned CSV + PDF report |
