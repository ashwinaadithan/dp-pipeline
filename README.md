# Dynamic Pricing Data Pipeline

Autonomous bus ticket scraper with visualization dashboard and cloud deployment.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Set up environment variables
cp .env.example .env
# Edit .env with your Neon DB credentials

# Run scraper locally
python src/scraper.py
```

## 📁 Project Structure

```
dp_pipeline/
├── src/
│   ├── scraper.py          # Main autonomous scraper
│   ├── database.py         # Neon DB integration
│   └── export.py           # Multi-format export (JSON/CSV/Excel)
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── charts.py           # Visualization components
├── data/                   # Scraped data storage
├── .github/workflows/
│   └── scrape.yml          # GitHub Actions (hourly scraping)
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Configuration

Edit `src/scraper.py` to configure:

```python
OPERATOR_URL = "https://www.redbus.in/travels/vignesh-tat"
SCRAPE_INTERVAL_HOURS = 1
TOTAL_DURATION_DAYS = 7
DAYS_AHEAD = 7
```

## 🌐 Free Cloud Deployment

### GitHub Actions (Scraping)
- Runs every hour automatically
- FREE: 2000 minutes/month
- Data saved to Neon DB

### Streamlit Cloud (Dashboard)
- Live visualization
- FREE tier available
- Auto-updates from DB

### Neon DB (Storage)
- PostgreSQL database
- FREE: 0.5GB storage
- Always-on serverless

## 📊 Dashboard Features

- **Price Trends**: Track price changes over time
- **Occupancy Analysis**: See booking patterns
- **Route Comparison**: Compare routes side-by-side
- **ML-Ready Data**: Export for model training
