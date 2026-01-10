"""
🎯 Dynamic Pricing Intelligence Dashboard
==========================================
Sciative + RevMax Analytics for Bus Yield Management
Pipeline: Oracle Cron → Hourly Scrape → Neon DB → Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# ==============================================================================
# 🔌 SAFE DATABASE CONNECTION
# ==============================================================================

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db_connection():
    """Safely get database connection with proper error handling."""
    try:
        # Check for Streamlit secrets first
        db_url = None
        if hasattr(st, 'secrets') and 'NEON_DATABASE_URL' in st.secrets:
            db_url = st.secrets['NEON_DATABASE_URL']
        
        if not db_url:
            db_url = os.getenv('NEON_DATABASE_URL')
        
        if not db_url:
            return None, "NEON_DATABASE_URL not found in secrets or environment"
        
        import psycopg
        conn = psycopg.connect(db_url)
        return conn, None
        
    except ImportError:
        return None, "psycopg not installed"
    except Exception as e:
        return None, f"Connection failed: {str(e)}"


def fetch_latest_data(limit=5000):
    """Fetch data directly without going through database.py module."""
    conn, error = get_db_connection()
    
    if conn is None:
        return None, error
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                bus_id, operator, bus_type,
                from_city, to_city, travel_date,
                departure_time, base_price,
                available_seats, sold_seats,
                min_price, max_price, scraped_at
            FROM buses
            ORDER BY scraped_at DESC
            LIMIT %s
        """, (limit,))
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        
        if rows:
            return [dict(zip(columns, row)) for row in rows], None
        else:
            return None, "No data in database"
            
    except Exception as e:
        return None, f"Query failed: {str(e)}"


# ==============================================================================
# 🎨 PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional dark theme
st.markdown("""
<style>
    /* Dark professional background */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    
    /* Clean metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1c2333 0%, #252d3d 100%);
        border: 1px solid #30363d;
        padding: 1rem;
        border-radius: 12px;
    }
    
    div[data-testid="metric-container"] label {
        color: #8b949e !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
    }
    
    /* Section headers */
    .section-title {
        color: #58a6ff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #21262d;
    }
    
    /* Status badges */
    .badge-live {
        background: #238636;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-demo {
        background: #9e6a03;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Clean dividers */
    hr {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 📊 DATA LOADING
# ==============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load from database or generate demo data."""
    data, error = fetch_latest_data(limit=5000)
    
    if data:
        return pd.DataFrame(data), None
    else:
        # Generate realistic demo data
        return generate_demo_data(), error


def generate_demo_data():
    """Demo data for preview."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    records = []
    
    routes = [
        ("Chennai", "Tirunelveli", 1200),
        ("Tirunelveli", "Chennai", 1200),
        ("Chennai", "Madurai", 900),
        ("Madurai", "Chennai", 900),
        ("Chennai", "Coimbatore", 800),
        ("Coimbatore", "Chennai", 800)
    ]
    
    for d in dates:
        is_weekend = d.dayofweek >= 5
        demand_mult = 1.25 if is_weekend else 1.0
        
        for from_city, to_city, base in routes:
            price = int(base * demand_mult * np.random.uniform(0.9, 1.15))
            avail = np.random.randint(8, 28)
            sold = np.random.randint(12, 35)
            
            records.append({
                "bus_id": f"VT_{len(records)}",
                "travel_date": d,
                "from_city": from_city,
                "to_city": to_city,
                "base_price": price,
                "min_price": int(price * 0.9),
                "max_price": int(price * 1.1),
                "available_seats": avail,
                "sold_seats": sold,
                "bus_type": np.random.choice(["Volvo Sleeper", "Multi-Axle", "AC Sleeper"]),
                "operator": "Vignesh TAT",
                "departure_time": np.random.choice(["20:00", "21:30", "22:00"]),
                "scraped_at": datetime.now()
            })
    
    return pd.DataFrame(records)


def preprocess(df):
    """Add computed columns."""
    if df.empty:
        return df
    
    df = df.copy()
    df['route'] = df['from_city'] + ' → ' + df['to_city']
    df['total_seats'] = df['sold_seats'] + df['available_seats']
    df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100).round(1)
    df['travel_date'] = pd.to_datetime(df['travel_date'])
    df['day_name'] = df['travel_date'].dt.day_name()
    df['day_of_week'] = df['travel_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['days_to_departure'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    else:
        df['days_to_departure'] = 7
    
    return df


# ==============================================================================
# 📈 CHART FUNCTIONS
# ==============================================================================

def chart_config():
    """Common chart configuration."""
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#c9d1d9", "size": 11},
        "margin": {"l": 40, "r": 40, "t": 50, "b": 40}
    }


def price_demand_curve(df):
    """Price vs Days-to-Departure with Occupancy."""
    if 'days_to_departure' not in df.columns or df.empty:
        return None
    
    agg = df.groupby('days_to_departure').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index().sort_values('days_to_departure', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['base_price'],
        mode='lines+markers', name='Price ₹',
        line=dict(color='#58a6ff', width=2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.1)'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['occupancy'],
        mode='lines+markers', name='Occupancy %',
        line=dict(color='#f78166', width=2)
    ), secondary_y=True)
    
    fig.update_layout(**chart_config(), height=320, title="📈 Price & Demand vs Days Before Travel")
    fig.update_xaxes(title="Days Until Departure", autorange="reversed")
    fig.update_yaxes(title="Avg Price ₹", secondary_y=False)
    fig.update_yaxes(title="Occupancy %", secondary_y=True)
    
    return fig


def price_elasticity_scatter(df):
    """Revenue opportunity analysis."""
    fig = px.scatter(
        df, x='base_price', y='occupancy', color='route',
        size='total_seats', hover_data=['travel_date', 'bus_type'],
        title="💹 Price vs Occupancy by Route"
    )
    fig.update_layout(**chart_config(), height=320, showlegend=True, legend=dict(orientation="h", y=-0.15))
    return fig


def yield_heatmap(df):
    """Route × Day pricing matrix."""
    pivot = df.pivot_table(values='base_price', index='route', columns='day_name', aggfunc='mean')
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    rename_map = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 
                  'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}
    pivot = pivot.rename(columns=rename_map)
    pivot = pivot[[d for d in day_order if d in pivot.columns]]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale='RdYlGn', text=np.round(pivot.values, 0),
        texttemplate='₹%{text:.0f}', textfont={"size": 9},
        hoverongaps=False
    ))
    fig.update_layout(**chart_config(), height=280, title="🗓️ Yield Matrix: Route × Day")
    return fig


def weekend_bar(df):
    """Weekend premium comparison."""
    if not df['is_weekend'].any() or df['is_weekend'].all():
        return None
    
    wkday = df[~df['is_weekend']]['base_price'].mean()
    wkend = df[df['is_weekend']]['base_price'].mean()
    premium = ((wkend - wkday) / wkday * 100) if wkday > 0 else 0
    
    fig = go.Figure(go.Bar(
        x=['Weekday', 'Weekend'], y=[wkday, wkend],
        marker_color=['#58a6ff', '#f78166'],
        text=[f'₹{wkday:.0f}', f'₹{wkend:.0f}'], textposition='outside'
    ))
    fig.update_layout(**chart_config(), height=250, title=f"📊 Weekend Premium: +{premium:.1f}%", showlegend=False)
    return fig


def occupancy_dial(occ):
    """Gauge for fleet occupancy."""
    color = '#238636' if occ >= 70 else '#9e6a03' if occ >= 50 else '#da3633'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=occ,
        number={'suffix': '%', 'font': {'size': 28, 'color': '#f0f6fc'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#484f58'},
            'bar': {'color': color},
            'bgcolor': '#21262d',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(218,54,51,0.2)'},
                {'range': [50, 70], 'color': 'rgba(158,106,3,0.2)'},
                {'range': [70, 100], 'color': 'rgba(35,134,54,0.2)'}
            ]
        }
    ))
    fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10))
    return fig


def route_ranking(df):
    """Bar chart of route occupancy."""
    route_stats = df.groupby('route')['occupancy'].mean().sort_values().reset_index()
    
    fig = go.Figure(go.Bar(
        y=route_stats['route'], x=route_stats['occupancy'],
        orientation='h', marker_color='#58a6ff',
        text=route_stats['occupancy'].round(1).astype(str) + '%', textposition='inside'
    ))
    fig.update_layout(**chart_config(), height=280, title="🛤️ Route Occupancy Ranking", showlegend=False)
    return fig


def bus_type_box(df):
    """Price distribution by bus type."""
    fig = px.box(df, x='bus_type', y='base_price', color='bus_type', title="🚌 Price by Coach Type")
    fig.update_layout(**chart_config(), height=280, showlegend=False)
    return fig


# ==============================================================================
# 🖥️ MAIN APP
# ==============================================================================

def main():
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 🎯 Dynamic Pricing Intelligence")
        st.caption("Sciative + RevMax Analytics • Vignesh TAT")
    
    # Load data
    df, error = load_data()
    
    with col2:
        if error:
            st.markdown(f"<span class='badge-demo'>⚠ DEMO</span>", unsafe_allow_html=True)
            with st.expander("Details"):
                st.code(error, language=None)
        else:
            st.markdown("<span class='badge-live'>● LIVE</span>", unsafe_allow_html=True)
    
    df = preprocess(df)
    
    st.divider()
    
    # ===== KPIs =====
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Records", f"{len(df):,}")
    k2.metric("Avg Price", f"₹{df['base_price'].mean():,.0f}")
    k3.metric("Avg Occupancy", f"{df['occupancy'].mean():.1f}%")
    k4.metric("Seats Sold", f"{df['sold_seats'].sum():,}")
    
    if df['is_weekend'].any() and (~df['is_weekend']).any():
        wknd_prem = ((df[df['is_weekend']]['base_price'].mean() / df[~df['is_weekend']]['base_price'].mean()) - 1) * 100
        k5.metric("Weekend Lift", f"+{wknd_prem:.1f}%")
    else:
        k5.metric("Weekend Lift", "—")
    
    st.divider()
    
    # ===== Row 1: Demand Curves =====
    st.markdown("<div class='section-title'>📈 Demand & Price Analytics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        fig = price_demand_curve(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.plotly_chart(price_elasticity_scatter(df), use_container_width=True)
    
    st.divider()
    
    # ===== Row 2: Yield Management =====
    st.markdown("<div class='section-title'>🎯 Yield Management</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([3, 2, 2])
    
    with c1:
        st.plotly_chart(yield_heatmap(df), use_container_width=True)
    
    with c2:
        fig = weekend_bar(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with c3:
        st.markdown("**Fleet Occupancy**")
        st.plotly_chart(occupancy_dial(df['occupancy'].mean()), use_container_width=True)
    
    st.divider()
    
    # ===== Row 3: Route & Type Analysis =====
    st.markdown("<div class='section-title'>🛤️ Route & Fleet Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(route_ranking(df), use_container_width=True)
    
    with c2:
        st.plotly_chart(bus_type_box(df), use_container_width=True)
    
    st.divider()
    
    # ===== Data Export =====
    st.markdown("<div class='section-title'>📥 ML Training Data</div>", unsafe_allow_html=True)
    
    with st.expander("Download Data for Dynamic Pricing Model", expanded=False):
        cols = ['travel_date', 'day_name', 'day_of_week', 'is_weekend', 'days_to_departure',
                'route', 'bus_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy']
        export_cols = [c for c in cols if c in df.columns]
        
        st.dataframe(df[export_cols].head(100), use_container_width=True, height=300)
        
        csv = df[export_cols].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", data=csv, 
                           file_name=f"pricing_data_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")
    
    # Footer
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')} • Pipeline: Oracle Cron → Neon DB → Dashboard")


if __name__ == "__main__":
    main()
