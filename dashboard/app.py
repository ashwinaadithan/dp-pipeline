"""
🎯 Dynamic Pricing Intelligence Dashboard
==========================================
Premium Design | Vignesh TAT
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
from io import BytesIO

# ==============================================================================
# DATABASE
# ==============================================================================

src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db():
    try:
        db_url = st.secrets.get('NEON_DATABASE_URL') if hasattr(st, 'secrets') else os.getenv('NEON_DATABASE_URL')
        if not db_url:
            return None, "DB not configured"
        import psycopg
        return psycopg.connect(db_url), None
    except Exception as e:
        return None, str(e)


def fetch_buses(limit=2000):
    conn, err = get_db()
    if not conn:
        return pd.DataFrame(), err
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bus_id, operator, bus_type, from_city, to_city, travel_date,
                   departure_time, base_price, available_seats, sold_seats, scraped_at
            FROM buses ORDER BY scraped_at DESC LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def fetch_price_history(bus_id=None, limit=500):
    conn, err = get_db()
    if not conn:
        return pd.DataFrame()
    try:
        cur = conn.cursor()
        if bus_id:
            cur.execute("""
                SELECT bus_id, base_price, available_seats, sold_seats, scraped_at
                FROM price_history WHERE bus_id = %s ORDER BY scraped_at ASC LIMIT %s
            """, (bus_id, limit))
        else:
            cur.execute("""
                SELECT bus_id, base_price, available_seats, sold_seats, scraped_at
                FROM price_history ORDER BY scraped_at DESC LIMIT %s
            """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


def fetch_seat_history(bus_id, limit=1000):
    conn, err = get_db()
    if not conn:
        return pd.DataFrame()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT seat_id, price, deck, is_window, scraped_at
            FROM seat_prices WHERE bus_id = %s ORDER BY scraped_at ASC LIMIT %s
        """, (bus_id, limit))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


def fetch_buses_with_changes():
    conn, err = get_db()
    if not conn:
        return pd.DataFrame()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                ph.bus_id,
                b.from_city, b.to_city, b.travel_date, b.departure_time, b.bus_type,
                MIN(ph.base_price) as min_price,
                MAX(ph.base_price) as max_price,
                COUNT(*) as scrape_count,
                MAX(ph.base_price) - MIN(ph.base_price) as price_change
            FROM price_history ph
            JOIN buses b ON ph.bus_id = b.bus_id
            GROUP BY ph.bus_id, b.from_city, b.to_city, b.travel_date, b.departure_time, b.bus_type
            ORDER BY COUNT(*) DESC, price_change DESC
            LIMIT 100
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


# ==============================================================================
# PREMIUM STYLING
# ==============================================================================

st.set_page_config(page_title="Vignesh TAT DP", page_icon="🚌", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0c1222 0%, #1a1f3a 50%, #0d1426 100%);
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding: 1rem 2rem !important; }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .header-left { display: flex; align-items: center; gap: 15px; }
    .logo { font-size: 2.5rem; }
    .brand-text h1 { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: #f1f5f9; 
        margin: 0;
        background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-text p { color: #64748b; font-size: 0.85rem; margin: 0; }
    .live-badge {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 4px 15px rgba(34,197,94,0.3);
    }
    .live-dot { 
        width: 8px; height: 8px; 
        background: white; 
        border-radius: 50%; 
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.9) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(99,102,241,0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.15);
    }
    .card-label { color: #64748b; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    .card-value { color: #f1f5f9; font-size: 2rem; font-weight: 700; }
    .card-delta { font-size: 0.85rem; margin-top: 0.3rem; }
    .delta-up { color: #22c55e; }
    .delta-down { color: #ef4444; }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.5rem 0 1rem;
    }
    .section-header h2 {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    .section-line {
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.5) 0%, transparent 100%);
    }
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(135deg, rgba(30,41,59,0.6) 0%, rgba(15,23,42,0.8) 100%);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Price Tags */
    .price-up-tag {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px 0;
    }
    .price-down-tag {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 4px 0;
    }
    .price-stable-tag {
        background: rgba(100,116,139,0.3);
        color: #94a3b8;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 4px 0;
    }
    
    /* Festival Badge */
    .festival-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        box-shadow: 0 4px 15px rgba(245,158,11,0.3);
    }
    .weekend-badge {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(139,92,246,0.3);
    }
    
    /* Seat Card */
    .seat-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .seat-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .seat-id { color: #f1f5f9; font-weight: 600; font-size: 1rem; }
    .seat-type { color: #64748b; font-size: 0.75rem; }
    .seat-prices { display: flex; align-items: center; gap: 8px; }
    .seat-old-price { color: #64748b; text-decoration: line-through; font-size: 0.9rem; }
    .seat-new-price { color: #22c55e; font-weight: 700; font-size: 1.2rem; }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30,41,59,0.5);
        border-radius: 10px;
        padding: 10px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================

FESTIVALS = {
    (1, 13): "Bhogi", (1, 14): "Pongal", (1, 15): "Mattu Pongal", (1, 16): "Kaanum Pongal",
    (1, 26): "Republic Day", (8, 15): "Independence Day", (10, 12): "Dussehra", (11, 1): "Deepavali"
}

def get_day_type(date):
    if isinstance(date, str):
        date = pd.to_datetime(date)
    key = (date.month, date.day)
    if key in FESTIVALS:
        return f"🎉 {FESTIVALS[key]}"
    elif date.weekday() >= 5:
        return "📅 Weekend"
    return "Weekday"


def preprocess(df):
    if df.empty:
        return df
    df = df.copy()
    df['route'] = df['from_city'] + ' → ' + df['to_city']
    df['total_seats'] = df['sold_seats'] + df['available_seats']
    df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100).round(1)
    df['travel_date'] = pd.to_datetime(df['travel_date'])
    df['day_type'] = df['travel_date'].apply(get_day_type)
    df['is_festival'] = df['day_type'].str.contains('🎉')
    df['is_weekend'] = df['day_type'].str.contains('Weekend')
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['days_to_dep'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Data')
    return out.getvalue()


# ==============================================================================
# PREMIUM CHARTS
# ==============================================================================

def chart_price_timeline(history_df, bus_info):
    if history_df.empty:
        return None
    
    history_df = history_df.copy()
    history_df['scraped_at'] = pd.to_datetime(history_df['scraped_at'])
    history_df = history_df.sort_values('scraped_at')
    
    fig = go.Figure()
    
    # Gradient area fill
    fig.add_trace(go.Scatter(
        x=history_df['scraped_at'], y=history_df['base_price'],
        mode='lines', name='',
        line=dict(color='rgba(99,102,241,0)', width=0),
        fill='tozeroy', 
        fillgradient=dict(
            type='vertical',
            colorscale=[[0, 'rgba(99,102,241,0)'], [1, 'rgba(99,102,241,0.3)']]
        ),
        showlegend=False
    ))
    
    # Main line with glow effect
    fig.add_trace(go.Scatter(
        x=history_df['scraped_at'], y=history_df['base_price'],
        mode='lines+markers', name='Price',
        line=dict(color='#6366f1', width=3, shape='spline'),
        marker=dict(size=10, color='#6366f1', line=dict(width=2, color='white'))
    ))
    
    route = f"{bus_info.get('from_city', '')} → {bus_info.get('to_city', '')}"
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#e2e8f0"),
        height=320,
        title=dict(text=f"<b>{route}</b> | {bus_info.get('departure_time', '')}", font=dict(size=14, color='#f1f5f9')),
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=False,
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)', zerolinecolor='rgba(99,102,241,0.1)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)', zerolinecolor='rgba(99,102,241,0.1)', title='Price ₹')
    )
    
    return fig


def chart_seat_comparison(seat_df):
    if seat_df.empty:
        return None
    
    seat_df = seat_df.copy()
    seat_df['scraped_at'] = pd.to_datetime(seat_df['scraped_at'])
    seats = seat_df['seat_id'].unique()[:8]
    
    colors = ['#6366f1', '#ec4899', '#22c55e', '#f59e0b', '#06b6d4', '#8b5cf6', '#ef4444', '#14b8a6']
    
    fig = go.Figure()
    
    for i, seat in enumerate(seats):
        seat_data = seat_df[seat_df['seat_id'] == seat].sort_values('scraped_at')
        is_window = seat_data['is_window'].iloc[0] if not seat_data.empty else False
        
        fig.add_trace(go.Scatter(
            x=seat_data['scraped_at'], y=seat_data['price'],
            mode='lines+markers', name=f"{seat} {'🪟' if is_window else ''}",
            line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
            marker=dict(size=7, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#e2e8f0"),
        height=350,
        title=dict(text="<b>Seat Price Comparison</b>", font=dict(size=14, color='#f1f5f9')),
        margin=dict(l=50, r=20, t=50, b=60),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)', title='Price ₹')
    )
    
    return fig


def chart_day_type(df):
    if df.empty:
        return None
    
    groups = df.groupby('day_type')['base_price'].mean().reset_index()
    groups = groups.sort_values('base_price', ascending=True)
    
    colors = []
    for dt in groups['day_type']:
        if '🎉' in dt:
            colors.append('#f59e0b')
        elif 'Weekend' in dt:
            colors.append('#8b5cf6')
        else:
            colors.append('#6366f1')
    
    fig = go.Figure(go.Bar(
        x=groups['base_price'], y=groups['day_type'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'₹{p:,.0f}' for p in groups['base_price']],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=12, family='Inter')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#e2e8f0"),
        height=280,
        title=dict(text="<b>Price by Day Type</b>", font=dict(size=14, color='#f1f5f9')),
        margin=dict(l=120, r=60, t=50, b=30),
        showlegend=False,
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)', title='Avg Price ₹'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)')
    )
    
    return fig


def chart_timing(df):
    if df.empty or 'days_to_dep' not in df.columns:
        return None
    
    agg = df.groupby('days_to_dep').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    agg = agg[agg['days_to_dep'] >= 0].sort_values('days_to_dep')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price bars with gradient
    fig.add_trace(go.Bar(
        x=agg['days_to_dep'], y=agg['base_price'], name='Price',
        marker=dict(color='#6366f1', opacity=0.8)
    ), secondary_y=False)
    
    # Occupancy line
    fig.add_trace(go.Scatter(
        x=agg['days_to_dep'], y=agg['occupancy'], name='Occupancy',
        mode='lines+markers',
        line=dict(color='#ec4899', width=3, shape='spline'),
        marker=dict(size=8, color='#ec4899')
    ), secondary_y=True)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#e2e8f0"),
        height=280,
        title=dict(text="<b>Price & Occupancy by Days to Departure</b>", font=dict(size=14, color='#f1f5f9')),
        margin=dict(l=50, r=50, t=50, b=40),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(gridcolor='rgba(99,102,241,0.1)', title='Days Until Travel'),
        yaxis=dict(gridcolor='rgba(99,102,241,0.1)', title='Price ₹'),
        yaxis2=dict(gridcolor='rgba(99,102,241,0.1)', title='Occupancy %')
    )
    
    return fig


def chart_route_heatmap(df):
    if df.empty:
        return None
    
    pivot = df.pivot_table(values='base_price', index='route', columns='day_type', aggfunc='mean')
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=[[0, '#1e1b4b'], [0.5, '#6366f1'], [1, '#ec4899']],
        text=np.round(pivot.values, 0),
        texttemplate='₹%{text:.0f}',
        textfont=dict(size=10, color='white'),
        hoverongaps=False
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#e2e8f0"),
        height=350,
        title=dict(text="<b>Route × Day Pricing Matrix</b>", font=dict(size=14, color='#f1f5f9')),
        margin=dict(l=150, r=20, t=50, b=60),
        xaxis=dict(side='bottom'),
    )
    
    return fig


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    df, err = fetch_buses(2000)
    df = preprocess(df)
    
    if df.empty:
        st.error(f"No data: {err}")
        return
    
    # Premium Header
    st.markdown(f"""
    <div class="premium-header">
        <div class="header-left">
            <div class="logo">🚌</div>
            <div class="brand-text">
                <h1>Dynamic Pricing Intelligence</h1>
                <p>Vignesh TAT • Real-time Analytics</p>
            </div>
        </div>
        <div class="live-badge">
            <div class="live-dot"></div>
            LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Average Price</div>
            <div class="card-value">₹{df['base_price'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        occ = df['occupancy'].mean()
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Fleet Occupancy</div>
            <div class="card-value">{occ:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-label">Active Routes</div>
            <div class="card-value">{df['route'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        weekday_price = df[~df['is_weekend'] & ~df['is_festival']]['base_price'].mean()
        special_price = df[df['is_weekend'] | df['is_festival']]['base_price'].mean()
        if weekday_price > 0 and not pd.isna(special_price):
            premium = ((special_price / weekday_price) - 1) * 100
            delta_class = 'delta-up' if premium > 0 else 'delta-down'
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Weekend/Festival Premium</div>
                <div class="card-value">{premium:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-label">Weekend/Festival Premium</div>
                <div class="card-value">—</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price Tracking", "🪑 Seat Analysis", "📊 Insights"])
    
    # ===== TAB 1: PRICE TRACKING =====
    with tab1:
        st.markdown('<div class="section-header"><h2>🎯 Bus Price History</h2><div class="section-line"></div></div>', unsafe_allow_html=True)
        
        buses_tracked = fetch_buses_with_changes()
        
        if buses_tracked.empty:
            st.info("⏳ Collecting data... Price history will appear after 2+ hourly scrapes.")
        else:
            bus_options = []
            for _, row in buses_tracked.head(15).iterrows():
                route = f"{row['from_city']} → {row['to_city']}"
                change = row.get('price_change', 0)
                label = f"{route} | {row.get('departure_time', '')} | {change:+.0f} change" if change else f"{route} | {row.get('departure_time', '')}"
                bus_options.append((row['bus_id'], label, row))
            
            selected = st.selectbox("Select a bus to analyze:", [opt[1] for opt in bus_options], key="price_bus")
            selected_idx = [opt[1] for opt in bus_options].index(selected)
            selected_bus_id = bus_options[selected_idx][0]
            bus_info = bus_options[selected_idx][2]
            
            history = fetch_price_history(selected_bus_id)
            
            if not history.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = chart_price_timeline(history, bus_info)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("##### 💹 Price Changes")
                    history['scraped_at'] = pd.to_datetime(history['scraped_at'])
                    history = history.sort_values('scraped_at')
                    prices = history['base_price'].tolist()
                    times = history['scraped_at'].tolist()
                    
                    changes_found = False
                    for i in range(len(prices) - 1, 0, -1):
                        change = prices[i] - prices[i-1]
                        if change != 0:
                            changes_found = True
                            time_str = times[i].strftime('%I:%M %p')
                            if change > 0:
                                st.markdown(f'<div class="price-up-tag">⬆ ₹{change} at {time_str}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="price-down-tag">⬇ ₹{abs(change)} at {time_str}</div>', unsafe_allow_html=True)
                    
                    if not changes_found:
                        st.markdown('<div class="price-stable-tag">No price changes yet</div>', unsafe_allow_html=True)
                    
                    travel_date = bus_info.get('travel_date')
                    if travel_date:
                        day_type = get_day_type(travel_date)
                        st.markdown("<br>", unsafe_allow_html=True)
                        if '🎉' in day_type:
                            st.markdown(f'<div class="festival-badge">{day_type}</div>', unsafe_allow_html=True)
                        elif 'Weekend' in day_type:
                            st.markdown(f'<div class="weekend-badge">{day_type}</div>', unsafe_allow_html=True)
    
    # ===== TAB 2: SEAT ANALYSIS =====
    with tab2:
        st.markdown('<div class="section-header"><h2>🪟 Seat-Level Tracking</h2><div class="section-line"></div></div>', unsafe_allow_html=True)
        
        if buses_tracked.empty:
            st.info("⏳ Waiting for more data...")
        else:
            seat_bus = st.selectbox(
                "Select bus:", 
                [f"{row['from_city']} → {row['to_city']} | {row.get('departure_time', '')}" for _, row in buses_tracked.head(10).iterrows()],
                key="seat_bus"
            )
            idx = [f"{row['from_city']} → {row['to_city']} | {row.get('departure_time', '')}" for _, row in buses_tracked.head(10).iterrows()].index(seat_bus)
            bus_id = buses_tracked.iloc[idx]['bus_id']
            
            seat_history = fetch_seat_history(bus_id)
            
            if seat_history.empty:
                st.info("No seat-level data for this bus yet")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = chart_seat_comparison(seat_history)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("##### 🎫 Seat Summary")
                    seat_history['scraped_at'] = pd.to_datetime(seat_history['scraped_at'])
                    
                    for seat_id in seat_history['seat_id'].unique()[:6]:
                        seat_data = seat_history[seat_history['seat_id'] == seat_id].sort_values('scraped_at')
                        if len(seat_data) >= 1:
                            first_price = seat_data.iloc[0]['price']
                            last_price = seat_data.iloc[-1]['price']
                            change = last_price - first_price if len(seat_data) > 1 else 0
                            is_window = seat_data.iloc[0]['is_window']
                            
                            st.markdown(f"""
                            <div class="seat-card">
                                <div class="seat-card-header">
                                    <span class="seat-id">{seat_id} {'🪟' if is_window else '💺'}</span>
                                    <span class="seat-type">{'Window' if is_window else 'Aisle'}</span>
                                </div>
                                <div class="seat-prices">
                                    {'<span class="seat-old-price">₹' + str(first_price) + '</span>' if change != 0 else ''}
                                    <span class="seat-new-price">₹{last_price}</span>
                                    {f'<span class="{"delta-up" if change > 0 else "delta-down"}">({"+" if change > 0 else ""}{change})</span>' if change != 0 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # ===== TAB 3: INSIGHTS =====
    with tab3:
        st.markdown('<div class="section-header"><h2>📊 Pricing Insights</h2><div class="section-line"></div></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = chart_day_type(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = chart_timing(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header"><h2>🗺️ Route Pricing Matrix</h2><div class="section-line"></div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig = chart_route_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export
        st.markdown('<div class="section-header"><h2>📥 Export Data</h2><div class="section-line"></div></div>', unsafe_allow_html=True)
        
        cols = ['travel_date', 'route', 'day_type', 'bus_type', 'departure_time', 'base_price', 'occupancy']
        cols = [c for c in cols if c in df.columns]
        
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            st.download_button("⬇️ Download CSV", df[cols].to_csv(index=False).encode(), "vignesh_tat_data.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Download Excel", to_excel(df[cols]), "vignesh_tat_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


if __name__ == "__main__":
    main()
