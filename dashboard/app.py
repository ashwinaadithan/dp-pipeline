"""
📊 Vignesh TAT - Dynamic Pricing Dashboard
==========================================
All data pulled dynamically from Neon PostgreSQL DB
No hardcoded data - everything from hourly scraper pipeline
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
import sys
from io import BytesIO

# ==============================================================================
# DATABASE CONNECTION - ALL DATA FROM NEON DB
# ==============================================================================

src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db_connection():
    try:
        db_url = st.secrets.get('NEON_DATABASE_URL') if hasattr(st, 'secrets') else os.getenv('NEON_DATABASE_URL')
        if not db_url:
            return None, "DB not configured"
        import psycopg
        return psycopg.connect(db_url), None
    except Exception as e:
        return None, str(e)


def fetch_data(limit=5000):
    """Fetch ALL data from Neon DB - no hardcoded values."""
    conn, error = get_db_connection()
    if not conn:
        return None, error
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
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else None, None
    except Exception as e:
        return None, str(e)


# ==============================================================================
# PAGE CONFIG - REMOVE TOP WHITESPACE
# ==============================================================================

st.set_page_config(page_title="Vignesh TAT Pricing", page_icon="🚌", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: #0f1219;
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove top whitespace */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Header styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    
    .operator-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .record-count {
        background: #1e293b;
        color: #94a3b8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
    }
    
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Section headers */
    .section-title {
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1rem 0 0.6rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1e293b;
    }
    
    hr { border: none; border-top: 1px solid #1e293b; margin: 0.8rem 0; }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA PROCESSING
# ==============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load from Neon DB only - no fallback demo data."""
    df, err = fetch_data(5000)
    if df is not None and not df.empty:
        return df, None
    return pd.DataFrame(), err  # Return empty if no DB connection


def preprocess(df):
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
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Pricing_Data')
    return out.getvalue()


# ==============================================================================
# CHARTS - All using live DB data
# ==============================================================================

COLORS = {'blue': '#3b82f6', 'pink': '#ec4899', 'green': '#22c55e', 'yellow': '#eab308', 'red': '#ef4444', 'purple': '#a855f7'}

def layout(h=340, title=""):
    return {
        "template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter", "color": "#e2e8f0", "size": 11},
        "margin": {"l": 50, "r": 20, "t": 45, "b": 40}, "height": h,
        "title": {"text": title, "font": {"size": 13, "color": "#f1f5f9"}, "x": 0.01, "y": 0.97}
    }


def chart_demand_curve(df):
    """Shows how price and occupancy change as departure date approaches."""
    if 'days_to_departure' not in df.columns or df.empty:
        return None
    agg = df.groupby('days_to_departure').agg({'base_price': 'mean', 'occupancy': 'mean'}).reset_index()
    agg = agg.sort_values('days_to_departure', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['base_price'], name='Price ₹',
        mode='lines+markers', line=dict(color=COLORS['blue'], width=2.5, shape='spline'),
        marker=dict(size=7), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['occupancy'], name='Occupancy %',
        mode='lines+markers', line=dict(color=COLORS['pink'], width=2.5, shape='spline'),
        marker=dict(size=7)
    ), secondary_y=True)
    
    fig.update_layout(**layout(340, "Price & Occupancy vs Days to Travel"))
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=10)))
    fig.update_xaxes(title="Days Until Departure", autorange="reversed", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Price ₹", secondary_y=False, gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Occupancy %", secondary_y=True, gridcolor="#1e293b", zeroline=False)
    return fig


def chart_scatter(df):
    """Price elasticity - relationship between price and demand."""
    if df.empty:
        return None
    top_routes = df['route'].value_counts().head(5).index.tolist()
    df_plot = df[df['route'].isin(top_routes)]
    
    fig = px.scatter(df_plot, x='base_price', y='occupancy', color='route',
                     size='total_seats', size_max=16, opacity=0.8,
                     color_discrete_sequence=[COLORS['green'], COLORS['yellow'], COLORS['pink'], COLORS['blue'], COLORS['purple']])
    
    fig.update_layout(**layout(340, "Price vs Occupancy"))
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02, font=dict(size=9)),
                      margin={"l": 50, "r": 130, "t": 45, "b": 45})
    fig.update_xaxes(title="Price ₹", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Occupancy %", gridcolor="#1e293b", zeroline=False)
    return fig


def chart_heatmap(df):
    """Yield matrix - price patterns across routes and days."""
    if df.empty:
        return None
    pivot = df.pivot_table(values='base_price', index='route', columns='day_name', aggfunc='mean')
    day_map = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 
               'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}
    pivot = pivot.rename(columns=day_map)
    order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot = pivot[[d for d in order if d in pivot.columns]]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, '#0f172a'], [0.5, '#3b82f6'], [1, '#ec4899']],
        text=np.round(pivot.values, 0), texttemplate='₹%{text:.0f}',
        textfont={"size": 9, "color": "white"}, hoverongaps=False
    ))
    fig.update_layout(**layout(280, "Yield Matrix"))
    return fig


def chart_weekend(df):
    """Weekend premium analysis."""
    if df.empty or not df['is_weekend'].any() or df['is_weekend'].all():
        return None
    
    wd, we = df[~df['is_weekend']]['base_price'].mean(), df[df['is_weekend']]['base_price'].mean()
    prem = ((we - wd) / wd * 100) if wd > 0 else 0
    
    fig = go.Figure(go.Bar(x=['Weekday', 'Weekend'], y=[wd, we], marker_color=[COLORS['blue'], COLORS['pink']],
                           text=[f'₹{wd:.0f}', f'₹{we:.0f}'], textposition='outside', textfont=dict(color='#e2e8f0', size=11)))
    fig.update_layout(**layout(260, f"Weekend: {prem:+.1f}%"))
    fig.update_layout(showlegend=False, bargap=0.5)
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    return fig


def chart_gauge(occ):
    """Fleet occupancy gauge."""
    color = COLORS['green'] if occ >= 70 else COLORS['yellow'] if occ >= 50 else COLORS['red']
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=occ, number={'suffix': '%', 'font': {'size': 28, 'color': '#f1f5f9'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color, 'thickness': 0.7}, 'bgcolor': '#1e293b', 'borderwidth': 0}
    ))
    fig.update_layout(height=180, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=25, b=5, l=20, r=20))
    return fig


def chart_route_bar(df):
    """Route occupancy ranking."""
    if df.empty:
        return None
    stats = df.groupby('route')['occupancy'].mean().sort_values().reset_index()
    colors = [COLORS['red'] if v < 50 else COLORS['yellow'] if v < 70 else COLORS['green'] for v in stats['occupancy']]
    
    fig = go.Figure(go.Bar(y=stats['route'], x=stats['occupancy'], orientation='h', marker_color=colors,
                           text=stats['occupancy'].round(1).astype(str) + '%', textposition='inside', textfont=dict(color='white', size=9)))
    fig.update_layout(**layout(280, "Route Occupancy"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="#1e293b", zeroline=False)
    return fig


def chart_bus_box(df):
    """Price distribution by coach type."""
    if df.empty:
        return None
    fig = px.box(df, x='bus_type', y='base_price', color='bus_type',
                 color_discrete_sequence=[COLORS['blue'], COLORS['pink'], COLORS['purple'], COLORS['green']])
    fig.update_layout(**layout(280, "Price by Coach"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    return fig


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Load data from Neon DB
    df, err = load_data()
    df = preprocess(df)
    
    # Header with operator name
    if df.empty:
        st.error(f"❌ No data from Neon DB: {err}")
        return
    
    st.markdown(f"""
        <div class="main-header">
            <h1 class="main-title">🚌 Pricing Dashboard</h1>
            <span class="operator-badge">Vignesh TAT</span>
            <span class="record-count">{len(df):,} records</span>
        </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Price", f"₹{df['base_price'].mean():,.0f}")
    k2.metric("Occupancy", f"{df['occupancy'].mean():.1f}%")
    k3.metric("Seats Sold", f"{df['sold_seats'].sum():,}")
    k4.metric("Routes", f"{df['route'].nunique()}")
    if df['is_weekend'].any() and (~df['is_weekend']).any():
        prem = ((df[df['is_weekend']]['base_price'].mean() / df[~df['is_weekend']]['base_price'].mean()) - 1) * 100
        k5.metric("Weekend Δ", f"{prem:+.1f}%")
    else:
        k5.metric("Weekend Δ", "—")
    
    st.markdown("---")
    
    # Row 1
    st.markdown("<div class='section-title'>Demand Analytics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_demand_curve(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = chart_scatter(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2
    st.markdown("<div class='section-title'>Yield Management</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
    with c1:
        fig = chart_heatmap(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = chart_weekend(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.caption("Fleet Occupancy")
        st.plotly_chart(chart_gauge(df['occupancy'].mean()), use_container_width=True)
    
    st.markdown("---")
    
    # Row 3
    st.markdown("<div class='section-title'>Route Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_route_bar(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = chart_bus_box(df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data Export
    st.markdown("<div class='section-title'>Data Export</div>", unsafe_allow_html=True)
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        routes_opt = ['All'] + sorted(df['route'].unique().tolist())
        sel_routes = st.multiselect("Routes", routes_opt, default=['All'], label_visibility="collapsed", placeholder="Routes...")
    with f2:
        types_opt = ['All'] + sorted(df['bus_type'].unique().tolist())
        sel_types = st.multiselect("Types", types_opt, default=['All'], label_visibility="collapsed", placeholder="Coach...")
    with f3:
        price_rng = st.slider("Price", int(df['base_price'].min()), int(df['base_price'].max()), 
                              (int(df['base_price'].min()), int(df['base_price'].max())), label_visibility="collapsed")
    with f4:
        occ_rng = st.slider("Occ", 0, 100, (0, 100), label_visibility="collapsed")
    
    mask = pd.Series([True] * len(df))
    if 'All' not in sel_routes and sel_routes: mask &= df['route'].isin(sel_routes)
    if 'All' not in sel_types and sel_types: mask &= df['bus_type'].isin(sel_types)
    mask &= (df['base_price'] >= price_rng[0]) & (df['base_price'] <= price_rng[1])
    mask &= (df['occupancy'] >= occ_rng[0]) & (df['occupancy'] <= occ_rng[1])
    fdf = df[mask]
    
    cols = ['travel_date', 'route', 'bus_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy']
    cols = [c for c in cols if c in fdf.columns]
    
    st.caption(f"{len(fdf):,} / {len(df):,} records")
    st.dataframe(fdf[cols].sort_values('travel_date', ascending=False), use_container_width=True, height=320,
                 column_config={"travel_date": st.column_config.DateColumn("Date", format="DD-MMM"),
                                "base_price": st.column_config.NumberColumn("Price", format="₹%d"),
                                "occupancy": st.column_config.ProgressColumn("Occ%", min_value=0, max_value=100)})
    
    e1, e2, _ = st.columns([1, 1, 4])
    with e1:
        st.download_button("⬇️ CSV", fdf[cols].to_csv(index=False).encode(), f"vignesh_tat_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
    with e2:
        st.download_button("⬇️ Excel", to_excel(fdf[cols]), f"vignesh_tat_{datetime.now():%Y%m%d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


if __name__ == "__main__":
    main()
