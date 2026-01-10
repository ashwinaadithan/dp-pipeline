"""
🎯 Dynamic Pricing Intelligence Dashboard
==========================================
Sciative + RevMax Analytics for Bus Yield Management
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
# 🔌 DATABASE CONNECTION
# ==============================================================================

src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db_connection():
    try:
        db_url = st.secrets.get('NEON_DATABASE_URL') if hasattr(st, 'secrets') else None
        if not db_url:
            db_url = os.getenv('NEON_DATABASE_URL')
        if not db_url:
            return None, "DB URL not configured"
        
        import psycopg
        return psycopg.connect(db_url), None
    except Exception as e:
        return None, str(e)


def fetch_data(limit=5000):
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
        return [dict(zip(cols, r)) for r in rows] if rows else None, None
    except Exception as e:
        return None, str(e)


# ==============================================================================
# 🎨 PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(page_title="Pricing Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: #0f1219;
        font-family: 'Inter', sans-serif;
    }
    
    /* Minimal header */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0 1rem 0;
    }
    
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    
    .dashboard-count {
        background: #1e293b;
        color: #94a3b8;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    
    div[data-testid="metric-container"] label {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    
    /* Section headers */
    .section-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.25rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
    }
    
    /* Dividers */
    hr { border: none; border-top: 1px solid #1e293b; margin: 1rem 0; }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Filter labels */
    .stMultiSelect label, .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 📊 DATA FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=300)
def load_data():
    data, err = fetch_data(5000)
    if data:
        return pd.DataFrame(data), None
    return generate_demo(), err


def generate_demo():
    np.random.seed(42)
    dates = pd.date_range(datetime.now(), periods=14, freq='D')
    records = []
    routes = [("Chennai", "Tirunelveli"), ("Tirunelveli", "Chennai"), ("Chennai", "Madurai"), ("Madurai", "Chennai")]
    for d in dates:
        for f, t in routes:
            records.append({
                "bus_id": f"VT_{len(records)}", "travel_date": d, "from_city": f, "to_city": t,
                "base_price": int(1100 * (1.15 if d.dayofweek >= 5 else 1) * np.random.uniform(0.9, 1.1)),
                "available_seats": np.random.randint(8, 25), "sold_seats": np.random.randint(15, 35),
                "bus_type": np.random.choice(["Volvo Sleeper", "Multi-Axle"]),
                "operator": "Vignesh TAT", "scraped_at": datetime.now()
            })
    return pd.DataFrame(records)


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
    else:
        df['days_to_departure'] = 7
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Data')
    return out.getvalue()


# ==============================================================================
# 📈 CHARTS
# ==============================================================================

COLORS = {'blue': '#3b82f6', 'pink': '#ec4899', 'green': '#22c55e', 'yellow': '#eab308', 'red': '#ef4444', 'purple': '#a855f7', 'gray': '#64748b'}

def layout(h=340, title=""):
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter", "color": "#e2e8f0", "size": 11},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "height": h,
        "title": {"text": title, "font": {"size": 14, "color": "#f1f5f9"}, "x": 0.01, "y": 0.97}
    }


def chart_demand_curve(df):
    if 'days_to_departure' not in df.columns:
        return None
    agg = df.groupby('days_to_departure').agg({'base_price': 'mean', 'occupancy': 'mean'}).reset_index()
    agg = agg.sort_values('days_to_departure', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['base_price'], name='Price ₹',
        mode='lines+markers', line=dict(color=COLORS['blue'], width=2.5, shape='spline'),
        marker=dict(size=8, color=COLORS['blue']), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['occupancy'], name='Occupancy %',
        mode='lines+markers', line=dict(color=COLORS['pink'], width=2.5, shape='spline'),
        marker=dict(size=8, color=COLORS['pink'])
    ), secondary_y=True)
    
    fig.update_layout(**layout(360, "Price & Occupancy vs Days to Travel"))
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center", font=dict(size=11)))
    fig.update_xaxes(title="Days Until Departure", autorange="reversed", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Price ₹", secondary_y=False, gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Occupancy %", secondary_y=True, gridcolor="#1e293b", zeroline=False)
    return fig


def chart_scatter(df):
    # Limit routes for cleaner display
    top_routes = df['route'].value_counts().head(5).index.tolist()
    df_plot = df[df['route'].isin(top_routes)].copy()
    
    fig = px.scatter(
        df_plot, x='base_price', y='occupancy', color='route',
        size='total_seats', size_max=18, opacity=0.8,
        hover_data={'travel_date': True, 'bus_type': True},
        color_discrete_sequence=[COLORS['green'], COLORS['yellow'], COLORS['pink'], COLORS['blue'], COLORS['purple']]
    )
    
    fig.update_layout(**layout(360, "Price vs Occupancy"))
    # Fix legend position - move to right side to avoid overlap
    fig.update_layout(
        legend=dict(
            orientation="v", 
            yanchor="top", y=0.98, 
            xanchor="left", x=1.02,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin={"l": 50, "r": 140, "t": 50, "b": 50}  # More right margin for legend
    )
    fig.update_xaxes(title="Base Price ₹", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Occupancy %", gridcolor="#1e293b", zeroline=False)
    return fig


def chart_heatmap(df):
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
        textfont={"size": 10, "color": "white"}, hoverongaps=False,
        colorbar=dict(title="₹", tickfont=dict(color="#94a3b8"))
    ))
    fig.update_layout(**layout(300, "Yield Matrix: Route × Day"))
    return fig


def chart_weekend(df):
    if not df['is_weekend'].any() or df['is_weekend'].all():
        return None
    
    wd = df[~df['is_weekend']]['base_price'].mean()
    we = df[df['is_weekend']]['base_price'].mean()
    prem = ((we - wd) / wd * 100) if wd > 0 else 0
    
    fig = go.Figure(go.Bar(
        x=['Weekday', 'Weekend'], y=[wd, we],
        marker_color=[COLORS['blue'], COLORS['pink']],
        text=[f'₹{wd:.0f}', f'₹{we:.0f}'], textposition='outside', textfont=dict(color='#e2e8f0', size=12)
    ))
    fig.update_layout(**layout(280, f"Weekend: {prem:+.1f}%"))
    fig.update_layout(showlegend=False, bargap=0.5)
    fig.update_yaxes(title="Avg Price", gridcolor="#1e293b", zeroline=False)
    return fig


def chart_gauge(occ):
    color = COLORS['green'] if occ >= 70 else COLORS['yellow'] if occ >= 50 else COLORS['red']
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=occ,
        number={'suffix': '%', 'font': {'size': 32, 'color': '#f1f5f9'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#475569'},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': '#1e293b', 'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239,68,68,0.1)'},
                {'range': [50, 70], 'color': 'rgba(234,179,8,0.1)'},
                {'range': [70, 100], 'color': 'rgba(34,197,94,0.1)'}
            ]
        }
    ))
    fig.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=10, l=25, r=25))
    return fig


def chart_route_bar(df):
    stats = df.groupby('route')['occupancy'].mean().sort_values().reset_index()
    colors = [COLORS['red'] if v < 50 else COLORS['yellow'] if v < 70 else COLORS['green'] for v in stats['occupancy']]
    
    fig = go.Figure(go.Bar(
        y=stats['route'], x=stats['occupancy'], orientation='h', marker_color=colors,
        text=stats['occupancy'].round(1).astype(str) + '%', textposition='inside', textfont=dict(color='white', size=10)
    ))
    fig.update_layout(**layout(300, "Route Occupancy"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Occupancy %", gridcolor="#1e293b", zeroline=False)
    return fig


def chart_bus_box(df):
    fig = px.box(df, x='bus_type', y='base_price', color='bus_type',
                 color_discrete_sequence=[COLORS['blue'], COLORS['pink'], COLORS['purple'], COLORS['green']])
    fig.update_layout(**layout(300, "Price by Coach Type"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="", gridcolor="#1e293b")
    fig.update_yaxes(title="Price ₹", gridcolor="#1e293b", zeroline=False)
    return fig


# ==============================================================================
# 🖥️ MAIN APP
# ==============================================================================

def main():
    # Simple header with record count
    df, _ = load_data()
    df = preprocess(df)
    
    st.markdown(f"""
        <div class="dashboard-header">
            <h1 class="dashboard-title">📊 Pricing Dashboard</h1>
            <span class="dashboard-count">{len(df):,} records</span>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Avg Price", f"₹{df['base_price'].mean():,.0f}")
    k2.metric("Avg Occupancy", f"{df['occupancy'].mean():.1f}%")
    k3.metric("Seats Sold", f"{df['sold_seats'].sum():,}")
    k4.metric("Routes", f"{df['route'].nunique()}")
    
    if df['is_weekend'].any() and (~df['is_weekend']).any():
        prem = ((df[df['is_weekend']]['base_price'].mean() / df[~df['is_weekend']]['base_price'].mean()) - 1) * 100
        k5.metric("Weekend Lift", f"{prem:+.1f}%")
    else:
        k5.metric("Weekend Lift", "—")
    
    st.markdown("---")
    
    # Row 1: Demand Charts
    st.markdown("<div class='section-title'>Demand Analytics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = chart_demand_curve(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(chart_scatter(df), use_container_width=True)
    
    st.markdown("---")
    
    # Row 2: Yield Management
    st.markdown("<div class='section-title'>Yield Management</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
    with c1:
        st.plotly_chart(chart_heatmap(df), use_container_width=True)
    with c2:
        fig = chart_weekend(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.caption("Fleet Occupancy")
        st.plotly_chart(chart_gauge(df['occupancy'].mean()), use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: Route Analysis
    st.markdown("<div class='section-title'>Route Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_route_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(chart_bus_box(df), use_container_width=True)
    
    st.markdown("---")
    
    # Data Table with Filters
    st.markdown("<div class='section-title'>Data Export</div>", unsafe_allow_html=True)
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        routes_opt = ['All'] + sorted(df['route'].unique().tolist())
        sel_routes = st.multiselect("Routes", routes_opt, default=['All'], label_visibility="collapsed", placeholder="Filter routes...")
    with f2:
        types_opt = ['All'] + sorted(df['bus_type'].unique().tolist())
        sel_types = st.multiselect("Types", types_opt, default=['All'], label_visibility="collapsed", placeholder="Filter coach...")
    with f3:
        price_min, price_max = int(df['base_price'].min()), int(df['base_price'].max())
        price_rng = st.slider("Price", price_min, price_max, (price_min, price_max), label_visibility="collapsed")
    with f4:
        occ_rng = st.slider("Occupancy", 0, 100, (0, 100), label_visibility="collapsed")
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    if 'All' not in sel_routes and sel_routes:
        mask &= df['route'].isin(sel_routes)
    if 'All' not in sel_types and sel_types:
        mask &= df['bus_type'].isin(sel_types)
    mask &= (df['base_price'] >= price_rng[0]) & (df['base_price'] <= price_rng[1])
    mask &= (df['occupancy'] >= occ_rng[0]) & (df['occupancy'] <= occ_rng[1])
    fdf = df[mask]
    
    cols = ['travel_date', 'route', 'bus_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy']
    cols = [c for c in cols if c in fdf.columns]
    
    st.caption(f"Showing {len(fdf):,} of {len(df):,} records")
    st.dataframe(fdf[cols].sort_values('travel_date', ascending=False), use_container_width=True, height=350,
                 column_config={
                     "travel_date": st.column_config.DateColumn("Date", format="DD-MMM"),
                     "base_price": st.column_config.NumberColumn("Price", format="₹%d"),
                     "occupancy": st.column_config.ProgressColumn("Occ%", min_value=0, max_value=100)
                 })
    
    # Export buttons
    e1, e2, _ = st.columns([1, 1, 4])
    with e1:
        st.download_button("⬇️ CSV", fdf[cols].to_csv(index=False).encode(), f"data_{datetime.now():%Y%m%d}.csv", "text/csv", use_container_width=True)
    with e2:
        st.download_button("⬇️ Excel", to_excel(fdf[cols]), f"data_{datetime.now():%Y%m%d}.xlsx", 
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


if __name__ == "__main__":
    main()
