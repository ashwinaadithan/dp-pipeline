"""
Dynamic Pricing Intelligence Platform
=====================================
Vignesh TAT | Enterprise Pricing Analytics
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
# DATABASE CONNECTION
# ==============================================================================

src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db():
    try:
        db_url = st.secrets.get('NEON_DATABASE_URL') if hasattr(st, 'secrets') else os.getenv('NEON_DATABASE_URL')
        if not db_url:
            return None, "Database not configured"
        import psycopg
        return psycopg.connect(db_url), None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60)
def fetch_all_data():
    conn, err = get_db()
    if not conn:
        return pd.DataFrame(), err
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bus_id, operator, bus_type, from_city, to_city, travel_date,
                   departure_time, base_price, available_seats, sold_seats, 
                   min_price, max_price, scraped_at
            FROM buses 
            ORDER BY scraped_at DESC
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(), None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================

FESTIVALS = {
    (1, 13): "Bhogi", (1, 14): "Pongal", (1, 15): "Mattu Pongal", (1, 16): "Kaanum Pongal",
    (1, 26): "Republic Day", (8, 15): "Independence Day", (10, 2): "Gandhi Jayanti",
    (10, 12): "Dussehra", (11, 1): "Deepavali", (12, 25): "Christmas"
}

def get_day_type(date):
    if isinstance(date, str):
        date = pd.to_datetime(date)
    key = (date.month, date.day)
    if key in FESTIVALS:
        return FESTIVALS[key]
    elif date.weekday() >= 5:
        return "Weekend"
    return "Weekday"


def preprocess(df):
    if df.empty:
        return df
    df = df.copy()
    df['route'] = df['from_city'] + ' > ' + df['to_city']
    df['total_seats'] = df['sold_seats'] + df['available_seats']
    df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100).round(1)
    df['travel_date'] = pd.to_datetime(df['travel_date'])
    df['scraped_at'] = pd.to_datetime(df['scraped_at'])
    df['day_type'] = df['travel_date'].apply(get_day_type)
    df['is_special'] = df['day_type'].apply(lambda x: x not in ['Weekday', 'Weekend'])
    df['day_of_week'] = df['travel_date'].dt.day_name()
    df['days_ahead'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Data')
    return out.getvalue()


# ==============================================================================
# PAGE CONFIG & STYLING - StockPeers Dark Blue Theme
# ==============================================================================

st.set_page_config(
    page_title="Pricing Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# StockPeers-inspired dark blue theme
st.markdown("""
<style>
    /* Main Background - Deep Navy */
    .stApp {
        background-color: #0d1117;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    h1 { font-size: 1.75rem; }
    h2 { font-size: 1.25rem; }
    h3 { font-size: 1rem; }
    
    /* Subtext */
    p, span, label {
        color: #8b949e;
    }
    
    /* Metrics - Clean Cards */
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 600;
        color: #58a6ff;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tabs - Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        font-weight: 500;
        padding: 12px 20px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #c9d1d9;
        border-bottom: 2px solid #58a6ff;
        background: transparent;
    }
    
    /* Cards */
    .card {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
    }
    .card-title {
        color: #c9d1d9;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .card-value.green { color: #3fb950; }
    .card-value.red { color: #f85149; }
    .card-value.yellow { color: #d29922; }
    .card-subtitle {
        color: #8b949e;
        font-size: 0.75rem;
        margin-top: 4px;
    }
    
    /* Pill Tags */
    .pill {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        color: #c9d1d9;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
    }
    .pill.active {
        background: #388bfd;
        border-color: #388bfd;
        color: white;
    }
    
    /* Data Table */
    [data-testid="stDataFrame"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 6px;
    }
    
    /* Inputs */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stDateInput > div > div {
        background: #21262d;
        border-color: #30363d;
        color: #c9d1d9;
    }
    
    /* Buttons */
    .stDownloadButton > button {
        background: #238636;
        color: white;
        border: none;
        font-weight: 500;
        border-radius: 6px;
    }
    .stDownloadButton > button:hover {
        background: #2ea043;
    }
    
    /* Dividers */
    hr {
        border-color: #21262d;
    }
    
    /* Multiselect Tags - White text for readability */
    [data-baseweb="tag"] {
        background-color: #388bfd !important;
        color: white !important;
    }
    [data-baseweb="tag"] span {
        color: white !important;
    }
    [data-baseweb="tag"] svg {
        fill: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CHART STYLING
# ==============================================================================

COLORS = {
    'blue': '#58a6ff',
    'green': '#3fb950',
    'yellow': '#d29922',
    'red': '#f85149',
    'purple': '#a371f7',
    'cyan': '#56d4dd',
    'orange': '#db6d28',
    'pink': '#db61a2'
}

CHART_COLORS = [COLORS['blue'], COLORS['green'], COLORS['yellow'], COLORS['red'], 
                COLORS['purple'], COLORS['cyan'], COLORS['orange'], COLORS['pink']]


def chart_layout(height=380, title=""):
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(13,17,23,0.8)",
        "font": {"family": "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica", "color": "#c9d1d9", "size": 12},
        "height": height,
        "margin": {"l": 50, "r": 30, "t": 40 if title else 20, "b": 40},
        "title": {"text": title, "font": {"size": 14, "color": "#c9d1d9"}, "x": 0.02} if title else None,
        "xaxis": {"gridcolor": "#21262d", "zerolinecolor": "#21262d", "linecolor": "#21262d"},
        "yaxis": {"gridcolor": "#21262d", "zerolinecolor": "#21262d", "linecolor": "#21262d"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "font": {"size": 11}}
    }


def chart_price_timeline(df, selected_routes):
    if df.empty or not selected_routes:
        return None
    
    filtered = df[df['route'].isin(selected_routes)].copy()
    if filtered.empty:
        return None
    
    agg = filtered.groupby(['travel_date', 'route']).agg({
        'base_price': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    for i, route in enumerate(selected_routes[:8]):
        route_data = agg[agg['route'] == route].sort_values('travel_date')
        fig.add_trace(go.Scatter(
            x=route_data['travel_date'],
            y=route_data['base_price'],
            name=route,
            mode='lines+markers',
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{route}</b><br>%{{x|%d %b}}: Rs %{{y:,.0f}}<extra></extra>"
        ))
    
    fig.update_layout(**chart_layout(360, "Price Trends"))
    fig.update_xaxes(title="Date", tickformat="%d %b")
    fig.update_yaxes(title="Avg Price (Rs)", tickprefix="Rs ")
    
    return fig


def chart_days_ahead(df):
    if df.empty or 'days_ahead' not in df.columns:
        return None
    
    filtered = df[(df['days_ahead'] >= 0) & (df['days_ahead'] <= 14)]
    if filtered.empty:
        return None
    
    agg = filtered.groupby('days_ahead').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=agg['days_ahead'],
        y=agg['base_price'],
        name='Price',
        marker_color=COLORS['blue'],
        opacity=0.85
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=agg['days_ahead'],
        y=agg['occupancy'],
        name='Occupancy',
        mode='lines+markers',
        line=dict(color=COLORS['green'], width=2),
        marker=dict(size=6)
    ), secondary_y=True)
    
    fig.update_layout(**chart_layout(340, "Days to Departure"))
    fig.update_xaxes(title="Days Ahead", dtick=1)
    fig.update_yaxes(title="Price (Rs)", tickprefix="Rs ", secondary_y=False)
    fig.update_yaxes(title="Occupancy %", ticksuffix="%", range=[0, 100], secondary_y=True)
    
    return fig


def chart_elasticity(df):
    if df.empty:
        return None
    
    sample = df.sample(min(1500, len(df))) if len(df) > 1500 else df
    
    fig = px.scatter(
        sample, x='base_price', y='occupancy',
        color='day_type',
        hover_data=['route'],
        color_discrete_map={
            'Weekday': '#8b949e', 'Weekend': COLORS['purple'],
            'Pongal': COLORS['yellow'], 'Deepavali': COLORS['red']
        }
    )
    
    fig.update_layout(**chart_layout(380, "Price vs Occupancy"))
    fig.update_xaxes(title="Price (Rs)", tickprefix="Rs ")
    fig.update_yaxes(title="Occupancy %", ticksuffix="%", range=[0, 105])
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    
    return fig


def chart_route_ranking(df):
    if df.empty:
        return None
    
    agg = df.groupby('route').agg({
        'occupancy': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg = agg.sort_values('occupancy', ascending=True).tail(12)
    
    colors = [COLORS['red'] if o < 40 else COLORS['yellow'] if o < 60 else COLORS['green'] for o in agg['occupancy']]
    
    fig = go.Figure(go.Bar(
        y=agg['route'],
        x=agg['occupancy'],
        orientation='h',
        marker_color=colors,
        text=[f"{o:.1f}%" for o in agg['occupancy']],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=10)
    ))
    
    fig.update_layout(**chart_layout(400, "Route Performance"))
    fig.update_layout(margin={"l": 160})
    fig.update_xaxes(title="Occupancy %", range=[0, 100], ticksuffix="%")
    
    return fig


def chart_day_type(df):
    if df.empty:
        return None
    
    agg = df.groupby('day_type').agg({
        'base_price': 'mean',
        'bus_id': 'count'
    }).reset_index().sort_values('base_price', ascending=False)
    
    colors = {'Weekday': '#8b949e', 'Weekend': COLORS['purple']}
    bar_colors = [colors.get(d, COLORS['yellow']) for d in agg['day_type']]
    
    fig = go.Figure(go.Bar(
        x=agg['day_type'],
        y=agg['base_price'],
        marker_color=bar_colors,
        text=[f"Rs {p:,.0f}" for p in agg['base_price']],
        textposition='outside',
        textfont=dict(color='#c9d1d9', size=11)
    ))
    
    fig.update_layout(**chart_layout(340, "Price by Day Type"))
    fig.update_yaxes(title="Avg Price (Rs)", tickprefix="Rs ")
    
    return fig


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    df_raw, err = fetch_all_data()
    if df_raw.empty:
        st.error(f"No data available: {err}")
        return
    
    df = preprocess(df_raw)
    
    # Sidebar Filters
    st.sidebar.markdown("### Filters")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Date Range
    st.sidebar.markdown("**Date Range**")
    min_date = df['travel_date'].min().date()
    max_date = df['travel_date'].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
    with col2:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, label_visibility="collapsed")
    
    # Routes
    st.sidebar.markdown("**Routes**")
    all_routes = sorted(df['route'].unique())
    route_selection = st.sidebar.multiselect(
        "Routes", options=["All Routes"] + all_routes, default=["All Routes"], label_visibility="collapsed"
    )
    selected_routes = all_routes if "All Routes" in route_selection else route_selection
    
    # Bus Type (Seater/Sleeper)
    st.sidebar.markdown("**Bus Type**")
    bus_types = sorted(df['bus_type'].unique())
    type_selection = st.sidebar.multiselect(
        "Bus Type", options=["All Types"] + bus_types, default=["All Types"], label_visibility="collapsed"
    )
    selected_types = bus_types if "All Types" in type_selection else type_selection
    
    # Day Type
    st.sidebar.markdown("**Day Type**")
    day_types = ['Weekday', 'Weekend'] + sorted([d for d in df['day_type'].unique() if d not in ['Weekday', 'Weekend']])
    day_selection = st.sidebar.multiselect(
        "Day Type", options=["All Days"] + day_types, default=["All Days"], label_visibility="collapsed"
    )
    selected_days = day_types if "All Days" in day_selection else day_selection
    
    # Apply filters
    mask = (
        (df['travel_date'].dt.date >= start_date) &
        (df['travel_date'].dt.date <= end_date) &
        (df['route'].isin(selected_routes) if selected_routes else True) &
        (df['bus_type'].isin(selected_types) if selected_types else True) &
        (df['day_type'].isin(selected_days) if selected_days else True)
    )
    df_filtered = df[mask]
    
    # Header
    st.markdown("# Dynamic Pricing Intelligence")
    st.markdown(f"<p style='color:#8b949e; margin-top:-10px;'>Vignesh TAT | {len(df):,} total records | Showing {len(df_filtered):,} filtered across {len(selected_routes)} routes</p>", unsafe_allow_html=True)
    
    # KPI Metrics
    if not df_filtered.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        avg_price = df_filtered['base_price'].mean()
        avg_occ = df_filtered['occupancy'].mean()
        total = len(df_filtered)
        routes = df_filtered['route'].nunique()
        
        weekday_avg = df_filtered[df_filtered['day_type'] == 'Weekday']['base_price'].mean()
        special_avg = df_filtered[df_filtered['is_special']]['base_price'].mean()
        premium = ((special_avg - weekday_avg) / weekday_avg * 100) if pd.notna(weekday_avg) and weekday_avg > 0 and pd.notna(special_avg) else 0
        
        with col1:
            st.metric("Avg Price", f"Rs {avg_price:,.0f}")
        with col2:
            st.metric("Occupancy", f"{avg_occ:.1f}%")
        with col3:
            st.metric("Total Trips", f"{total:,}")
        with col4:
            st.metric("Routes", f"{routes}")
        with col5:
            st.metric("Festival Premium", f"+{premium:.1f}%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Demand Insights", "Route Performance", "Data Explorer"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### Route Comparison")
            compare_routes = st.multiselect(
                "Select routes to compare",
                options=all_routes,
                default=selected_routes[:4] if selected_routes else all_routes[:4],
                key="compare",
                label_visibility="collapsed"
            )
            fig = chart_price_timeline(df_filtered, compare_routes[:8])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Metrics")
            if not df_filtered.empty and compare_routes:
                route_stats = df_filtered[df_filtered['route'].isin(compare_routes)].groupby('route')['base_price'].agg(['min', 'max', 'mean'])
                if not route_stats.empty:
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Highest Avg Price</div>
                        <div class="card-value">Rs {route_stats['mean'].max():,.0f}</div>
                        <div class="card-subtitle">{route_stats['mean'].idxmax()}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Price Range</div>
                        <div class="card-value yellow">Rs {route_stats['max'].max() - route_stats['min'].min():,.0f}</div>
                        <div class="card-subtitle">Max swing across routes</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = chart_days_ahead(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = chart_day_type(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### Price Elasticity")
            st.caption("How price affects demand. Color indicates day type.")
            fig = chart_elasticity(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Elasticity Analysis")
            if not df_filtered.empty:
                low_q = df_filtered['base_price'].quantile(0.33)
                high_q = df_filtered['base_price'].quantile(0.66)
                low_occ = df_filtered[df_filtered['base_price'] < low_q]['occupancy'].mean()
                high_occ = df_filtered[df_filtered['base_price'] > high_q]['occupancy'].mean()
                
                if pd.notna(low_occ) and pd.notna(high_occ):
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">Low Price Segment</div>
                        <div class="card-value green">{low_occ:.1f}%</div>
                        <div class="card-subtitle">Below Rs {low_q:,.0f}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">High Price Segment</div>
                        <div class="card-value {'red' if high_occ < low_occ else 'green'}">{high_occ:.1f}%</div>
                        <div class="card-subtitle">Above Rs {high_q:,.0f}</div>
                    </div>
                    <div class="card">
                        <div class="card-title">Demand Sensitivity</div>
                        <div class="card-value yellow">{abs(high_occ - low_occ):.1f}%</div>
                        <div class="card-subtitle">Occupancy change with price</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig = chart_route_ranking(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Route Summary")
            if not df_filtered.empty:
                summary = df_filtered.groupby('route').agg({
                    'base_price': 'mean',
                    'occupancy': 'mean',
                    'bus_id': 'count'
                }).reset_index()
                summary.columns = ['Route', 'Avg Price', 'Occ %', 'Trips']
                summary = summary.sort_values('Trips', ascending=False).head(8)
                summary['Avg Price'] = summary['Avg Price'].apply(lambda x: f"Rs {x:,.0f}")
                summary['Occ %'] = summary['Occ %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(summary, hide_index=True, use_container_width=True)
    
    with tab4:
        st.markdown("#### Data Explorer")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price_min = st.number_input("Min Price", value=0, step=100)
        with col2:
            price_max = st.number_input("Max Price", value=int(df_filtered['base_price'].max()) if not df_filtered.empty else 5000, step=100)
        with col3:
            occ_min = st.slider("Min Occ %", 0, 100, 0)
        with col4:
            occ_max = st.slider("Max Occ %", 0, 100, 100)
        
        table_mask = (
            (df_filtered['base_price'] >= price_min) &
            (df_filtered['base_price'] <= price_max) &
            (df_filtered['occupancy'] >= occ_min) &
            (df_filtered['occupancy'] <= occ_max)
        )
        df_table = df_filtered[table_mask].copy()
        
        st.markdown(f"**{len(df_table):,} records** matching filters")
        
        cols = ['bus_id', 'travel_date', 'route', 'bus_type', 'departure_time', 'base_price', 
                'available_seats', 'sold_seats', 'occupancy', 'day_type']
        cols = [c for c in cols if c in df_table.columns]
        
        st.dataframe(
            df_table[cols].sort_values('travel_date', ascending=False).head(500),
            use_container_width=True,
            height=350
        )
        
        # Bus Price History
        st.markdown("---")
        st.markdown("#### 📈 Bus Price History")
        
        if not df_table.empty and 'bus_id' in df_table.columns:
            bus_ids = df_table['bus_id'].unique().tolist()
            selected_bus = st.selectbox("Select a bus to view price history", options=bus_ids, index=0)
            
            if selected_bus:
                bus_history = df[df['bus_id'] == selected_bus].sort_values('scraped_at')
                if len(bus_history) > 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=bus_history['scraped_at'],
                        y=bus_history['base_price'],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color=COLORS['blue'], width=2),
                        marker=dict(size=8),
                        hovertemplate="<b>%{x}</b><br>Rs %{y:,.0f}<extra></extra>"
                    ))
                    fig.update_layout(**chart_layout(300, f"Price History: {selected_bus}"))
                    fig.update_xaxes(title="Scraped At")
                    fig.update_yaxes(title="Price (Rs)", tickprefix="Rs ")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Price", f"Rs {bus_history['base_price'].min():,.0f}")
                    with col2:
                        st.metric("Max Price", f"Rs {bus_history['base_price'].max():,.0f}")
                    with col3:
                        price_change = bus_history['base_price'].iloc[-1] - bus_history['base_price'].iloc[0]
                        st.metric("Price Change", f"Rs {price_change:+,.0f}")
                else:
                    st.info(f"Only 1 data point for this bus. Need multiple scrapes to show price history.")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            csv = df_table.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, f"data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        with col2:
            excel = to_excel(df_table)
            st.download_button("Download Excel", excel, f"data_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.8rem;'>Dynamic Pricing Intelligence | Data updated hourly</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
