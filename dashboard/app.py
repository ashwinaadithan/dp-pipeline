"""
🚌 Dynamic Pricing Intelligence Platform
=========================================
Vignesh TAT | Enterprise-Grade Pricing Analytics

The ultimate tool for understanding and optimizing bus ticket pricing.
Built for data-driven pricing decisions.
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


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_all_data():
    """Fetch all bus data from database."""
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
            LIMIT 100000
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame(), None
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=300)
def fetch_price_history():
    """Fetch price change history."""
    conn, err = get_db()
    if not conn:
        return pd.DataFrame()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bus_id, base_price, available_seats, sold_seats, scraped_at
            FROM price_history 
            ORDER BY scraped_at DESC 
            LIMIT 50000
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


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
    df['route'] = df['from_city'] + ' → ' + df['to_city']
    df['total_seats'] = df['sold_seats'] + df['available_seats']
    df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100).round(1)
    df['travel_date'] = pd.to_datetime(df['travel_date'])
    df['scraped_at'] = pd.to_datetime(df['scraped_at'])
    df['day_type'] = df['travel_date'].apply(get_day_type)
    df['is_special'] = df['day_type'].apply(lambda x: x not in ['Weekday', 'Weekend'])
    df['day_of_week'] = df['travel_date'].dt.day_name()
    df['days_ahead'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    df['scrape_date'] = df['scraped_at'].dt.date
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Pricing Data')
    return out.getvalue()


# ==============================================================================
# PAGE CONFIG & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme
st.markdown("""
<style>
    /* Base */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Headers */
    h1 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e293b;
        padding: 8px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    /* Cards */
    .insight-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .insight-card h4 {
        color: #60a5fa;
        margin-bottom: 10px;
        font-size: 1rem;
    }
    .insight-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .insight-value {
        font-size: 2rem;
        font-weight: 700;
        color: #22c55e;
    }
    
    /* Data Table */
    [data-testid="stDataFrame"] {
        background: #1e293b;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    /* Selectbox, Multiselect */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #1e293b;
        border-color: #334155;
    }
    
    /* Date Input */
    .stDateInput > div > div {
        background: #1e293b;
        border-color: #334155;
    }
    
    /* Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        border: none;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CHART FUNCTIONS
# ==============================================================================

def chart_layout(height=400, title=""):
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,15,26,0.5)",
        "font": {"family": "Inter, sans-serif", "color": "#e2e8f0"},
        "height": height,
        "margin": {"l": 60, "r": 30, "t": 50 if title else 30, "b": 50},
        "title": {"text": title, "font": {"size": 16, "color": "#e2e8f0"}, "x": 0.02} if title else None,
        "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"},
        "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0}
    }


def chart_price_timeline(df, selected_routes):
    """Price changes over time for selected routes."""
    if df.empty or not selected_routes:
        return None
    
    filtered = df[df['route'].isin(selected_routes)].copy()
    if filtered.empty:
        return None
    
    # Aggregate by date and route
    agg = filtered.groupby(['travel_date', 'route']).agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16']
    
    for i, route in enumerate(selected_routes):
        route_data = agg[agg['route'] == route].sort_values('travel_date')
        fig.add_trace(go.Scatter(
            x=route_data['travel_date'],
            y=route_data['base_price'],
            name=route,
            mode='lines+markers',
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8),
            hovertemplate=f"<b>{route}</b><br>Date: %{{x|%d %b}}<br>Price: ₹%{{y:,.0f}}<extra></extra>"
        ))
    
    fig.update_layout(**chart_layout(380, "💰 Price Trends by Route"))
    fig.update_xaxes(title="Travel Date", tickformat="%d %b")
    fig.update_yaxes(title="Average Price (₹)", tickprefix="₹")
    
    return fig


def chart_days_ahead_impact(df):
    """How price changes as departure approaches."""
    if df.empty or 'days_ahead' not in df.columns:
        return None
    
    # Filter valid days ahead (0 to 14)
    filtered = df[(df['days_ahead'] >= 0) & (df['days_ahead'] <= 14)]
    if filtered.empty:
        return None
    
    agg = filtered.groupby('days_ahead').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg.columns = ['days_ahead', 'avg_price', 'avg_occupancy', 'samples']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price bars
    fig.add_trace(go.Bar(
        x=agg['days_ahead'],
        y=agg['avg_price'],
        name='Avg Price',
        marker_color='#3b82f6',
        opacity=0.8,
        hovertemplate="Day %{x}<br>Price: ₹%{y:,.0f}<extra></extra>"
    ), secondary_y=False)
    
    # Occupancy line
    fig.add_trace(go.Scatter(
        x=agg['days_ahead'],
        y=agg['avg_occupancy'],
        name='Occupancy %',
        mode='lines+markers',
        line=dict(color='#22c55e', width=3),
        marker=dict(size=8),
        hovertemplate="Day %{x}<br>Occupancy: %{y:.1f}%<extra></extra>"
    ), secondary_y=True)
    
    fig.update_layout(**chart_layout(380, "📅 Days to Departure Impact"))
    fig.update_xaxes(title="Days Before Departure", dtick=1)
    fig.update_yaxes(title="Price (₹)", tickprefix="₹", secondary_y=False)
    fig.update_yaxes(title="Occupancy %", ticksuffix="%", range=[0, 100], secondary_y=True)
    
    return fig


def chart_price_vs_occupancy(df):
    """Scatter plot showing price elasticity."""
    if df.empty:
        return None
    
    # Sample if too large
    sample = df.sample(min(2000, len(df))) if len(df) > 2000 else df
    
    fig = px.scatter(
        sample,
        x='base_price',
        y='occupancy',
        color='day_type',
        size='total_seats',
        hover_data=['route', 'travel_date'],
        color_discrete_map={
            'Weekday': '#64748b',
            'Weekend': '#8b5cf6',
            'Pongal': '#f59e0b',
            'Deepavali': '#ef4444',
            'Bhogi': '#f59e0b',
            'Mattu Pongal': '#f59e0b',
            'Kaanum Pongal': '#f59e0b'
        }
    )
    
    fig.update_layout(**chart_layout(400, "📊 Price vs Occupancy (Elasticity Analysis)"))
    fig.update_xaxes(title="Ticket Price (₹)", tickprefix="₹")
    fig.update_yaxes(title="Occupancy %", ticksuffix="%", range=[0, 105])
    
    # Add trendline annotation
    if len(sample) > 10:
        corr = sample['base_price'].corr(sample['occupancy'])
        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=f"Correlation: {corr:.2f}",
            showarrow=False,
            font=dict(size=12, color="#94a3b8"),
            bgcolor="#1e293b",
            borderpad=8
        )
    
    return fig


def chart_route_comparison(df):
    """Compare routes by occupancy and price."""
    if df.empty:
        return None
    
    agg = df.groupby('route').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg.columns = ['route', 'avg_price', 'avg_occupancy', 'trips']
    agg = agg.sort_values('avg_occupancy', ascending=True).tail(15)
    
    # Color by occupancy
    colors = ['#ef4444' if o < 40 else '#f59e0b' if o < 60 else '#22c55e' for o in agg['avg_occupancy']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=agg['route'],
        x=agg['avg_occupancy'],
        orientation='h',
        marker_color=colors,
        text=[f"{o:.1f}%" for o in agg['avg_occupancy']],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=11),
        hovertemplate="<b>%{y}</b><br>Occupancy: %{x:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(**chart_layout(450, "🛣️ Route Performance Ranking"))
    fig.update_layout(margin={"l": 180})
    fig.update_xaxes(title="Average Occupancy %", range=[0, 100], ticksuffix="%")
    
    return fig


def chart_day_type_analysis(df):
    """Weekend vs Weekday vs Festival pricing."""
    if df.empty:
        return None
    
    agg = df.groupby('day_type').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg.columns = ['day_type', 'avg_price', 'avg_occupancy', 'trips']
    
    # Order: Weekday first, then Weekend, then festivals
    order = ['Weekday', 'Weekend'] + [d for d in agg['day_type'] if d not in ['Weekday', 'Weekend']]
    agg['order'] = agg['day_type'].apply(lambda x: order.index(x) if x in order else 99)
    agg = agg.sort_values('order')
    
    colors = {'Weekday': '#64748b', 'Weekend': '#8b5cf6'}
    bar_colors = [colors.get(d, '#f59e0b') for d in agg['day_type']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=agg['day_type'],
        y=agg['avg_price'],
        marker_color=bar_colors,
        text=[f"₹{p:,.0f}" for p in agg['avg_price']],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=12, weight=600),
        hovertemplate="<b>%{x}</b><br>Avg Price: ₹%{y:,.0f}<br>Trips: %{customdata}<extra></extra>",
        customdata=agg['trips']
    ))
    
    fig.update_layout(**chart_layout(380, "🎉 Day Type Price Analysis"))
    fig.update_yaxes(title="Average Price (₹)", tickprefix="₹")
    
    return fig


def chart_hourly_price_changes(df):
    """Price changes by scrape hour."""
    if df.empty:
        return None
    
    df_copy = df.copy()
    df_copy['scrape_hour'] = df_copy['scraped_at'].dt.hour
    
    agg = df_copy.groupby('scrape_hour').agg({
        'base_price': 'mean',
        'bus_id': 'count'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=agg['scrape_hour'],
        y=agg['base_price'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#3b82f6', width=2),
        fillcolor='rgba(59, 130, 246, 0.2)',
        hovertemplate="Hour: %{x}:00<br>Avg Price: ₹%{y:,.0f}<extra></extra>"
    ))
    
    fig.update_layout(**chart_layout(300, "⏰ Price by Time of Day"))
    fig.update_xaxes(title="Hour of Day", dtick=3, range=[0, 23])
    fig.update_yaxes(title="Average Price (₹)", tickprefix="₹")
    
    return fig


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Load Data
    df_raw, err = fetch_all_data()
    if df_raw.empty:
        st.error(f"❌ No data available: {err}")
        st.info("Please check your database connection in Streamlit Secrets.")
        return
    
    df = preprocess(df_raw)
    
    # ==================== SIDEBAR FILTERS ====================
    st.sidebar.markdown("# 🎛️ Filters")
    st.sidebar.markdown("---")
    
    # Date Range Filter
    st.sidebar.markdown("### 📅 Date Range")
    min_date = df['travel_date'].min().date()
    max_date = df['travel_date'].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)
    
    # Route Filter
    st.sidebar.markdown("### 🛣️ Routes")
    all_routes = sorted(df['route'].unique())
    selected_routes = st.sidebar.multiselect(
        "Select routes to analyze",
        options=all_routes,
        default=all_routes[:5] if len(all_routes) > 5 else all_routes
    )
    
    # Bus Type Filter
    st.sidebar.markdown("### 🚌 Bus Type")
    bus_types = sorted(df['bus_type'].unique())
    selected_types = st.sidebar.multiselect(
        "Select bus types",
        options=bus_types,
        default=bus_types
    )
    
    # Day Type Filter
    st.sidebar.markdown("### 📆 Day Type")
    day_types = ['Weekday', 'Weekend'] + sorted([d for d in df['day_type'].unique() if d not in ['Weekday', 'Weekend']])
    selected_days = st.sidebar.multiselect(
        "Select day types",
        options=day_types,
        default=day_types
    )
    
    # Apply Filters
    mask = (
        (df['travel_date'].dt.date >= start_date) &
        (df['travel_date'].dt.date <= end_date) &
        (df['route'].isin(selected_routes) if selected_routes else True) &
        (df['bus_type'].isin(selected_types) if selected_types else True) &
        (df['day_type'].isin(selected_days) if selected_days else True)
    )
    df_filtered = df[mask]
    
    # ==================== HEADER ====================
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 5px;">📊 Dynamic Pricing Intelligence</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">Vignesh TAT | Real-time Bus Pricing Analytics & Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown(f"<p style='text-align: center; color: #64748b; margin-bottom: 30px;'>Analyzing <b style='color: #60a5fa;'>{len(df_filtered):,}</b> records from <b style='color: #60a5fa;'>{len(selected_routes)}</b> routes</p>", unsafe_allow_html=True)
    
    # ==================== KPI METRICS ====================
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if not df_filtered.empty:
        avg_price = df_filtered['base_price'].mean()
        avg_occupancy = df_filtered['occupancy'].mean()
        total_trips = len(df_filtered)
        total_routes = df_filtered['route'].nunique()
        
        # Calculate price change
        weekday_avg = df_filtered[df_filtered['day_type'] == 'Weekday']['base_price'].mean()
        special_avg = df_filtered[df_filtered['is_special']]['base_price'].mean()
        if pd.notna(weekday_avg) and weekday_avg > 0 and pd.notna(special_avg):
            festival_premium = ((special_avg - weekday_avg) / weekday_avg) * 100
        else:
            festival_premium = 0
        
        with col1:
            st.metric("Average Price", f"₹{avg_price:,.0f}")
        with col2:
            st.metric("Fleet Occupancy", f"{avg_occupancy:.1f}%")
        with col3:
            st.metric("Total Trips", f"{total_trips:,}")
        with col4:
            st.metric("Active Routes", f"{total_routes}")
        with col5:
            st.metric("Festival Premium", f"+{festival_premium:.1f}%")
    
    st.markdown("---")
    
    # ==================== MAIN TABS ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "🎯 Demand Insights", "🛣️ Route Intelligence", "📋 Data Explorer"])
    
    # ==================== TAB 1: PRICE ANALYSIS ====================
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Select Routes for Price Trend Comparison")
            compare_routes = st.multiselect(
                "Choose up to 6 routes",
                options=all_routes,
                default=all_routes[:3] if len(all_routes) >= 3 else all_routes,
                key="price_trend_routes"
            )
            
            fig = chart_price_timeline(df_filtered, compare_routes[:6])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select routes to see price trends")
        
        with col2:
            st.markdown("#### 💡 Key Insight")
            if not df_filtered.empty and compare_routes:
                route_stats = df_filtered[df_filtered['route'].isin(compare_routes)].groupby('route')['base_price'].agg(['min', 'max', 'mean'])
                if not route_stats.empty:
                    best_route = route_stats['mean'].idxmax()
                    highest_price = route_stats['mean'].max()
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Highest Priced Route</h4>
                        <p class="insight-value">{best_route.split(' → ')[0][:8]}...</p>
                        <p>Average ticket: <b>₹{highest_price:,.0f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    price_range = route_stats['max'].max() - route_stats['min'].min()
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Price Volatility</h4>
                        <p class="insight-value">₹{price_range:,.0f}</p>
                        <p>Max price swing across selected routes</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = chart_days_ahead_impact(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = chart_day_type_analysis(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: DEMAND INSIGHTS ====================
    with tab2:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### Understanding Price Elasticity")
            st.caption("How does price affect demand? The scatter below shows the relationship.")
            fig = chart_price_vs_occupancy(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🧠 Elasticity Insights")
            
            if not df_filtered.empty:
                # Calculate elasticity stats
                low_price = df_filtered[df_filtered['base_price'] < df_filtered['base_price'].quantile(0.33)]['occupancy'].mean()
                high_price = df_filtered[df_filtered['base_price'] > df_filtered['base_price'].quantile(0.66)]['occupancy'].mean()
                
                if pd.notna(low_price) and pd.notna(high_price):
                    occ_diff = high_price - low_price
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Low Price Occupancy</h4>
                        <p class="insight-value" style="color: #22c55e;">{low_price:.1f}%</p>
                        <p>When prices are in bottom 33%</p>
                    </div>
                    
                    <div class="insight-card">
                        <h4>High Price Occupancy</h4>
                        <p class="insight-value" style="color: {'#ef4444' if high_price < low_price else '#22c55e'};">{high_price:.1f}%</p>
                        <p>When prices are in top 33%</p>
                    </div>
                    
                    <div class="insight-card">
                        <h4>Demand Sensitivity</h4>
                        <p class="insight-value" style="color: #f59e0b;">{abs(occ_diff):.1f}%</p>
                        <p>Occupancy {'drops' if occ_diff < 0 else 'increases'} when price is high</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        fig = chart_hourly_price_changes(df_filtered)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: ROUTE INTELLIGENCE ====================
    with tab3:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig = chart_route_comparison(df_filtered)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Route Summary")
            if not df_filtered.empty:
                route_summary = df_filtered.groupby('route').agg({
                    'base_price': 'mean',
                    'occupancy': 'mean',
                    'bus_id': 'count'
                }).reset_index()
                route_summary.columns = ['Route', 'Avg Price', 'Occupancy %', 'Trips']
                route_summary = route_summary.sort_values('Trips', ascending=False).head(10)
                
                # Format
                route_summary['Avg Price'] = route_summary['Avg Price'].apply(lambda x: f"₹{x:,.0f}")
                route_summary['Occupancy %'] = route_summary['Occupancy %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(route_summary, hide_index=True, use_container_width=True)
                
                # Top performer
                best = df_filtered.groupby('route')['occupancy'].mean().idxmax()
                best_occ = df_filtered.groupby('route')['occupancy'].mean().max()
                st.success(f"🏆 **Best Performer:** {best} ({best_occ:.1f}% occupancy)")
    
    # ==================== TAB 4: DATA EXPLORER ====================
    with tab4:
        st.markdown("#### 🔍 Explore Raw Data")
        st.caption("Filter, sort, and export your pricing data")
        
        # Additional filters for the table
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_min = st.number_input("Min Price (₹)", value=0, step=100)
        with col2:
            price_max = st.number_input("Max Price (₹)", value=int(df_filtered['base_price'].max()) if not df_filtered.empty else 5000, step=100)
        with col3:
            occ_min = st.slider("Min Occupancy %", 0, 100, 0)
        with col4:
            occ_max = st.slider("Max Occupancy %", 0, 100, 100)
        
        # Apply table filters
        table_mask = (
            (df_filtered['base_price'] >= price_min) &
            (df_filtered['base_price'] <= price_max) &
            (df_filtered['occupancy'] >= occ_min) &
            (df_filtered['occupancy'] <= occ_max)
        )
        df_table = df_filtered[table_mask].copy()
        
        # Prepare display columns
        display_cols = ['travel_date', 'route', 'bus_type', 'departure_time', 'base_price', 
                       'available_seats', 'sold_seats', 'occupancy', 'day_type', 'days_ahead', 'scraped_at']
        display_cols = [c for c in display_cols if c in df_table.columns]
        
        st.markdown(f"**{len(df_table):,} records** matching filters")
        
        # Show data
        st.dataframe(
            df_table[display_cols].sort_values('scraped_at', ascending=False).head(1000),
            use_container_width=True,
            height=500
        )
        
        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            csv = df_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📄 Download CSV",
                csv,
                f"pricing_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
        
        with col2:
            excel = to_excel(df_table)
            st.download_button(
                "📊 Download Excel",
                excel,
                f"pricing_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <p>🚌 Dynamic Pricing Intelligence Platform | Data updated every hour</p>
        <p style="font-size: 0.8rem;">Built for Vignesh TAT pricing optimization</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
