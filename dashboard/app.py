"""
Dynamic Pricing Dashboard
=========================
Vignesh TAT | Real-time Analytics
Professional Client-Ready Design
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


def fetch_buses(limit=50000):
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


def fetch_buses_with_history():
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
                COUNT(*) as data_points,
                MAX(ph.base_price) - MIN(ph.base_price) as price_variance
            FROM price_history ph
            JOIN buses b ON ph.bus_id = b.bus_id
            GROUP BY ph.bus_id, b.from_city, b.to_city, b.travel_date, b.departure_time, b.bus_type
            ORDER BY COUNT(*) DESC
            LIMIT 50
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else pd.DataFrame()
    except:
        return pd.DataFrame()


# ==============================================================================
# STYLING - Professional Clean Design
# ==============================================================================

st.set_page_config(page_title="Vignesh TAT Pricing", page_icon="", layout="wide", initial_sidebar_state="collapsed")

# Color palette inspired by StockPeers
COLORS = {
    'primary': '#4F8EF7',      # Blue
    'secondary': '#10B981',    # Green
    'accent': '#F59E0B',       # Amber
    'danger': '#EF4444',       # Red
    'purple': '#8B5CF6',       # Purple
    'teal': '#14B8A6',         # Teal
    'bg_dark': '#0E1117',
    'bg_card': '#1A1F2E',
    'border': '#2D3748',
    'text': '#E2E8F0',
    'text_muted': '#64748B'
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        background-color: {COLORS['bg_dark']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    .block-container {{ padding: 1.5rem 2rem !important; max-width: 1400px; }}
    
    /* Header */
    .main-header {{
        margin-bottom: 1.5rem;
    }}
    .main-header h1 {{
        font-size: 1.75rem;
        font-weight: 600;
        color: {COLORS['text']};
        margin: 0 0 0.25rem 0;
    }}
    .main-header p {{
        color: {COLORS['text_muted']};
        font-size: 0.9rem;
        margin: 0;
    }}
    
    /* Metric Cards */
    div[data-testid="metric-container"] {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
    }}
    div[data-testid="metric-container"] label {{
        color: {COLORS['text_muted']} !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
        color: {COLORS['text']} !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }}
    
    /* Pills/Tags */
    .pill {{
        display: inline-block;
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['primary']};
        color: {COLORS['primary']};
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.2s;
    }}
    .pill.active {{
        background: {COLORS['primary']};
        color: white;
    }}
    .pill:hover {{
        background: {COLORS['primary']};
        color: white;
    }}
    
    /* Section Headers */
    .section-header {{
        color: {COLORS['text']};
        font-size: 1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid {COLORS['border']};
    }}
    
    /* Chart Container */
    .chart-box {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    
    /* Indicator badges */
    .badge-up {{
        background: rgba(16, 185, 129, 0.15);
        color: {COLORS['secondary']};
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-down {{
        background: rgba(239, 68, 68, 0.15);
        color: {COLORS['danger']};
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-neutral {{
        background: rgba(100, 116, 139, 0.15);
        color: {COLORS['text_muted']};
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }}
    
    /* Day type badges */
    .badge-festival {{
        background: {COLORS['accent']};
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-weekend {{
        background: {COLORS['purple']};
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    
    /* Data Table */
    .dataframe {{
        font-size: 0.85rem !important;
    }}
    
    /* Hide default elements */
    #MainMenu, footer, header {{ visibility: hidden; }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {COLORS['bg_card']};
        border-radius: 8px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        color: {COLORS['text_muted']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']};
        color: white;
    }}
    
    /* Multiselect */
    .stMultiSelect > div {{
        background: {COLORS['bg_card']};
        border-color: {COLORS['border']};
    }}
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
    df['day_type'] = df['travel_date'].apply(get_day_type)
    df['is_festival'] = df['day_type'].apply(lambda x: x not in ['Weekday', 'Weekend'])
    df['is_weekend'] = df['day_type'] == 'Weekend'
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['days_ahead'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    return df


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Pricing Data')
    return out.getvalue()


# ==============================================================================
# CHARTS - Clean Professional Style
# ==============================================================================

def base_layout(height=320, title=""):
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": COLORS['text'], "size": 12},
        "height": height,
        "margin": {"l": 50, "r": 30, "t": 40, "b": 40},
        "title": {"text": title, "font": {"size": 14, "color": COLORS['text']}, "x": 0},
        "xaxis": {"gridcolor": COLORS['border'], "zerolinecolor": COLORS['border']},
        "yaxis": {"gridcolor": COLORS['border'], "zerolinecolor": COLORS['border']}
    }


def chart_price_trends(history_df, buses_info):
    """Multi-line price trend chart like StockPeers."""
    if history_df.empty:
        return None
    
    fig = go.Figure()
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
              COLORS['danger'], COLORS['purple'], COLORS['teal']]
    
    # Group by bus_id and plot each
    for i, (bus_id, group) in enumerate(history_df.groupby('bus_id')):
        if i >= 6:  # Limit to 6 buses
            break
        
        group = group.sort_values('scraped_at')
        
        # Get route name from buses_info
        bus_info = buses_info[buses_info['bus_id'] == bus_id]
        if not bus_info.empty:
            label = f"{bus_info.iloc[0]['from_city']} > {bus_info.iloc[0]['to_city']}"
        else:
            label = bus_id[:20]
        
        fig.add_trace(go.Scatter(
            x=group['scraped_at'],
            y=group['base_price'],
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(**base_layout(350, "Price Trends"))
    fig.update_layout(
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, font=dict(size=11)),
        margin={"l": 50, "r": 150, "t": 40, "b": 40}
    )
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Price (INR)")
    
    return fig


def chart_seat_prices(seat_df):
    """Seat price comparison over time."""
    if seat_df.empty:
        return None
    
    seat_df = seat_df.copy()
    seat_df['scraped_at'] = pd.to_datetime(seat_df['scraped_at'])
    
    seats = seat_df['seat_id'].unique()[:8]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], 
              COLORS['danger'], COLORS['purple'], COLORS['teal'], '#EC4899', '#06B6D4']
    
    fig = go.Figure()
    
    for i, seat in enumerate(seats):
        data = seat_df[seat_df['seat_id'] == seat].sort_values('scraped_at')
        is_window = data['is_window'].iloc[0] if len(data) > 0 else False
        label = f"{seat} (W)" if is_window else seat
        
        fig.add_trace(go.Scatter(
            x=data['scraped_at'],
            y=data['price'],
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5)
        ))
    
    fig.update_layout(**base_layout(320, "Seat Price Comparison"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10))
    )
    fig.update_xaxes(title="Time")
    fig.update_yaxes(title="Price (INR)")
    
    return fig


def chart_day_comparison(df):
    """Bar chart comparing weekday vs weekend vs festival prices."""
    if df.empty:
        return None
    
    # Aggregate by day type
    agg = df.groupby('day_type').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg.columns = ['Day Type', 'Avg Price', 'Avg Occupancy', 'Count']
    agg = agg.sort_values('Avg Price', ascending=True)
    
    # Assign colors
    color_map = {'Weekday': COLORS['primary'], 'Weekend': COLORS['purple']}
    for festival in FESTIVALS.values():
        color_map[festival] = COLORS['accent']
    
    colors = [color_map.get(dt, COLORS['accent']) for dt in agg['Day Type']]
    
    fig = go.Figure(go.Bar(
        y=agg['Day Type'],
        x=agg['Avg Price'],
        orientation='h',
        marker_color=colors,
        text=[f'₹{p:,.0f}' for p in agg['Avg Price']],
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    fig.update_layout(**base_layout(280, "Average Price by Day Type"))
    fig.update_layout(margin={"l": 100, "r": 60, "t": 40, "b": 40})
    fig.update_xaxes(title="Price (INR)")
    
    return fig


def chart_timing_impact(df):
    """Days to departure impact on pricing."""
    if df.empty or 'days_ahead' not in df.columns:
        return None
    
    agg = df.groupby('days_ahead').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    agg = agg[agg['days_ahead'] >= 0].sort_values('days_ahead')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=agg['days_ahead'],
        y=agg['base_price'],
        name='Avg Price',
        marker_color=COLORS['primary'],
        opacity=0.8
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=agg['days_ahead'],
        y=agg['occupancy'],
        name='Occupancy %',
        mode='lines+markers',
        line=dict(color=COLORS['accent'], width=3),
        marker=dict(size=7)
    ), secondary_y=True)
    
    layout = base_layout(280, "Price & Occupancy by Days to Departure")
    layout['legend'] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10))
    fig.update_layout(**layout)
    fig.update_xaxes(title="Days Ahead")
    fig.update_yaxes(title="Price (INR)", secondary_y=False)
    fig.update_yaxes(title="Occupancy %", secondary_y=True)
    
    return fig


def chart_route_performance(df):
    """Route occupancy ranking."""
    if df.empty:
        return None
    
    agg = df.groupby('route').agg({
        'occupancy': 'mean',
        'base_price': 'mean',
        'bus_id': 'count'
    }).reset_index()
    agg.columns = ['Route', 'Occupancy', 'Avg Price', 'Trips']
    agg = agg.sort_values('Occupancy', ascending=True).tail(10)
    
    # Color based on occupancy
    colors = [COLORS['danger'] if o < 50 else COLORS['accent'] if o < 70 else COLORS['secondary'] for o in agg['Occupancy']]
    
    fig = go.Figure(go.Bar(
        y=agg['Route'],
        x=agg['Occupancy'],
        orientation='h',
        marker_color=colors,
        text=[f'{o:.1f}%' for o in agg['Occupancy']],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(**base_layout(320, "Route Occupancy Performance"))
    fig.update_layout(margin={"l": 150, "r": 50, "t": 40, "b": 40})
    fig.update_xaxes(title="Occupancy %", range=[0, 100])
    
    return fig


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Load data
    df, err = fetch_buses(3000)
    df = preprocess(df)
    
    if df.empty:
        st.error(f"No data available: {err}")
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Dynamic Pricing Dashboard</h1>
        <p>Vignesh TAT | Real-time bus pricing intelligence and analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Average Price", f"₹{df['base_price'].mean():,.0f}")
    with col2:
        st.metric("Fleet Occupancy", f"{df['occupancy'].mean():.1f}%")
    with col3:
        st.metric("Routes", f"{df['route'].nunique()}")
    with col4:
        st.metric("Total Trips", f"{len(df):,}")
    with col5:
        # Weekend/Festival premium
        weekday = df[df['day_type'] == 'Weekday']['base_price'].mean()
        special = df[df['day_type'] != 'Weekday']['base_price'].mean()
        if weekday > 0 and not pd.isna(special):
            premium = ((special / weekday) - 1) * 100
            st.metric("Weekend/Festival", f"{premium:+.1f}%")
        else:
            st.metric("Weekend/Festival", "--")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["Price Analysis", "Seat Tracking", "Route Insights"])
    
    # ==================== TAB 1: PRICE ANALYSIS ====================
    with tab1:
        st.markdown('<div class="section-header">Price Trend Analysis</div>', unsafe_allow_html=True)
        
        # Get buses with history
        buses_with_history = fetch_buses_with_history()
        
        if buses_with_history.empty:
            st.info("Collecting price history data. Charts will appear after multiple hourly scrapes.")
        else:
            # Route filter
            routes = buses_with_history.apply(lambda r: f"{r['from_city']} > {r['to_city']}", axis=1).unique().tolist()
            selected_routes = st.multiselect("Select routes to compare:", routes, default=routes[:3], key="route_filter")
            
            if selected_routes:
                # Filter buses
                mask = buses_with_history.apply(lambda r: f"{r['from_city']} > {r['to_city']}" in selected_routes, axis=1)
                filtered_buses = buses_with_history[mask]
                
                # Fetch all history for selected buses
                all_history = pd.DataFrame()
                for bus_id in filtered_buses['bus_id'].head(6):
                    h = fetch_price_history(bus_id)
                    if not h.empty:
                        all_history = pd.concat([all_history, h])
                
                if not all_history.empty:
                    fig = chart_price_trends(all_history, filtered_buses)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Best and worst performers
                    col1, col2 = st.columns(2)
                    with col1:
                        best = filtered_buses.loc[filtered_buses['price_variance'].idxmax()] if len(filtered_buses) > 0 else None
                        if best is not None:
                            st.markdown("**Highest Price Variance**")
                            st.markdown(f"<span style='font-size: 1.2rem; color: {COLORS['secondary']}'>{best['from_city']} > {best['to_city']}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='badge-up'>₹{best['price_variance']:,.0f} variance</span>", unsafe_allow_html=True)
                    
                    with col2:
                        stable = filtered_buses.loc[filtered_buses['price_variance'].idxmin()] if len(filtered_buses) > 0 else None
                        if stable is not None:
                            st.markdown("**Most Stable Pricing**")
                            st.markdown(f"<span style='font-size: 1.2rem; color: {COLORS['primary']}'>{stable['from_city']} > {stable['to_city']}</span>", unsafe_allow_html=True)
                            st.markdown(f"<span class='badge-neutral'>₹{stable['price_variance']:,.0f} variance</span>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Day type and timing analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = chart_day_comparison(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = chart_timing_impact(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 2: SEAT TRACKING ====================
    with tab2:
        st.markdown('<div class="section-header">Seat-Level Price Analysis</div>', unsafe_allow_html=True)
        
        if buses_with_history.empty:
            st.info("Waiting for seat-level data...")
        else:
            # Bus selector
            bus_options = [f"{r['from_city']} > {r['to_city']} | {r['departure_time']}" for _, r in buses_with_history.head(10).iterrows()]
            selected_bus = st.selectbox("Select bus:", bus_options, key="seat_bus_select")
            
            if selected_bus:
                idx = bus_options.index(selected_bus)
                bus_id = buses_with_history.iloc[idx]['bus_id']
                bus_info = buses_with_history.iloc[idx]
                
                # Fetch seat history
                seat_history = fetch_seat_history(bus_id)
                
                if seat_history.empty:
                    st.info("No seat-level data for this bus yet.")
                else:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = chart_seat_prices(seat_history)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Seat Summary**")
                        seat_history['scraped_at'] = pd.to_datetime(seat_history['scraped_at'])
                        
                        summary_data = []
                        for seat_id in seat_history['seat_id'].unique()[:8]:
                            data = seat_history[seat_history['seat_id'] == seat_id].sort_values('scraped_at')
                            if len(data) >= 1:
                                first = data.iloc[0]['price']
                                last = data.iloc[-1]['price']
                                change = last - first if len(data) > 1 else 0
                                is_window = "Window" if data.iloc[0]['is_window'] else "Aisle"
                                summary_data.append({
                                    'Seat': seat_id,
                                    'Type': is_window,
                                    'Current': f"₹{last:,}",
                                    'Change': f"+₹{change}" if change > 0 else f"₹{change}" if change < 0 else "--"
                                })
                        
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ==================== TAB 3: ROUTE INSIGHTS ====================
    with tab3:
        st.markdown('<div class="section-header">Route Performance</div>', unsafe_allow_html=True)
        
        fig = chart_route_performance(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== DATA TABLE WITH FILTERS ====================
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    
    # Filters row
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        route_options = ['All Routes'] + sorted(df['route'].unique().tolist())
        selected_route_filter = st.multiselect("Routes", route_options, default=['All Routes'], key="table_route")
    
    with f2:
        coach_options = ['All Types'] + sorted(df['bus_type'].dropna().unique().tolist())
        selected_coach = st.multiselect("Coach Type", coach_options, default=['All Types'], key="table_coach")
    
    with f3:
        day_options = ['All Days'] + sorted(df['day_type'].unique().tolist())
        selected_day = st.multiselect("Day Type", day_options, default=['All Days'], key="table_day")
    
    with f4:
        min_occ, max_occ = st.slider("Occupancy %", 0, 100, (0, 100), key="table_occ")
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'All Routes' not in selected_route_filter and selected_route_filter:
        filtered_df = filtered_df[filtered_df['route'].isin(selected_route_filter)]
    
    if 'All Types' not in selected_coach and selected_coach:
        filtered_df = filtered_df[filtered_df['bus_type'].isin(selected_coach)]
    
    if 'All Days' not in selected_day and selected_day:
        filtered_df = filtered_df[filtered_df['day_type'].isin(selected_day)]
    
    filtered_df = filtered_df[(filtered_df['occupancy'] >= min_occ) & (filtered_df['occupancy'] <= max_occ)]
    
    # Display table
    display_cols = ['travel_date', 'route', 'bus_type', 'departure_time', 'day_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy']
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
    
    st.dataframe(
        filtered_df[display_cols].sort_values('travel_date', ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "travel_date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "route": "Route",
            "bus_type": "Coach",
            "departure_time": "Departure",
            "day_type": "Day Type",
            "base_price": st.column_config.NumberColumn("Price", format="₹%d"),
            "available_seats": "Available",
            "sold_seats": "Sold",
            "occupancy": st.column_config.ProgressColumn("Occupancy", min_value=0, max_value=100, format="%.1f%%")
        }
    )
    
    # Export buttons
    col1, col2, col3 = st.columns([1, 1, 6])
    
    with col1:
        csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv_data,
            f"vignesh_tat_pricing_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_data = to_excel(filtered_df[display_cols])
        st.download_button(
            "Download Excel",
            excel_data,
            f"vignesh_tat_pricing_{datetime.now().strftime('%Y%m%d')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
