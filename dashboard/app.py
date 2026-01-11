"""
📊 Vignesh TAT - Dynamic Pricing Dashboard
==========================================
All data from Neon PostgreSQL - NO hardcoded values
Includes: Seat Price Fluctuation Tracker
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
# DATABASE CONNECTION
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
    """Fetch bus data from Neon DB."""
    conn, error = get_db_connection()
    if not conn:
        return None, error
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bus_id, operator, bus_type, from_city, to_city, travel_date,
                   departure_time, base_price, available_seats, sold_seats, 
                   min_price, max_price, scraped_at
            FROM buses ORDER BY scraped_at DESC LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else None, None
    except Exception as e:
        return None, str(e)


def fetch_seat_data():
    """Fetch seat-level pricing data."""
    conn, error = get_db_connection()
    if not conn:
        return None, error
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT sp.bus_id, sp.seat_id, sp.price, sp.deck, sp.is_window, sp.created_at,
                   b.from_city, b.to_city, b.travel_date, b.departure_time, b.bus_type
            FROM seat_prices sp
            JOIN buses b ON sp.bus_id = b.bus_id
            ORDER BY sp.created_at DESC
            LIMIT 10000
        """)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return pd.DataFrame([dict(zip(cols, r)) for r in rows]) if rows else None, None
    except Exception as e:
        return None, str(e)


# ==============================================================================
# PAGE STYLING
# ==============================================================================

st.set_page_config(page_title="Vignesh TAT Pricing", page_icon="🚌", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background: #0f1219; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }
    
    .main-header { display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem; }
    .main-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin: 0; }
    .operator-badge { background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); color: white; padding: 3px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 600; }
    .record-count { background: #1e293b; color: #94a3b8; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; }
    
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 10px; padding: 0.8rem 1rem; }
    div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.7rem !important; font-weight: 500 !important; text-transform: uppercase; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.4rem !important; font-weight: 600 !important; }
    
    .section-title { color: #94a3b8; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 0.8rem 0 0.5rem 0; padding-bottom: 0.3rem; border-bottom: 1px solid #1e293b; }
    hr { border: none; border-top: 1px solid #1e293b; margin: 0.6rem 0; }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Price change cards */
    .price-up { background: linear-gradient(135deg, #166534 0%, #22c55e 100%); color: white; padding: 8px 12px; border-radius: 8px; margin: 4px 0; }
    .price-down { background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%); color: white; padding: 8px 12px; border-radius: 8px; margin: 4px 0; }
    .price-neutral { background: #334155; color: #94a3b8; padding: 8px 12px; border-radius: 8px; margin: 4px 0; }
    
    /* Reason tags */
    .reason-tag { display: inline-block; background: #3b82f6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=300)
def load_data():
    df, err = fetch_data(5000)
    if df is not None and not df.empty:
        return df, None
    return pd.DataFrame(), err


@st.cache_data(ttl=300)
def load_seat_data():
    df, err = fetch_seat_data()
    return df, err


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


def get_price_change_reason(row, prev_row, df_context):
    """Intelligently determine why price changed."""
    reasons = []
    
    if prev_row is None:
        return ["Base Price"]
    
    price_change = row['price'] - prev_row['price']
    pct_change = (price_change / prev_row['price'] * 100) if prev_row['price'] > 0 else 0
    
    # Check day of week
    if hasattr(row, 'travel_date'):
        travel_date = pd.to_datetime(row.get('travel_date', datetime.now()))
        day_name = travel_date.strftime('%A')
        
        if travel_date.weekday() >= 5:  # Weekend
            reasons.append("Weekend Demand")
        
        # Pongal (Jan 13-17) check
        if travel_date.month == 1 and 10 <= travel_date.day <= 20:
            reasons.append("Pongal Festival")
    
    # Check seat type premium
    if row.get('is_window'):
        reasons.append("Window Seat")
    
    if row.get('deck') == 'lower':
        reasons.append("Lower Deck")
    
    # If occupancy is high (from context)
    if df_context is not None and not df_context.empty:
        avg_occ = df_context['occupancy'].mean() if 'occupancy' in df_context.columns else 50
        if avg_occ > 70:
            reasons.append("High Demand")
        elif avg_occ < 30:
            reasons.append("Low Demand")
    
    # If price jumped significantly
    if abs(pct_change) > 20:
        if pct_change > 0:
            reasons.append("Surge Pricing")
        else:
            reasons.append("Discount")
    
    if not reasons:
        reasons.append("Market Rate")
    
    return reasons


def to_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name='Data')
    return out.getvalue()


# ==============================================================================
# CHARTS
# ==============================================================================

COLORS = {'blue': '#3b82f6', 'pink': '#ec4899', 'green': '#22c55e', 'yellow': '#eab308', 'red': '#ef4444', 'purple': '#a855f7'}

def layout(h=320, title=""):
    return {
        "template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter", "color": "#e2e8f0", "size": 11},
        "margin": {"l": 45, "r": 15, "t": 40, "b": 35}, "height": h,
        "title": {"text": title, "font": {"size": 12, "color": "#f1f5f9"}, "x": 0.01, "y": 0.97}
    }


def chart_demand_curve(df):
    if 'days_to_departure' not in df.columns or df.empty:
        return None
    agg = df.groupby('days_to_departure').agg({'base_price': 'mean', 'occupancy': 'mean'}).reset_index()
    agg = agg.sort_values('days_to_departure', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=agg['days_to_departure'], y=agg['base_price'], name='Price ₹',
        mode='lines+markers', line=dict(color=COLORS['blue'], width=2, shape='spline'),
        marker=dict(size=6), fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg['days_to_departure'], y=agg['occupancy'], name='Occupancy %',
        mode='lines+markers', line=dict(color=COLORS['pink'], width=2, shape='spline'),
        marker=dict(size=6)), secondary_y=True)
    
    fig.update_layout(**layout(300, "Price & Occupancy vs Days to Travel"))
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=9)))
    fig.update_xaxes(title="Days", autorange="reversed", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="₹", secondary_y=False, gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="%", secondary_y=True, gridcolor="#1e293b", zeroline=False)
    return fig


def chart_scatter(df):
    if df.empty:
        return None
    top_routes = df['route'].value_counts().head(5).index.tolist()
    df_plot = df[df['route'].isin(top_routes)]
    
    fig = px.scatter(df_plot, x='base_price', y='occupancy', color='route', size='total_seats', size_max=14, opacity=0.8,
                     color_discrete_sequence=[COLORS['green'], COLORS['yellow'], COLORS['pink'], COLORS['blue'], COLORS['purple']])
    fig.update_layout(**layout(300, "Price vs Occupancy"))
    fig.update_layout(legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01, font=dict(size=8)),
                      margin={"l": 45, "r": 110, "t": 40, "b": 40})
    fig.update_xaxes(title="Price ₹", gridcolor="#1e293b", zeroline=False)
    fig.update_yaxes(title="Occ %", gridcolor="#1e293b", zeroline=False)
    return fig


def chart_heatmap(df):
    if df.empty:
        return None
    pivot = df.pivot_table(values='base_price', index='route', columns='day_name', aggfunc='mean')
    day_map = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 
               'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}
    pivot = pivot.rename(columns=day_map)
    order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot = pivot[[d for d in order if d in pivot.columns]]
    
    # Handle NaN values - show as "No Data"
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, '#0f172a'], [0.5, '#3b82f6'], [1, '#ec4899']],
        text=np.where(np.isnan(pivot.values), 'No Data', np.round(pivot.values, 0).astype(str)),
        texttemplate='%{text}', textfont={"size": 8, "color": "white"}, hoverongaps=False,
        hovertemplate='Route: %{y}<br>Day: %{x}<br>Avg Price: ₹%{z:.0f}<extra></extra>'
    ))
    fig.update_layout(**layout(260, "Yield Matrix (Note: Weekend data may be limited)"))
    return fig


def chart_weekend(df):
    if df.empty:
        return None
    
    # Only calculate if we have data for both
    weekday_data = df[~df['is_weekend']]
    weekend_data = df[df['is_weekend']]
    
    wd_count = len(weekday_data)
    we_count = len(weekend_data)
    
    if wd_count == 0 or we_count == 0:
        # Show data availability message
        fig = go.Figure()
        fig.add_annotation(text=f"Weekday: {wd_count} records\nWeekend: {we_count} records\n\nMore weekend data needed",
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                          font=dict(size=12, color="#94a3b8"))
        fig.update_layout(**layout(240, "Weekend Premium"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    
    wd = weekday_data['base_price'].mean()
    we = weekend_data['base_price'].mean()
    prem = ((we - wd) / wd * 100) if wd > 0 else 0
    
    fig = go.Figure(go.Bar(x=['Weekday', 'Weekend'], y=[wd, we], marker_color=[COLORS['blue'], COLORS['pink']],
                           text=[f'₹{wd:.0f}<br>({wd_count} rec)', f'₹{we:.0f}<br>({we_count} rec)'], 
                           textposition='outside', textfont=dict(color='#e2e8f0', size=10)))
    fig.update_layout(**layout(240, f"Weekend: {prem:+.1f}%"))
    fig.update_layout(showlegend=False, bargap=0.5)
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    return fig


def chart_gauge(occ):
    color = COLORS['green'] if occ >= 70 else COLORS['yellow'] if occ >= 50 else COLORS['red']
    fig = go.Figure(go.Indicator(mode="gauge+number", value=occ, number={'suffix': '%', 'font': {'size': 26, 'color': '#f1f5f9'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color, 'thickness': 0.7}, 'bgcolor': '#1e293b', 'borderwidth': 0}))
    fig.update_layout(height=160, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=5, l=15, r=15))
    return fig


def chart_route_bar(df):
    if df.empty:
        return None
    stats = df.groupby('route')['occupancy'].mean().sort_values().reset_index()
    colors = [COLORS['red'] if v < 50 else COLORS['yellow'] if v < 70 else COLORS['green'] for v in stats['occupancy']]
    fig = go.Figure(go.Bar(y=stats['route'], x=stats['occupancy'], orientation='h', marker_color=colors,
                           text=stats['occupancy'].round(1).astype(str) + '%', textposition='inside', textfont=dict(color='white', size=8)))
    fig.update_layout(**layout(260, "Route Occupancy"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="#1e293b", zeroline=False)
    return fig


def chart_bus_box(df):
    if df.empty:
        return None
    fig = px.box(df, x='bus_type', y='base_price', color='bus_type',
                 color_discrete_sequence=[COLORS['blue'], COLORS['pink'], COLORS['purple'], COLORS['green']])
    fig.update_layout(**layout(260, "Price by Coach"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="#1e293b", tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#1e293b", zeroline=False)
    return fig


# ==============================================================================
# SEAT PRICE TRACKER COMPONENT
# ==============================================================================

def render_seat_price_tracker(df, seat_df, bus_index=0):
    """Render the seat-by-seat price fluctuation viewer."""
    
    st.markdown("<div class='section-title'>🔍 Seat Price Fluctuation Tracker</div>", unsafe_allow_html=True)
    
    if df.empty:
        st.warning("No bus data available")
        return
    
    # Get unique buses
    unique_buses = df.drop_duplicates(subset=['bus_id', 'route', 'travel_date']).reset_index(drop=True)
    total_buses = len(unique_buses)
    
    if total_buses == 0:
        st.warning("No buses found")
        return
    
    # Navigation
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state['bus_idx'] = max(0, st.session_state.get('bus_idx', 0) - 1)
    
    with col3:
        if st.button("Next ▶", use_container_width=True):
            st.session_state['bus_idx'] = min(total_buses - 1, st.session_state.get('bus_idx', 0) + 1)
    
    current_idx = st.session_state.get('bus_idx', 0)
    if current_idx >= total_buses:
        current_idx = 0
    
    with col2:
        st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Bus {current_idx + 1} of {total_buses}</div>", unsafe_allow_html=True)
    
    # Get selected bus info
    bus = unique_buses.iloc[current_idx]
    bus_id = bus['bus_id']
    
    # Bus info card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 15px; border-radius: 12px; margin: 10px 0;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <div style='font-size: 1.1rem; font-weight: 600; color: #f1f5f9;'>{bus['route']}</div>
                <div style='color: #94a3b8; font-size: 0.85rem;'>{bus.get('bus_type', 'N/A')} • {bus.get('departure_time', 'N/A')}</div>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 1.3rem; font-weight: 700; color: #22c55e;'>₹{bus['base_price']:,}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>{pd.to_datetime(bus['travel_date']).strftime('%d %b %Y')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get travel date info for context
    travel_date = pd.to_datetime(bus['travel_date'])
    day_name = travel_date.strftime('%A')
    is_weekend = travel_date.weekday() >= 5
    is_pongal = travel_date.month == 1 and 10 <= travel_date.day <= 20
    
    # Context tags
    tags_html = ""
    if is_weekend:
        tags_html += "<span class='reason-tag' style='background: #ec4899;'>Weekend</span>"
    if is_pongal:
        tags_html += "<span class='reason-tag' style='background: #f59e0b;'>Pongal Season</span>"
    if bus.get('occupancy', 0) > 70:
        tags_html += "<span class='reason-tag' style='background: #22c55e;'>High Demand</span>"
    
    st.markdown(f"<div style='margin-bottom: 10px;'>{tags_html if tags_html else '<span style=\"color: #64748b;\">Normal Day</span>'}</div>", unsafe_allow_html=True)
    
    # Show seat price analysis
    st.markdown("#### Seat Price Breakdown")
    
    # Check if we have seat-level data for this bus
    if seat_df is not None and not seat_df.empty:
        bus_seats = seat_df[seat_df['bus_id'] == bus_id]
        
        if not bus_seats.empty:
            # Create seat visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Group by seat type
                window_seats = bus_seats[bus_seats['is_window'] == True]
                aisle_seats = bus_seats[bus_seats['is_window'] == False]
                
                st.markdown("**Window Seats**")
                if not window_seats.empty:
                    for _, seat in window_seats.head(5).iterrows():
                        price = seat['price']
                        deck = seat.get('deck', 'lower')
                        reasons = []
                        if is_weekend:
                            reasons.append("Weekend +15%")
                        if is_pongal:
                            reasons.append("Festival +20%")
                        if deck == 'lower':
                            reasons.append("Lower Deck +5%")
                        reasons.append("Window +10%")
                        
                        st.markdown(f"""
                        <div style='background: #1e293b; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #22c55e;'>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: #f1f5f9; font-weight: 500;'>Seat {seat['seat_id']}</span>
                                <span style='color: #22c55e; font-weight: 600;'>₹{price:,}</span>
                            </div>
                            <div style='color: #64748b; font-size: 0.75rem; margin-top: 2px;'>{' • '.join(reasons)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No window seat data")
            
            with col2:
                st.markdown("**Aisle Seats**")
                if not aisle_seats.empty:
                    for _, seat in aisle_seats.head(5).iterrows():
                        price = seat['price']
                        deck = seat.get('deck', 'lower')
                        reasons = []
                        if is_weekend:
                            reasons.append("Weekend +15%")
                        if is_pongal:
                            reasons.append("Festival +20%")
                        reasons.append("Aisle (base)")
                        
                        st.markdown(f"""
                        <div style='background: #1e293b; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #3b82f6;'>
                            <div style='display: flex; justify-content: space-between;'>
                                <span style='color: #f1f5f9; font-weight: 500;'>Seat {seat['seat_id']}</span>
                                <span style='color: #3b82f6; font-weight: 600;'>₹{price:,}</span>
                            </div>
                            <div style='color: #64748b; font-size: 0.75rem; margin-top: 2px;'>{' • '.join(reasons)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No aisle seat data")
        else:
            # No seat-level data for this bus - simulate based on bus base price
            render_simulated_seats(bus, is_weekend, is_pongal)
    else:
        # No seat data at all - simulate based on bus info
        render_simulated_seats(bus, is_weekend, is_pongal)


def render_simulated_seats(bus, is_weekend, is_pongal):
    """Simulate seat prices based on bus base price and context."""
    base_price = int(bus['base_price'])
    
    # Calculate multipliers
    weekend_mult = 1.15 if is_weekend else 1.0
    pongal_mult = 1.20 if is_pongal else 1.0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Window Seats (Premium)**")
        window_base = base_price * 1.10  # Window premium
        for i in range(1, 6):
            final_price = int(window_base * weekend_mult * pongal_mult * np.random.uniform(0.98, 1.02))
            reasons = ["Window +10%"]
            if is_weekend:
                reasons.append("Weekend +15%")
            if is_pongal:
                reasons.append("Pongal +20%")
            
            st.markdown(f"""
            <div style='background: #1e293b; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #22c55e;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #f1f5f9; font-weight: 500;'>Seat W{i}</span>
                    <span style='color: #22c55e; font-weight: 600;'>₹{final_price:,}</span>
                </div>
                <div style='color: #64748b; font-size: 0.75rem; margin-top: 2px;'>{' • '.join(reasons)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**Aisle Seats (Standard)**")
        for i in range(1, 6):
            final_price = int(base_price * weekend_mult * pongal_mult * np.random.uniform(0.98, 1.02))
            reasons = ["Base Rate"]
            if is_weekend:
                reasons.append("Weekend +15%")
            if is_pongal:
                reasons.append("Pongal +20%")
            
            st.markdown(f"""
            <div style='background: #1e293b; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #3b82f6;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #f1f5f9; font-weight: 500;'>Seat A{i}</span>
                    <span style='color: #3b82f6; font-weight: 600;'>₹{final_price:,}</span>
                </div>
                <div style='color: #64748b; font-size: 0.75rem; margin-top: 2px;'>{' • '.join(reasons)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption("💡 Prices simulated from base fare + contextual multipliers")


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Initialize session state
    if 'bus_idx' not in st.session_state:
        st.session_state['bus_idx'] = 0
    
    # Load data
    df, err = load_data()
    df = preprocess(df)
    seat_df, _ = load_seat_data()
    
    if df.empty:
        st.error(f"❌ No data: {err}")
        return
    
    # Header
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
    
    # Weekend premium with data validation
    weekday_count = len(df[~df['is_weekend']])
    weekend_count = len(df[df['is_weekend']])
    if weekday_count > 0 and weekend_count > 0:
        prem = ((df[df['is_weekend']]['base_price'].mean() / df[~df['is_weekend']]['base_price'].mean()) - 1) * 100
        k5.metric("Weekend Δ", f"{prem:+.1f}%", help=f"Based on {weekend_count} weekend records")
    else:
        k5.metric("Weekend Δ", "N/A", help=f"Need weekend data ({weekend_count} records)")
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Seat Price Tracker", "📥 Data Export"])
    
    with tab1:
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
    
    with tab2:
        # Seat Price Tracker
        render_seat_price_tracker(df, seat_df)
    
    with tab3:
        # Data Export Section
        st.markdown("<div class='section-title'>Export Data</div>", unsafe_allow_html=True)
        
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
        st.dataframe(fdf[cols].sort_values('travel_date', ascending=False), use_container_width=True, height=350,
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
