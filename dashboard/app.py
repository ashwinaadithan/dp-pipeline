"""
🎯 Dynamic Pricing Intelligence Dashboard
==========================================
Vignesh TAT - Seat-Level Price Tracking
Focus: Price Fluctuations, Timing, Festivals
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
    """Find buses that have price changes in history."""
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
# STYLING
# ==============================================================================

st.set_page_config(page_title="DP Intelligence", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background: #0a0e14; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.5rem !important; }
    
    .header { display: flex; align-items: center; gap: 12px; margin-bottom: 0.5rem; }
    .title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin: 0; }
    .badge { background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155; border-radius: 12px; padding: 0.8rem 1rem;
    }
    div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.7rem !important; text-transform: uppercase; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.4rem !important; font-weight: 600 !important; }
    
    .section { color: #94a3b8; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 1rem 0 0.5rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.3rem; }
    hr { border: none; border-top: 1px solid #1e293b; margin: 0.8rem 0; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .price-up { color: #22c55e; font-weight: 600; }
    .price-down { color: #ef4444; font-weight: 600; }
    .festival { background: linear-gradient(135deg, #f59e0b, #ef4444); color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
    .weekend { background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# HELPERS
# ==============================================================================

COLORS = {'blue': '#3b82f6', 'pink': '#ec4899', 'green': '#22c55e', 'yellow': '#eab308', 'red': '#ef4444', 'purple': '#a855f7'}

# Tamil Nadu festivals 2026
FESTIVALS = {
    (1, 13): "Bhogi", (1, 14): "Pongal", (1, 15): "Mattu Pongal", (1, 16): "Kaanum Pongal",
    (1, 26): "Republic Day", (8, 15): "Independence Day", (10, 2): "Gandhi Jayanti",
    (10, 12): "Dussehra", (11, 1): "Deepavali"
}

def get_day_type(date):
    """Get day type: Festival, Weekend, or Weekday."""
    if isinstance(date, str):
        date = pd.to_datetime(date)
    key = (date.month, date.day)
    if key in FESTIVALS:
        return f"🎉 {FESTIVALS[key]}"
    elif date.weekday() >= 5:
        return "📅 Weekend"
    else:
        return "Weekday"


def layout(h=300, title=""):
    return {
        "template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter", "color": "#e2e8f0", "size": 11},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 35}, "height": h,
        "title": {"text": title, "font": {"size": 12, "color": "#f1f5f9"}, "x": 0.01}
    }


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
# CHARTS
# ==============================================================================

def chart_price_timeline(history_df, bus_info):
    """Price changes over time for a single bus."""
    if history_df.empty:
        return None
    
    fig = go.Figure()
    
    history_df['scraped_at'] = pd.to_datetime(history_df['scraped_at'])
    history_df = history_df.sort_values('scraped_at')
    
    # Price line
    fig.add_trace(go.Scatter(
        x=history_df['scraped_at'], y=history_df['base_price'],
        mode='lines+markers', name='Price',
        line=dict(color=COLORS['blue'], width=3),
        marker=dict(size=8, color=COLORS['blue']),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.1)'
    ))
    
    # Add annotations for price changes
    prices = history_df['base_price'].tolist()
    times = history_df['scraped_at'].tolist()
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        if change != 0:
            color = COLORS['green'] if change > 0 else COLORS['red']
            fig.add_annotation(
                x=times[i], y=prices[i],
                text=f"{'+' if change > 0 else ''}{change}",
                showarrow=True, arrowhead=2, arrowcolor=color,
                font=dict(color=color, size=10),
                bgcolor='rgba(0,0,0,0.7)', borderpad=3
            )
    
    route = f"{bus_info.get('from_city', '')} → {bus_info.get('to_city', '')}"
    title = f"Price Timeline: {route} | {bus_info.get('departure_time', '')}"
    
    fig.update_layout(**layout(280, title))
    fig.update_xaxes(title="Time", gridcolor="#1e293b")
    fig.update_yaxes(title="Price ₹", gridcolor="#1e293b")
    
    return fig


def chart_seat_comparison(seat_df):
    """Compare seat prices over time."""
    if seat_df.empty:
        return None
    
    seat_df['scraped_at'] = pd.to_datetime(seat_df['scraped_at'])
    
    # Get unique seats
    seats = seat_df['seat_id'].unique()[:10]  # Limit to 10 seats
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, seat in enumerate(seats):
        seat_data = seat_df[seat_df['seat_id'] == seat].sort_values('scraped_at')
        is_window = seat_data['is_window'].iloc[0] if not seat_data.empty else False
        
        fig.add_trace(go.Scatter(
            x=seat_data['scraped_at'], y=seat_data['price'],
            mode='lines+markers', name=f"{seat} {'🪟' if is_window else ''}",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(**layout(320, "Seat-Level Price Changes"))
    fig.update_layout(legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=9)))
    fig.update_xaxes(title="Time", gridcolor="#1e293b")
    fig.update_yaxes(title="Price ₹", gridcolor="#1e293b")
    
    return fig


def chart_day_type_comparison(df):
    """Compare prices by day type: Festival, Weekend, Weekday."""
    if df.empty:
        return None
    
    # Group by day type
    groups = df.groupby('day_type').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    
    # Sort: Festival first, then Weekend, then Weekday
    order = {'🎉': 0, '📅': 1, 'Weekday': 2}
    groups['sort_key'] = groups['day_type'].apply(lambda x: min([order.get(k, 2) for k in order.keys() if k in x]))
    groups = groups.sort_values('sort_key')
    
    colors_map = []
    for dt in groups['day_type']:
        if '🎉' in dt:
            colors_map.append(COLORS['yellow'])
        elif 'Weekend' in dt:
            colors_map.append(COLORS['purple'])
        else:
            colors_map.append(COLORS['blue'])
    
    fig = go.Figure(go.Bar(
        x=groups['day_type'], y=groups['base_price'],
        marker_color=colors_map,
        text=[f'₹{p:.0f}' for p in groups['base_price']], 
        textposition='outside', textfont=dict(color='#e2e8f0', size=11)
    ))
    
    fig.update_layout(**layout(280, "Price by Day Type"))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Avg Price ₹", gridcolor="#1e293b")
    
    return fig


def chart_timing_analysis(df):
    """Price by days to departure."""
    if df.empty or 'days_to_dep' not in df.columns:
        return None
    
    agg = df.groupby('days_to_dep').agg({
        'base_price': 'mean',
        'occupancy': 'mean'
    }).reset_index()
    agg = agg[agg['days_to_dep'] >= 0].sort_values('days_to_dep', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(
        x=agg['days_to_dep'], y=agg['base_price'], name='Price',
        marker_color=COLORS['blue'], opacity=0.8
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=agg['days_to_dep'], y=agg['occupancy'], name='Occupancy %',
        mode='lines+markers', line=dict(color=COLORS['pink'], width=3),
        marker=dict(size=8)
    ), secondary_y=True)
    
    fig.update_layout(**layout(280, "Price vs Days to Departure"))
    fig.update_layout(legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=9)))
    fig.update_xaxes(title="Days Until Departure", autorange="reversed", gridcolor="#1e293b")
    fig.update_yaxes(title="Price ₹", secondary_y=False, gridcolor="#1e293b")
    fig.update_yaxes(title="Occupancy %", secondary_y=True, gridcolor="#1e293b")
    
    return fig


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    # Load data
    df, err = fetch_buses(2000)
    df = preprocess(df)
    
    if df.empty:
        st.error(f"No data: {err}")
        return
    
    # Header
    st.markdown(f"""
        <div class="header">
            <h1 class="title">🎯 Dynamic Pricing Intelligence</h1>
            <span class="badge">Vignesh TAT</span>
        </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Price", f"₹{df['base_price'].mean():,.0f}")
    k2.metric("Occupancy", f"{df['occupancy'].mean():.1f}%")
    k3.metric("Routes", f"{df['route'].nunique()}")
    
    # Festival/Weekend premium
    weekday_price = df[~df['is_weekend'] & ~df['is_festival']]['base_price'].mean()
    special_price = df[df['is_weekend'] | df['is_festival']]['base_price'].mean()
    if weekday_price > 0 and not pd.isna(special_price):
        premium = ((special_price / weekday_price) - 1) * 100
        k4.metric("Festival/Weekend ↑", f"{premium:+.1f}%")
    else:
        k4.metric("Festival/Weekend ↑", "—")
    
    st.markdown("---")
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price Fluctuations", "🪑 Seat Tracking", "📊 Analysis"])
    
    # ===== TAB 1: PRICE FLUCTUATIONS =====
    with tab1:
        st.markdown("<div class='section'>Bus Price History</div>", unsafe_allow_html=True)
        
        # Get buses with price history
        buses_tracked = fetch_buses_with_changes()
        
        if buses_tracked.empty:
            st.info("🔄 Waiting for more hourly data to show price fluctuations...")
            st.caption("Data will appear after 2+ hourly scrapes of the same bus")
        else:
            # Bus selector
            bus_options = []
            for _, row in buses_tracked.head(20).iterrows():
                route = f"{row['from_city']} → {row['to_city']}"
                label = f"{route} | {row.get('departure_time', '?')} | {row.get('travel_date', '')}"
                if row.get('price_change', 0) > 0:
                    label += f" | Δ₹{row['price_change']}"
                bus_options.append((row['bus_id'], label))
            
            if bus_options:
                selected_label = st.selectbox(
                    "Select a bus to view price history:",
                    [opt[1] for opt in bus_options],
                    label_visibility="collapsed"
                )
                selected_bus_id = next(opt[0] for opt in bus_options if opt[1] == selected_label)
                
                # Fetch and display history
                history = fetch_price_history(selected_bus_id)
                bus_info = buses_tracked[buses_tracked['bus_id'] == selected_bus_id].iloc[0].to_dict()
                
                if not history.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = chart_price_timeline(history, bus_info)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Price change summary
                        st.markdown("#### Price Changes")
                        history['scraped_at'] = pd.to_datetime(history['scraped_at'])
                        history = history.sort_values('scraped_at')
                        
                        prices = history['base_price'].tolist()
                        times = history['scraped_at'].tolist()
                        
                        for i in range(len(prices) - 1, 0, -1):
                            change = prices[i] - prices[i-1]
                            time_str = times[i].strftime('%H:%M')
                            
                            if change > 0:
                                st.markdown(f"<div class='price-up'>⬆ ₹{change} at {time_str}</div>", unsafe_allow_html=True)
                            elif change < 0:
                                st.markdown(f"<div class='price-down'>⬇ ₹{abs(change)} at {time_str}</div>", unsafe_allow_html=True)
                        
                        # Day type
                        travel_date = bus_info.get('travel_date')
                        if travel_date:
                            day_type = get_day_type(travel_date)
                            if '🎉' in day_type:
                                st.markdown(f"<span class='festival'>{day_type}</span>", unsafe_allow_html=True)
                            elif 'Weekend' in day_type:
                                st.markdown(f"<span class='weekend'>{day_type}</span>", unsafe_allow_html=True)
    
    # ===== TAB 2: SEAT TRACKING =====
    with tab2:
        st.markdown("<div class='section'>Seat-Level Price Tracking</div>", unsafe_allow_html=True)
        
        if buses_tracked.empty:
            st.info("🔄 Waiting for more data...")
        else:
            # Reuse bus selector
            selected_bus = st.selectbox(
                "Select bus for seat tracking:",
                [f"{row['from_city']} → {row['to_city']} | {row.get('departure_time', '')} | {row.get('travel_date', '')}" 
                 for _, row in buses_tracked.head(10).iterrows()],
                key="seat_bus_select",
                label_visibility="collapsed"
            )
            
            # Find bus_id
            idx = [f"{row['from_city']} → {row['to_city']} | {row.get('departure_time', '')} | {row.get('travel_date', '')}" 
                   for _, row in buses_tracked.head(10).iterrows()].index(selected_bus)
            bus_id = buses_tracked.iloc[idx]['bus_id']
            
            # Fetch seat history
            seat_history = fetch_seat_history(bus_id)
            
            if seat_history.empty:
                st.info("No seat-level history yet for this bus")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = chart_seat_comparison(seat_history)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Seat Price Summary")
                    
                    # Get latest and earliest prices per seat
                    seat_history['scraped_at'] = pd.to_datetime(seat_history['scraped_at'])
                    
                    for seat_id in seat_history['seat_id'].unique()[:8]:
                        seat_data = seat_history[seat_history['seat_id'] == seat_id].sort_values('scraped_at')
                        
                        if len(seat_data) >= 2:
                            first_price = seat_data.iloc[0]['price']
                            last_price = seat_data.iloc[-1]['price']
                            change = last_price - first_price
                            is_window = '🪟' if seat_data.iloc[0]['is_window'] else ''
                            
                            if change > 0:
                                st.markdown(f"**{seat_id}** {is_window}: ₹{first_price} → ₹{last_price} <span class='price-up'>(+{change})</span>", unsafe_allow_html=True)
                            elif change < 0:
                                st.markdown(f"**{seat_id}** {is_window}: ₹{first_price} → ₹{last_price} <span class='price-down'>({change})</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{seat_id}** {is_window}: ₹{last_price} (stable)")
    
    # ===== TAB 3: ANALYSIS =====
    with tab3:
        st.markdown("<div class='section'>Pricing Analysis</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = chart_day_type_comparison(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = chart_timing_analysis(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Data export
        st.markdown("<div class='section'>Export Data</div>", unsafe_allow_html=True)
        
        cols = ['travel_date', 'route', 'day_type', 'bus_type', 'departure_time', 'base_price', 'occupancy', 'days_to_dep']
        cols = [c for c in cols if c in df.columns]
        
        st.dataframe(df[cols].sort_values('travel_date'), use_container_width=True, height=300)
        
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            st.download_button("⬇️ CSV", df[cols].to_csv(index=False).encode(), "dp_data.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ Excel", to_excel(df[cols]), "dp_data.xlsx", 
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


if __name__ == "__main__":
    main()
