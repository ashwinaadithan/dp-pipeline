"""
🚌 Dynamic Pricing Intelligence - Premium Dashboard
===================================================
A "StockPeers" style financial dashboard for bus pricing.
Designed for high-frequency pricing decisions and ML training.
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

# Add src to path for database import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import database module
try:
    from database import get_dashboard_metrics, get_route_timeseries, get_latest_data
    DB_CONNECTED = True
except ImportError:
    DB_CONNECTED = False

# ==============================================================================
# 🎨 PREMIUM THEME CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "StockPeers" Look
st.markdown("""
<style>
    /* Dark Finance Theme */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Metrics Ticker */
    div[data-testid="metric-container"] {
        background-color: #1e2530;
        border: 1px solid #2b3342;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Chart Containers */
    .chart-box {
        background-color: #161b24;
        border: 1px solid #2b3342;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Input Fields */
    .stSelectbox, .stTextInput {
        background-color: #1e2530;
        color: white;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h1 {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 🔄 DATA LOADING
# ==============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(route_tuple=None):
    """Fetch real data from DB or generate demo data."""
    if DB_CONNECTED:
        try:
            # Route tuple is (from_city, to_city)
            f, t = route_tuple if route_tuple else (None, None)
            
            # Get time series
            raw_ts = get_route_timeseries(f, t, limit=2000)
            if raw_ts:
                df = pd.DataFrame(raw_ts)
                df['travel_date'] = pd.to_datetime(df['travel_date'])
                df['scraped_at'] = pd.to_datetime(df['scraped_at'])
                return df
        except Exception as e:
            st.error(f"DB Error: {e}")
    
    # Fallback to Demo Data
    return generate_demo_data()

def generate_demo_data():
    """Generate realistic training data if DB empty."""
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    data = []
    
    base_prices = {"Seater": 800, "Sleeper": 1200}
    
    for d in dates:
        # Create demand curve based on days to departure
        days_left = (d - datetime.now()).days
        demand_mult = 1.5 if days_left < 2 else (1.0 if days_left < 7 else 0.8)
        
        for b_type in ["Seater", "Sleeper"]:
            price = base_prices[b_type] * demand_mult * np.random.uniform(0.9, 1.1)
            occupancy = max(10, 100 - (price/20) + np.random.normal(0, 5))
            
            data.append({
                "travel_date": d,
                "scraped_at": datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                "base_price": int(price),
                "available_seats": int(40 * (1 - occupancy/100)),
                "sold_seats": int(40 * (occupancy/100)),
                "occupancy": occupancy,
                "bus_type": b_type
            })
            
    return pd.DataFrame(data)

# ==============================================================================
# 🖥️ MAIN DASHBOARD
# ==============================================================================

def main():
    # 1. Header & Filters
    col_logo, col_search = st.columns([1, 4])
    with col_logo:
        st.title("PriceAI")
        st.caption("v2.1 Premium")
        
    with col_search:
        # Route Selector (The "Stock Ticker")
        # Hardcoded common routes for now (in production, fetch unique routes from DB)
        selected_route_str = st.selectbox(
            "🔍 Search Route (Ticker)",
            ["Chennai → Tirunelveli", "Tirunelveli → Chennai", "Chennai → Madurai", "Madurai → Chennai"],
            index=0
        )
        
    # Parse route
    if "→" in selected_route_str:
        f_city, t_city = selected_route_str.split(" → ")
        df = fetch_data((f_city, t_city))
    else:
        df = fetch_data()

    # 2. Key Metrics Ticker
    # Calculate changes vs yesterday
    latest_price = df['base_price'].iloc[-1]
    avg_price_7d = df['base_price'].mean()
    price_delta = ((latest_price - avg_price_7d) / avg_price_7d) * 100
    
    latest_occ = df['occupancy'].iloc[-1]
    avg_occ_7d = df['occupancy'].mean()
    occ_delta = latest_occ - avg_occ_7d
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"₹{latest_price}", f"{price_delta:.1f}%")
    with col2:
        st.metric("Occupancy", f"{latest_occ:.1f}%", f"{occ_delta:.1f}%")
    with col3:
        # Estimated Revenue (Price * Sold seats)
        rev = latest_price * df['sold_seats'].iloc[-1]
        st.metric("Est. Revenue (vBus)", f"₹{rev:,.0f}", "Live")
    with col4:
        st.metric("Market Position", "Premium", "Top 10%")

    st.markdown("---")

    # 3. Main Chart: The "Demand Curve"
    # Overlays Price (Line) on top of Demand (Area)
    st.markdown("### 📈 Price vs Volume Analysis")
    
    fig_main = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Area: Demand (Occupancy)
    fig_main.add_trace(
        go.Scatter(
            x=df['travel_date'], y=df['occupancy'],
            name="Occupancy %",
            fill='tozeroy',
            line=dict(width=0),
            marker=dict(color='rgba(139, 92, 246, 0.2)'), # Purple faint
        ),
        secondary_y=True
    )
    
    # Line: Price
    fig_main.add_trace(
        go.Scatter(
            x=df['travel_date'], y=df['base_price'],
            name="Our Price",
            mode='lines+markers',
            line=dict(color='#00f2fe', width=3), # Cyan
            marker=dict(size=6, color='#00f2fe', line=dict(width=2, color='white'))
        ),
        secondary_y=False
    )
    
    fig_main.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=1.02, x=1),
        hovermode="x unified"
    )
    
    # Axis styling
    fig_main.update_yaxes(title_text="Price (₹)", gridcolor='rgba(255,255,255,0.1)', secondary_y=False)
    fig_main.update_yaxes(title_text="Occupancy %", showgrid=False, secondary_y=True)
    fig_main.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig_main, use_container_width=True)

    # 4. Small Multiples (Financial Analysis)
    st.markdown("### 🧠 ML Training Insights")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**🔬 Elasticity Scatter (Price Sensitivity)**")
        # Does higher price = lower sales?
        fig_scatter = px.scatter(
            df, x="base_price", y="occupancy",
            color="bus_type",
            trendline="ols", # Add regression line
            color_discrete_sequence=['#3b82f6', '#ec4899']
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20,20,30,0.5)',
            xaxis_title="Price Point (₹)",
            yaxis_title="Occupancy achieved (%)",
            height=350
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c2:
        st.markdown("**📊 Yield Heatmap (Day of Week)**")
        # Revenue per seat by day
        df['day_name'] = df['travel_date'].dt.day_name()
        df['revenue'] = df['base_price'] * df['sold_seats']
        
        # Aggregate
        heatmap_data = df.groupby('day_name')['revenue'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        
        fig_heat = px.bar(
            heatmap_data, x='day_name', y='revenue',
            color='revenue',
            color_continuous_scale='Viridis'
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="",
            yaxis_title="Avg Revenue (₹)",
            height=350
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # 5. Raw Data Table (Financial Style)
    with st.expander("📋 View Order Book (Raw Data)", expanded=True):
        st.dataframe(
            df[['travel_date', 'bus_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy']].style.background_gradient(subset=['occupancy'], cmap='RdYlGn'),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
