"""
🚌 Dynamic Pricing Dashboard - Demand & Seat Intelligence
=========================================================
Focus: Seat-level pricing, Demand forecasting, and Market trends.
Designed for: Pricing Analysts & ML Model Training.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
# 🎨 CONFIGURATION & CSS
# ==============================================================================

st.set_page_config(
    page_title="Seat Demand Intelligence",
    page_icon="💺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Professional Dark Theme */
    .stApp {
        background: radial-gradient(circle at top left, #1a1c23, #0e1117);
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background-color: #1e232e;
        border: 1px solid #2d3342;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6e6e6;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 🔄 DATA LOADING
# ==============================================================================

@st.cache_data(ttl=600)
def load_data():
    """Load latest data from DB or fallback to file/demo."""
    if DB_CONNECTED:
        try:
            # INCREASED LIMIT TO 10,000 to ensure full fleet visibility
            raw_data = get_latest_data(limit=10000)
            if raw_data:
                return pd.DataFrame(raw_data)
        except:
            pass
            
    # If no DB, generate demo data
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    data = []
    routes = [("Chennai", "Tirunelveli"), ("Tirunelveli", "Chennai"), ("Chennai", "Madurai")]
    
    for d in dates:
        for f, t in routes:
            price = np.random.randint(800, 1500)
            data.append({
                "travel_date": d,
                "from_city": f,
                "to_city": t,
                "base_price": price,
                "available_seats": np.random.randint(5, 30),
                "sold_seats": np.random.randint(10, 35),
                "bus_type": np.random.choice(["Seater", "Sleeper"]),
                "operator": "Vignesh Tat"
            })
    return pd.DataFrame(data)

# ==============================================================================
# 🖥️ MAIN DASHBOARD
# ==============================================================================

def main():
    # Header
    st.markdown("# 💺 <span class='gradient-text'>Seat Demand Intelligence</span>", unsafe_allow_html=True)
    st.markdown("Dynamic Pricing & Occupancy Prediction Engine")
    st.markdown("---")
    
    df = load_data()
    
    # Preprocessing
    if not df.empty:
        df['route'] = df['from_city'] + ' → ' + df['to_city']
        df['total_seats'] = df['sold_seats'] + df['available_seats']
        df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100)
        df['day_name'] = pd.to_datetime(df['travel_date']).dt.day_name()
    
    # 1. KPI Row (Focus on Demand, NOT Revenue)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scraped Buses", f"{len(df)}", "Live Data")
    c2.metric("Avg Seat Price", f"₹{df['base_price'].mean():.0f}", "+2%")
    c3.metric("Avg Occupancy", f"{df['occupancy'].mean():.1f}%", "Demand Strength")
    c4.metric("Sold Seats", f"{df['sold_seats'].sum():,}", "Volume")
    
    st.markdown("---")
    
    # 2. Charts Row 1: The "Sciative" View (Price vs Demand)
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Demand Curve: Price vs Days-to-Departure")
        # How does price change as we get closer to the date?
        df_sorted = df.sort_values('travel_date')
        fig = px.scatter(
            df_sorted, 
            x='travel_date', 
            y='base_price', 
            color='occupancy',
            size='sold_seats',
            hover_data=['route', 'bus_type'],
            color_continuous_scale='RdYlGn', # Red (Low Occ) -> Green (High Occ)
            title="Price Elasticity (Color = Occupancy %)"
        )
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("🗓️ Day-of-Week Effect")
        # Which days command higher prices?
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg = df.groupby('day_name')['base_price'].mean().reindex(day_order)
        
        fig2 = px.bar(
            x=day_avg.index, 
            y=day_avg.values,
            color=day_avg.values,
            color_continuous_scale='Viridis',
            title="Avg Price by Day"
        )
        fig2.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # 3. Charts Row 2: Seat Composition
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💺 Seat Type Pricing Spread")
        # Box plot to show price range/variance for Seater vs Sleeper
        fig3 = px.box(
            df, 
            x='bus_type', 
            y='base_price', 
            color='bus_type',
            points="all",
            title="Price Variance by Seat Type"
        )
        fig3.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
    with c2:
        st.subheader("🛣️ Route Demand Strength")
        # Which routes fill up fastest?
        route_occ = df.groupby('route')['occupancy'].mean().sort_values()
        fig4 = px.bar(
            x=route_occ.values,
            y=route_occ.index,
            orientation='h',
            title="Avg Occupancy by Route"
        )
        fig4.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig4, use_container_width=True)
        
    # 4. Detailed Data Table (ML Training Data)
    st.markdown("### 🧬 ML Training Data (Seat Level)")
    st.info("Download this data to train your Dynamic Pricing algorithm (Sciative/Revmax style).")
    
    # Filter Controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_route = st.multiselect("Filter Routes", options=df['route'].unique(), default=df['route'].unique())
    with fc2:
        sel_type = st.multiselect("Filter Coach Type", options=df['bus_type'].unique(), default=df['bus_type'].unique())
    with fc3:
        min_occ = st.slider("Min Occupancy %", 0, 100, 0)
        
    # Apply Filters
    mask = (df['route'].isin(sel_route)) & (df['bus_type'].isin(sel_type)) & (df['occupancy'] >= min_occ)
    filtered_df = df[mask]
    
    # Show Table
    st.dataframe(
        filtered_df[['travel_date', 'day_name', 'route', 'bus_type', 'base_price', 'total_seats', 'sold_seats', 'occupancy']].sort_values('occupancy', ascending=False),
        use_container_width=True,
        height=500
    )
    
    # Export
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Training Data (CSV)",
        data=csv,
        file_name=f"seat_demand_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
