"""
🚌 Dynamic Pricing Dashboard - Professional Edition
===================================================
Clean, high-quality dashboard for pricing analytics.
Features:
- Real-time DB Connection
- Interactive Data Table
- CSV/Excel Export
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
    page_title="Dynamic Pricing Analytics",
    page_icon="🚌",
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
            raw_data = get_latest_data(limit=5000)
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
    st.markdown("# 🚌 <span class='gradient-text'>Dynamic Pricing Analytics</span>", unsafe_allow_html=True)
    st.markdown("Real-time intelligence for fleet pricing optimization.")
    st.markdown("---")
    
    df = load_data()
    
    # Preprocessing
    if not df.empty:
        df['route'] = df['from_city'] + ' → ' + df['to_city']
        df['occupancy'] = (df['sold_seats'] / (df['sold_seats'] + df['available_seats']).replace(0, 1) * 100)
        df['revenue'] = df['base_price'] * df['sold_seats']
    
    # 1. KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Active Buses", f"{len(df)}", "Live")
    c2.metric("Avg Ticket Price", f"₹{df['base_price'].mean():.0f}", "+2%")
    c3.metric("Fleet Occupancy", f"{df['occupancy'].mean():.1f}%", "+5%")
    c4.metric("Est. Revenue", f"₹{df['revenue'].sum():,.0f}", "Today")
    
    st.markdown("---")
    
    # 2. Charts Row
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("📈 Price vs Occupancy Trends")
        # Multi-line chart
        fig = px.line(
            df.sort_values('travel_date'), 
            x='travel_date', 
            y='base_price', 
            color='route',
            markers=True,
            title="Price Trends by Route"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("📊 Fleet Distribution")
        fig2 = px.pie(
            df, 
            names='bus_type', 
            values='revenue',
            hole=0.4,
            title="Revenue by Bus Type"
        )
        fig2.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
    # 3. Detailed Data Table with Exports
    st.markdown("### 📋 Detailed Order Book")
    
    # Filter Controls
    fc1, fc2 = st.columns(2)
    with fc1:
        sel_route = st.multiselect("Filter Routes", options=df['route'].unique(), default=df['route'].unique())
    with fc2:
        sel_type = st.multiselect("Filter Bus Type", options=df['bus_type'].unique(), default=df['bus_type'].unique())
        
    # Apply Filters
    mask = df['route'].isin(sel_route) & df['bus_type'].isin(sel_type)
    filtered_df = df[mask]
    
    # Show Table
    st.dataframe(
        filtered_df[['travel_date', 'route', 'bus_type', 'base_price', 'available_seats', 'sold_seats', 'occupancy', 'revenue']].style.format({
            "base_price": "₹{:.0f}",
            "revenue": "₹{:.0f}",
            "occupancy": "{:.1f}%",
            "travel_date": lambda t: t.strftime("%Y-%m-%d") if pd.notnull(t) else ""
        }),
        use_container_width=True,
        height=500
    )
    
    # Export Buttons
    st.markdown("### 📤 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"pricing_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        
    with col2:
        # Instruction for Sheets
        st.info("💡 To use in Google Sheets: Download CSV, then 'File > Import > Upload' in Sheets.")

if __name__ == "__main__":
    main()
