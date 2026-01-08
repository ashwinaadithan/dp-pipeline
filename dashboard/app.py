"""
Dynamic Pricing Dashboard
=========================
Beautiful Streamlit dashboard for visualizing bus pricing data.
Deploy FREE on Streamlit Cloud: https://streamlit.io/cloud
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Page config
st.set_page_config(
    page_title="🚌 Dynamic Pricing Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def load_data_from_files(data_folder="data"):
    """Load latest data from JSON files."""
    all_buses = []
    
    if not os.path.exists(data_folder):
        return pd.DataFrame()
    
    # Find all JSON files
    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    
    for json_file in sorted(json_files, reverse=True)[:10]:  # Last 10 files
        try:
            with open(os.path.join(data_folder, json_file), 'r') as f:
                data = json.load(f)
            
            for session in data.get("scrape_sessions", []):
                for bus in session.get("buses", []):
                    route = bus.get("route", {})
                    seats = bus.get("seats", {})
                    
                    all_buses.append({
                        "bus_id": bus.get("bus_id"),
                        "operator": bus.get("operator"),
                        "bus_type": bus.get("bus_type"),
                        "from_city": route.get("from_city"),
                        "to_city": route.get("to_city"),
                        "travel_date": route.get("travel_date"),
                        "departure_time": bus.get("departure_time"),
                        "base_price": bus.get("base_price"),
                        "available_seats": seats.get("available_seats", 0),
                        "sold_seats": seats.get("sold_seats", 0),
                        "total_seats": seats.get("total_seats", 0),
                        "min_price": seats.get("price_range", {}).get("min"),
                        "max_price": seats.get("price_range", {}).get("max"),
                        "scraped_at": session.get("scraped_at")
                    })
        except:
            continue
    
    return pd.DataFrame(all_buses)


def load_data_from_db():
    """Load data from Neon database."""
    try:
        from database import get_latest_data
        data = get_latest_data(limit=5000)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()


def main():
    # Header
    st.markdown('<h1 class="main-header">🚌 Dynamic Pricing Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    data_source = st.sidebar.radio("Data Source", ["Local Files", "Database"])
    
    # Load data
    if data_source == "Database":
        df = load_data_from_db()
    else:
        df = load_data_from_files("../data")
    
    if df.empty:
        st.warning("📭 No data found. Run the scraper first!")
        st.info("""
        **How to get data:**
        1. Run `python src/scraper.py`
        2. Wait for scraping to complete
        3. Refresh this dashboard
        """)
        return
    
    # Metrics Row
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🚌 Total Buses", len(df))
    with col2:
        routes = df[['from_city', 'to_city']].drop_duplicates()
        st.metric("🛤️ Routes", len(routes))
    with col3:
        avg_price = df['base_price'].mean()
        st.metric("💰 Avg Price", f"₹{avg_price:,.0f}" if avg_price else "N/A")
    with col4:
        total_available = df['available_seats'].sum()
        st.metric("🪑 Available Seats", f"{total_available:,}")
    with col5:
        total_sold = df['sold_seats'].sum()
        total = total_available + total_sold
        occupancy = (total_sold / total * 100) if total > 0 else 0
        st.metric("📊 Occupancy", f"{occupancy:.1f}%")
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Price Distribution by Route")
        if 'from_city' in df.columns and 'base_price' in df.columns:
            df['route'] = df['from_city'] + ' → ' + df['to_city']
            fig = px.box(
                df[df['base_price'].notna()], 
                x='route', 
                y='base_price',
                color='route',
                title="Price Range per Route"
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Occupancy by Route")
        if 'from_city' in df.columns:
            route_stats = df.groupby(['from_city', 'to_city']).agg({
                'available_seats': 'sum',
                'sold_seats': 'sum'
            }).reset_index()
            route_stats['route'] = route_stats['from_city'] + ' → ' + route_stats['to_city']
            route_stats['occupancy'] = route_stats['sold_seats'] / (route_stats['available_seats'] + route_stats['sold_seats']) * 100
            
            fig = px.bar(
                route_stats, 
                x='route', 
                y='occupancy',
                color='occupancy',
                color_continuous_scale='RdYlGn_r',
                title="Occupancy Rate (%)"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚌 Bus Types")
        if 'bus_type' in df.columns:
            type_counts = df['bus_type'].value_counts()
            fig = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Distribution of Bus Types",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Price vs Occupancy")
        if 'base_price' in df.columns and 'sold_seats' in df.columns:
            df_valid = df[(df['base_price'].notna()) & (df['base_price'] > 0)]
            df_valid['total'] = df_valid['available_seats'] + df_valid['sold_seats']
            df_valid['occupancy'] = df_valid['sold_seats'] / df_valid['total'].replace(0, 1) * 100
            
            fig = px.scatter(
                df_valid,
                x='base_price',
                y='occupancy',
                color='route' if 'route' in df_valid.columns else None,
                title="Price-Demand Relationship",
                labels={'base_price': 'Price (₹)', 'occupancy': 'Occupancy (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data Table
    st.markdown("---")
    st.subheader("📋 Raw Data")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        if 'from_city' in df.columns:
            cities = ['All'] + sorted(df['from_city'].unique().tolist())
            selected_from = st.selectbox("From City", cities)
    with col2:
        if 'to_city' in df.columns:
            cities = ['All'] + sorted(df['to_city'].unique().tolist())
            selected_to = st.selectbox("To City", cities)
    
    # Filter data
    filtered_df = df.copy()
    if selected_from != 'All':
        filtered_df = filtered_df[filtered_df['from_city'] == selected_from]
    if selected_to != 'All':
        filtered_df = filtered_df[filtered_df['to_city'] == selected_to]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "bus_data.csv",
            "text/csv"
        )
    with col2:
        st.info("💡 Tip: Deploy this dashboard FREE on [Streamlit Cloud](https://streamlit.io/cloud)")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Dynamic Pricing Dashboard | Data updated hourly</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
