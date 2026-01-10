"""
🚌 Dynamic Pricing Intelligence Dashboard
==========================================
Combines Sciative + RevMax style analytics:
- Demand Forecasting & Price Elasticity (Sciative)
- Yield Management & Occupancy Optimization (RevMax)
- ML-Ready Data Exports for Algorithm Training

Pipeline: Oracle Cron → Hourly Scrape → Neon DB → This Dashboard
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
DB_CONNECTED = False
DB_ERROR = None

try:
    from database import get_dashboard_metrics, get_route_timeseries, get_latest_data
    DB_CONNECTED = True
except ImportError as e:
    DB_ERROR = f"ImportError: {str(e)}"
except Exception as e:
    DB_ERROR = f"Error: {str(e)}"

# ==============================================================================
# 🎨 PAGE CONFIG & PREMIUM STYLING
# ==============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Intelligence | Vignesh TAT",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Premium Dark Theme with Gradient Accents */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #1a1f2e 50%, #0d1117 100%);
    }
    
    /* Glass Morphism Cards */
    div[data-testid="metric-container"] {
        background: rgba(30, 35, 50, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 120, 180, 0.2);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(100, 120, 200, 0.2);
    }
    
    /* Gradient Text Headers */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Data Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(22, 27, 34, 0.8);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(100, 120, 180, 0.15);
    }
    
    /* Pills/Tags */
    .status-live {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-demo {
        background: linear-gradient(135deg, #ff6b6b 0%, #ffa726 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1f2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #4facfe;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 🔄 DATA LOADING WITH ENHANCED ERROR HANDLING
# ==============================================================================

@st.cache_data(ttl=300)  # 5-minute cache for live data
def load_data():
    """Load latest data from Neon DB with detailed error reporting."""
    error_info = {"status": "unknown", "message": None, "details": None}
    
    if not DB_CONNECTED:
        error_info = {
            "status": "import_failed",
            "message": "Database module import failed",
            "details": DB_ERROR
        }
        return generate_demo_data(), error_info
    
    try:
        # Fetch up to 10,000 records for comprehensive analysis
        raw_data = get_latest_data(limit=10000)
        
        if raw_data and len(raw_data) > 0:
            error_info = {"status": "connected", "message": None, "details": None}
            return pd.DataFrame(raw_data), error_info
        else:
            error_info = {
                "status": "empty_data",
                "message": "Database connected but no data found",
                "details": "The 'buses' table might be empty. Run the scraper first."
            }
            return generate_demo_data(), error_info
            
    except Exception as e:
        error_info = {
            "status": "connection_failed",
            "message": "Database query failed",
            "details": str(e)
        }
        return generate_demo_data(), error_info


def generate_demo_data():
    """Generate realistic demo data for development/preview."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    data = []
    
    routes = [
        ("Chennai", "Tirunelveli"), 
        ("Tirunelveli", "Chennai"), 
        ("Chennai", "Madurai"),
        ("Madurai", "Chennai"),
        ("Chennai", "Coimbatore"),
        ("Coimbatore", "Chennai")
    ]
    
    bus_types = ["Multi-Axle Sleeper", "Volvo Sleeper", "Seater/Sleeper", "AC Sleeper"]
    
    for d in dates:
        # Determine if it's a weekend/holiday (higher demand)
        is_weekend = d.dayofweek >= 5
        is_holiday = d.day in [1, 15, 26]  # Sample holidays
        demand_factor = 1.0 + (0.3 if is_weekend else 0) + (0.4 if is_holiday else 0)
        
        for f, t in routes:
            # Base price varies by route distance (Chennai-Tirunelveli > Chennai-Madurai)
            base = 1200 if "Tirunelveli" in (f, t) else 900
            
            # Dynamic pricing based on demand
            price = int(base * demand_factor * np.random.uniform(0.9, 1.2))
            
            # Occupancy inversely related to price elasticity
            occ_base = 70 if is_weekend else 55
            available = np.random.randint(5, 25)
            sold = int((40 - available) * (demand_factor * 0.8))
            
            data.append({
                "bus_id": f"DEMO_{len(data)}",
                "travel_date": d,
                "from_city": f,
                "to_city": t,
                "base_price": price,
                "min_price": int(price * 0.85),
                "max_price": int(price * 1.15),
                "available_seats": available,
                "sold_seats": max(sold, 0),
                "bus_type": np.random.choice(bus_types),
                "operator": "Vignesh TAT",
                "departure_time": np.random.choice(["18:00", "20:00", "21:30", "22:00", "23:00"]),
                "scraped_at": datetime.now() - timedelta(hours=np.random.randint(0, 24))
            })
    
    return pd.DataFrame(data)


def preprocess_data(df):
    """Add computed columns for analysis."""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Basic derived columns
    df['route'] = df['from_city'] + ' → ' + df['to_city']
    df['total_seats'] = df['sold_seats'] + df['available_seats']
    df['occupancy'] = (df['sold_seats'] / df['total_seats'].replace(0, 1) * 100).round(1)
    
    # Time-based features for ML
    df['travel_date'] = pd.to_datetime(df['travel_date'])
    df['day_name'] = df['travel_date'].dt.day_name()
    df['day_of_week'] = df['travel_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['week_number'] = df['travel_date'].dt.isocalendar().week
    
    # Days until departure (from scraped_at)
    if 'scraped_at' in df.columns:
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        df['days_to_departure'] = (df['travel_date'] - df['scraped_at'].dt.normalize()).dt.days
    else:
        df['days_to_departure'] = 7  # default
    
    # Price bands for analysis
    df['price_band'] = pd.cut(df['base_price'], 
                               bins=[0, 800, 1000, 1200, 1500, 9999],
                               labels=['Economy', 'Standard', 'Premium', 'Luxury', 'Peak'])
    
    return df


# ==============================================================================
# 📊 VISUALIZATION COMPONENTS
# ==============================================================================

def create_price_elasticity_chart(df):
    """Sciative-style: Price vs Occupancy showing demand elasticity."""
    fig = px.scatter(
        df,
        x='base_price',
        y='occupancy',
        color='route',
        size='total_seats',
        hover_data=['travel_date', 'bus_type', 'days_to_departure'],
        title="💹 Price Elasticity Analysis",
        labels={'base_price': 'Ticket Price (₹)', 'occupancy': 'Occupancy (%)'}
    )
    
    # Add trend line
    if len(df) > 5:
        z = np.polyfit(df['base_price'], df['occupancy'], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['base_price'].min(), df['base_price'].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_range, y=p(x_range),
            mode='lines',
            name='Demand Curve',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6e6e6')
    )
    return fig


def create_yield_heatmap(df):
    """RevMax-style: Revenue potential heatmap by day and route."""
    pivot = df.pivot_table(
        values='base_price',
        index='route',
        columns='day_name',
        aggfunc='mean'
    )
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        text=np.round(pivot.values, 0),
        texttemplate='₹%{text:.0f}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Avg Price")
    ))
    
    fig.update_layout(
        title="📅 Yield Matrix: Route × Day Pricing",
        template="plotly_dark",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_demand_forecast_chart(df):
    """Sciative-style: Days-to-departure demand curve."""
    if 'days_to_departure' not in df.columns:
        return None
    
    agg = df.groupby('days_to_departure').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'sold_seats': 'sum'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price curve
    fig.add_trace(
        go.Scatter(
            x=agg['days_to_departure'],
            y=agg['base_price'],
            name='Avg Price',
            line=dict(color='#4facfe', width=3),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.2)'
        ),
        secondary_y=False
    )
    
    # Occupancy curve
    fig.add_trace(
        go.Scatter(
            x=agg['days_to_departure'],
            y=agg['occupancy'],
            name='Avg Occupancy %',
            line=dict(color='#f093fb', width=3)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="📈 Demand Curve: Price & Occupancy vs Days-to-Departure",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    
    fig.update_xaxes(title_text="Days Until Departure", autorange="reversed")
    fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Occupancy %", secondary_y=True)
    
    return fig


def create_weekend_impact_chart(df):
    """Show weekend/weekday pricing differential."""
    weekday_avg = df[~df['is_weekend']]['base_price'].mean()
    weekend_avg = df[df['is_weekend']]['base_price'].mean()
    
    premium = ((weekend_avg - weekday_avg) / weekday_avg * 100) if weekday_avg > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Weekday', 'Weekend'],
        y=[weekday_avg, weekend_avg],
        marker=dict(
            color=['#4facfe', '#f093fb'],
            line=dict(color=['#4facfe', '#f093fb'], width=2)
        ),
        text=[f'₹{weekday_avg:.0f}', f'₹{weekend_avg:.0f}'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"🗓️ Weekend Premium: +{premium:.1f}%",
        template="plotly_dark",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        yaxis_title="Average Price (₹)"
    )
    
    return fig


def create_occupancy_gauge(occupancy):
    """RevMax-style occupancy gauge."""
    color = '#00c853' if occupancy >= 70 else '#ffa726' if occupancy >= 50 else '#ff5252'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=occupancy,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#fff'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#888'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 82, 82, 0.3)'},
                {'range': [50, 70], 'color': 'rgba(255, 167, 38, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(0, 200, 83, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 2},
                'thickness': 0.8,
                'value': occupancy
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e6e6e6'}
    )
    
    return fig


def create_route_performance_chart(df):
    """Route-wise revenue and occupancy comparison."""
    route_stats = df.groupby('route').agg({
        'base_price': 'mean',
        'occupancy': 'mean',
        'sold_seats': 'sum',
        'bus_id': 'count'
    }).reset_index()
    route_stats.columns = ['Route', 'Avg Price', 'Avg Occupancy', 'Total Sold', 'Bus Count']
    route_stats = route_stats.sort_values('Avg Occupancy', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=route_stats['Route'],
        x=route_stats['Avg Occupancy'],
        orientation='h',
        name='Occupancy %',
        marker=dict(
            color=route_stats['Avg Occupancy'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Occ %", x=1.02)
        ),
        text=route_stats['Avg Occupancy'].round(1).astype(str) + '%',
        textposition='inside'
    ))
    
    fig.update_layout(
        title="🛣️ Route Performance: Occupancy Ranking",
        template="plotly_dark",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Average Occupancy %",
        showlegend=False
    )
    
    return fig


def create_seat_type_analysis(df):
    """Price distribution by bus/seat type."""
    fig = px.box(
        df,
        x='bus_type',
        y='base_price',
        color='bus_type',
        title="💺 Price Variance by Coach Type",
        labels={'base_price': 'Price (₹)', 'bus_type': 'Coach Type'}
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


# ==============================================================================
# 🖥️ MAIN DASHBOARD
# ==============================================================================

def main():
    # Header with status indicator
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h1 class='main-title'>🎯 Dynamic Pricing Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Sciative + RevMax Analytics | Powered by Vignesh TAT Data Pipeline</p>", unsafe_allow_html=True)
    
    # Load data
    df, error_info = load_data()
    
    with col2:
        if error_info["status"] == "connected":
            st.markdown("<span class='status-live'>🟢 LIVE DATA</span>", unsafe_allow_html=True)
            st.caption(f"Neon DB Connected")
        else:
            st.markdown("<span class='status-demo'>⚠️ DEMO MODE</span>", unsafe_allow_html=True)
            with st.expander("Debug Info"):
                st.code(f"Status: {error_info['status']}\n{error_info['message']}\n{error_info['details']}")
    
    # Preprocess
    df = preprocess_data(df)
    
    st.markdown("---")
    
    # ==== KPI CARDS ====
    st.markdown("<p class='section-header'>📊 Key Performance Indicators</p>", unsafe_allow_html=True)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.metric(
            "Total Records",
            f"{len(df):,}",
            delta="Live Data" if error_info["status"] == "connected" else "Demo",
            delta_color="normal" if error_info["status"] == "connected" else "off"
        )
    
    with k2:
        avg_price = df['base_price'].mean()
        st.metric("Avg Ticket Price", f"₹{avg_price:,.0f}")
    
    with k3:
        avg_occ = df['occupancy'].mean()
        occ_status = "High" if avg_occ >= 70 else "Medium" if avg_occ >= 50 else "Low"
        st.metric("Avg Occupancy", f"{avg_occ:.1f}%", delta=occ_status)
    
    with k4:
        total_sold = df['sold_seats'].sum()
        st.metric("Seats Sold", f"{total_sold:,}")
    
    with k5:
        # Weekend premium calculation
        if df['is_weekend'].any() and (~df['is_weekend']).any():
            wknd = df[df['is_weekend']]['base_price'].mean()
            wkdy = df[~df['is_weekend']]['base_price'].mean()
            premium = ((wknd - wkdy) / wkdy * 100) if wkdy > 0 else 0
            st.metric("Weekend Premium", f"+{premium:.1f}%")
        else:
            st.metric("Weekend Premium", "N/A")
    
    st.markdown("---")
    
    # ==== MAIN CHARTS - ROW 1 ====
    st.markdown("<p class='section-header'>📈 Demand & Price Analytics</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig = create_demand_forecast_chart(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.plotly_chart(create_price_elasticity_chart(df), use_container_width=True)
    
    # ==== MAIN CHARTS - ROW 2 ====
    st.markdown("<p class='section-header'>🎯 Yield Management</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([2, 1, 1])
    
    with c1:
        st.plotly_chart(create_yield_heatmap(df), use_container_width=True)
    
    with c2:
        st.plotly_chart(create_weekend_impact_chart(df), use_container_width=True)
    
    with c3:
        st.markdown("**Fleet Occupancy**")
        st.plotly_chart(create_occupancy_gauge(df['occupancy'].mean()), use_container_width=True)
    
    # ==== MAIN CHARTS - ROW 3 ====
    st.markdown("<p class='section-header'>🛣️ Route & Seat Analysis</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(create_route_performance_chart(df), use_container_width=True)
    
    with c2:
        st.plotly_chart(create_seat_type_analysis(df), use_container_width=True)
    
    st.markdown("---")
    
    # ==== ML DATA EXPORT SECTION ====
    st.markdown("<p class='section-header'>🧬 ML Training Data Export</p>", unsafe_allow_html=True)
    
    with st.expander("📥 Download Data for Dynamic Pricing Algorithm Training", expanded=True):
        st.info("""
        **Use this data to train your dynamic pricing ML model:**
        - Features: route, day_of_week, is_weekend, days_to_departure, bus_type, occupancy
        - Target: base_price (or optimal_price after training)
        - Train models like: XGBoost, LightGBM, or Neural Networks
        """)
        
        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        
        with fc1:
            sel_routes = st.multiselect(
                "Filter Routes",
                options=df['route'].unique().tolist(),
                default=df['route'].unique().tolist()
            )
        
        with fc2:
            sel_types = st.multiselect(
                "Filter Coach Types",
                options=df['bus_type'].unique().tolist(),
                default=df['bus_type'].unique().tolist()
            )
        
        with fc3:
            min_occ = st.slider("Min Occupancy %", 0, 100, 0)
        
        # Apply filters
        mask = (
            df['route'].isin(sel_routes) & 
            df['bus_type'].isin(sel_types) & 
            (df['occupancy'] >= min_occ)
        )
        filtered_df = df[mask]
        
        # Display columns for ML
        ml_columns = [
            'travel_date', 'day_name', 'day_of_week', 'is_weekend', 'days_to_departure',
            'route', 'from_city', 'to_city', 'bus_type',
            'base_price', 'min_price', 'max_price',
            'available_seats', 'sold_seats', 'total_seats', 'occupancy'
        ]
        display_cols = [c for c in ml_columns if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].sort_values('travel_date', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name=f"dp_training_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.caption(f"📊 {len(filtered_df):,} records | {len(display_cols)} features")
        
        with col3:
            st.caption(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
        "🚌 Vignesh TAT Dynamic Pricing System | "
        "Pipeline: Oracle Cron → Hourly Scrape → Neon DB → Dashboard | "
        f"Data as of {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
