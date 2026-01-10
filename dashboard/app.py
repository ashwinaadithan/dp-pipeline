"""
🎯 Dynamic Pricing Intelligence Dashboard
==========================================
Sciative + RevMax Analytics for Bus Yield Management
Pipeline: Oracle Cron → Hourly Scrape → Neon DB → Dashboard
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
# 🔌 SAFE DATABASE CONNECTION
# ==============================================================================

src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def get_db_connection():
    """Safely get database connection."""
    try:
        db_url = None
        if hasattr(st, 'secrets') and 'NEON_DATABASE_URL' in st.secrets:
            db_url = st.secrets['NEON_DATABASE_URL']
        if not db_url:
            db_url = os.getenv('NEON_DATABASE_URL')
        if not db_url:
            return None, "NEON_DATABASE_URL not configured"
        
        import psycopg
        conn = psycopg.connect(db_url)
        return conn, None
    except ImportError:
        return None, "psycopg not installed"
    except Exception as e:
        return None, f"Connection error: {str(e)}"


def fetch_latest_data(limit=5000):
    """Fetch data directly from database."""
    conn, error = get_db_connection()
    if conn is None:
        return None, error
    
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bus_id, operator, bus_type, from_city, to_city, travel_date,
                   departure_time, base_price, available_seats, sold_seats,
                   min_price, max_price, scraped_at
            FROM buses ORDER BY scraped_at DESC LIMIT %s
        """, (limit,))
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows] if rows else None, None if rows else "No data found"
    except Exception as e:
        return None, f"Query failed: {str(e)}"


# ==============================================================================
# 🎨 PAGE CONFIG & PREMIUM STYLING
# ==============================================================================

st.set_page_config(
    page_title="Dynamic Pricing Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Premium dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0d1321 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero header section */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    /* Glass morphism KPI cards */
    div[data-testid="metric-container"] {
        background: rgba(17, 24, 39, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    
    div[data-testid="metric-container"] label {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #f9fafb !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #60a5fa;
        font-size: 1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Live badge */
    .badge-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
        box-shadow: 0 2px 12px rgba(16, 185, 129, 0.4);
    }
    
    .badge-live::before {
        content: '';
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .badge-demo {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 2rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Clean dividers */
    hr {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        margin: 1.5rem 0;
    }
    
    /* Data tables */
    div[data-testid="stDataFrame"] {
        background: rgba(17, 24, 39, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    /* Filter section */
    .filter-container {
        background: rgba(17, 24, 39, 0.5);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #1f2937; }
    ::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 3px; }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(17, 24, 39, 0.8) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 📊 DATA FUNCTIONS
# ==============================================================================

@st.cache_data(ttl=300)
def load_data():
    """Load from database or generate demo data."""
    data, error = fetch_latest_data(limit=5000)
    if data:
        return pd.DataFrame(data), None
    return generate_demo_data(), error


def generate_demo_data():
    """Demo data for preview."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now(), periods=14, freq='D')
    records = []
    routes = [("Chennai", "Tirunelveli", 1200), ("Tirunelveli", "Chennai", 1200),
              ("Chennai", "Madurai", 900), ("Madurai", "Chennai", 900)]
    
    for d in dates:
        for from_city, to_city, base in routes:
            demand = 1.2 if d.dayofweek >= 5 else 1.0
            records.append({
                "bus_id": f"VT_{len(records)}", "travel_date": d, "from_city": from_city,
                "to_city": to_city, "base_price": int(base * demand * np.random.uniform(0.9, 1.15)),
                "available_seats": np.random.randint(8, 28), "sold_seats": np.random.randint(12, 35),
                "bus_type": np.random.choice(["Volvo Sleeper", "Multi-Axle", "AC Sleeper"]),
                "operator": "Vignesh TAT", "departure_time": np.random.choice(["20:00", "21:30", "22:00"]),
                "scraped_at": datetime.now()
            })
    return pd.DataFrame(records)


def preprocess(df):
    """Add computed columns."""
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
    else:
        df['days_to_departure'] = 7
    return df


def to_excel(df):
    """Convert DataFrame to Excel bytes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Pricing Data')
    return output.getvalue()


# ==============================================================================
# 📈 POLISHED CHART FUNCTIONS
# ==============================================================================

CHART_COLORS = {
    'primary': '#60a5fa',
    'secondary': '#f472b6',
    'accent': '#a78bfa',
    'success': '#34d399',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'grid': 'rgba(99, 102, 241, 0.1)',
    'text': '#e5e7eb'
}

def base_layout(height=350, title=""):
    """Common chart layout."""
    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": CHART_COLORS['text'], "size": 12},
        "margin": {"l": 50, "r": 30, "t": 60, "b": 50},
        "height": height,
        "title": {"text": title, "font": {"size": 15, "color": CHART_COLORS['text']}, "x": 0.02}
    }


def chart_demand_curve(df):
    """Price & Occupancy vs Days to Departure - polished dual axis."""
    if 'days_to_departure' not in df.columns:
        return None
    
    agg = df.groupby('days_to_departure').agg({
        'base_price': 'mean', 'occupancy': 'mean'
    }).reset_index().sort_values('days_to_departure', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price line with gradient fill
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['base_price'],
        mode='lines+markers', name='Avg Price ₹',
        line=dict(color=CHART_COLORS['primary'], width=3, shape='spline'),
        marker=dict(size=10, color=CHART_COLORS['primary'], line=dict(width=2, color='white')),
        fill='tozeroy', fillcolor='rgba(96, 165, 250, 0.15)'
    ), secondary_y=False)
    
    # Occupancy line
    fig.add_trace(go.Scatter(
        x=agg['days_to_departure'], y=agg['occupancy'],
        mode='lines+markers', name='Occupancy %',
        line=dict(color=CHART_COLORS['secondary'], width=3, shape='spline'),
        marker=dict(size=10, color=CHART_COLORS['secondary'], line=dict(width=2, color='white'))
    ), secondary_y=True)
    
    fig.update_layout(**base_layout(380, "📈 Price & Demand vs Days Before Travel"))
    fig.update_xaxes(title="Days Until Departure", autorange="reversed", gridcolor=CHART_COLORS['grid'], showline=True, linecolor=CHART_COLORS['grid'])
    fig.update_yaxes(title="Avg Price ₹", secondary_y=False, gridcolor=CHART_COLORS['grid'])
    fig.update_yaxes(title="Occupancy %", secondary_y=True, gridcolor=CHART_COLORS['grid'])
    
    return fig


def chart_price_scatter(df):
    """Price vs Occupancy scatter - polished with better legend."""
    # Limit to top 6 routes for cleaner legend
    top_routes = df['route'].value_counts().head(6).index.tolist()
    df_filtered = df[df['route'].isin(top_routes)]
    
    fig = px.scatter(
        df_filtered, x='base_price', y='occupancy', color='route',
        size='total_seats', size_max=20,
        hover_data={'travel_date': True, 'bus_type': True, 'base_price': ':.0f', 'occupancy': ':.1f'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Add trend line
    if len(df) > 10:
        z = np.polyfit(df['base_price'].dropna(), df['occupancy'].dropna(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(df['base_price'].min(), df['base_price'].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_range, y=p(x_range), mode='lines', name='Trend',
            line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dash')
        ))
    
    fig.update_layout(**base_layout(380, "💹 Price vs Occupancy by Route"))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)))
    fig.update_xaxes(title="Ticket Price ₹", gridcolor=CHART_COLORS['grid'])
    fig.update_yaxes(title="Occupancy %", gridcolor=CHART_COLORS['grid'])
    
    return fig


def chart_yield_heatmap(df):
    """Route × Day yield matrix - polished."""
    pivot = df.pivot_table(values='base_price', index='route', columns='day_name', aggfunc='mean')
    day_map = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 
               'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}
    pivot = pivot.rename(columns=day_map)
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot = pivot[[d for d in day_order if d in pivot.columns]]
    
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#f472b6']],
        text=np.round(pivot.values, 0), texttemplate='₹%{text:.0f}',
        textfont={"size": 10, "color": "white"}, hoverongaps=False,
        colorbar=dict(title="Price ₹", tickfont=dict(color=CHART_COLORS['text']))
    ))
    
    fig.update_layout(**base_layout(320, "🗓️ Yield Matrix: Route × Day"))
    
    return fig


def chart_weekend_comparison(df):
    """Weekend vs Weekday comparison - polished bar chart."""
    if not df['is_weekend'].any() or df['is_weekend'].all():
        return None
    
    wkday_price = df[~df['is_weekend']]['base_price'].mean()
    wkend_price = df[df['is_weekend']]['base_price'].mean()
    wkday_occ = df[~df['is_weekend']]['occupancy'].mean()
    wkend_occ = df[df['is_weekend']]['occupancy'].mean()
    premium = ((wkend_price - wkday_price) / wkday_price * 100) if wkday_price > 0 else 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg Price', x=['Weekday', 'Weekend'], y=[wkday_price, wkend_price],
        marker=dict(color=[CHART_COLORS['primary'], CHART_COLORS['secondary']], 
                    line=dict(width=0)),
        text=[f'₹{wkday_price:.0f}', f'₹{wkend_price:.0f}'], textposition='outside',
        textfont=dict(color=CHART_COLORS['text'], size=12)
    ))
    
    title_text = f"📊 Weekend Premium: {premium:+.1f}%"
    fig.update_layout(**base_layout(300, title_text))
    fig.update_layout(showlegend=False, bargap=0.4)
    fig.update_yaxes(title="Avg Price ₹", gridcolor=CHART_COLORS['grid'])
    
    return fig


def chart_occupancy_gauge(occ):
    """Fleet occupancy gauge - polished."""
    if occ >= 70:
        color = CHART_COLORS['success']
    elif occ >= 50:
        color = CHART_COLORS['warning']
    else:
        color = CHART_COLORS['danger']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=occ,
        number={'suffix': '%', 'font': {'size': 36, 'color': CHART_COLORS['text'], 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#4b5563', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': '#1f2937',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(248, 113, 113, 0.15)'},
                {'range': [50, 70], 'color': 'rgba(251, 191, 36, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(52, 211, 153, 0.15)'}
            ]
        }
    ))
    
    fig.update_layout(height=230, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=20, l=30, r=30))
    
    return fig


def chart_route_ranking(df):
    """Route occupancy ranking - horizontal bar."""
    route_stats = df.groupby('route')['occupancy'].mean().sort_values().reset_index()
    
    colors = [CHART_COLORS['danger'] if v < 50 else CHART_COLORS['warning'] if v < 70 else CHART_COLORS['success'] 
              for v in route_stats['occupancy']]
    
    fig = go.Figure(go.Bar(
        y=route_stats['route'], x=route_stats['occupancy'],
        orientation='h', marker_color=colors,
        text=route_stats['occupancy'].round(1).astype(str) + '%',
        textposition='inside', textfont=dict(color='white', size=11)
    ))
    
    fig.update_layout(**base_layout(320, "🛤️ Route Occupancy Ranking"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="Avg Occupancy %", gridcolor=CHART_COLORS['grid'])
    
    return fig


def chart_bus_type_box(df):
    """Price distribution by bus type - polished box plot."""
    fig = px.box(df, x='bus_type', y='base_price', color='bus_type',
                 color_discrete_sequence=[CHART_COLORS['primary'], CHART_COLORS['secondary'], 
                                          CHART_COLORS['accent'], CHART_COLORS['success']])
    
    fig.update_layout(**base_layout(320, "🚌 Price Distribution by Coach"))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="", gridcolor=CHART_COLORS['grid'])
    fig.update_yaxes(title="Price ₹", gridcolor=CHART_COLORS['grid'])
    
    return fig


# ==============================================================================
# 🖥️ MAIN APP
# ==============================================================================

def main():
    # ===== HERO HEADER =====
    col1, col2 = st.columns([5, 1])
    
    with col1:
        st.markdown("<h1 class='hero-title'>🎯 Dynamic Pricing Intelligence</h1>", unsafe_allow_html=True)
        st.markdown("<p class='hero-subtitle'>Sciative + RevMax Analytics • Powered by Vignesh TAT Data Pipeline</p>", unsafe_allow_html=True)
    
    # Load data
    df, error = load_data()
    
    with col2:
        if error:
            st.markdown("<span class='badge-demo'>⚠️ DEMO</span>", unsafe_allow_html=True)
            with st.expander("Details"):
                st.code(error)
        else:
            st.markdown("<span class='badge-live'>LIVE</span>", unsafe_allow_html=True)
    
    df = preprocess(df)
    st.markdown("---")
    
    # ===== KPI CARDS =====
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 Records", f"{len(df):,}")
    k2.metric("💰 Avg Price", f"₹{df['base_price'].mean():,.0f}")
    k3.metric("📈 Avg Occupancy", f"{df['occupancy'].mean():.1f}%")
    k4.metric("🎫 Seats Sold", f"{df['sold_seats'].sum():,}")
    
    if df['is_weekend'].any() and (~df['is_weekend']).any():
        wknd = df[df['is_weekend']]['base_price'].mean()
        wkdy = df[~df['is_weekend']]['base_price'].mean()
        prem = ((wknd / wkdy) - 1) * 100 if wkdy > 0 else 0
        k5.metric("🗓️ Weekend Lift", f"{prem:+.1f}%")
    else:
        k5.metric("🗓️ Weekend Lift", "—")
    
    st.markdown("---")
    
    # ===== ROW 1: DEMAND ANALYTICS =====
    st.markdown("<div class='section-header'>📈 Demand & Price Analytics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        fig = chart_demand_curve(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.plotly_chart(chart_price_scatter(df), use_container_width=True)
    
    st.markdown("---")
    
    # ===== ROW 2: YIELD MANAGEMENT =====
    st.markdown("<div class='section-header'>🎯 Yield Management</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.5, 1.5, 1.5])
    
    with c1:
        st.plotly_chart(chart_yield_heatmap(df), use_container_width=True)
    
    with c2:
        fig = chart_weekend_comparison(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with c3:
        st.markdown("##### 🎯 Fleet Occupancy")
        st.plotly_chart(chart_occupancy_gauge(df['occupancy'].mean()), use_container_width=True)
    
    st.markdown("---")
    
    # ===== ROW 3: ROUTE & FLEET =====
    st.markdown("<div class='section-header'>🛤️ Route & Fleet Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.plotly_chart(chart_route_ranking(df), use_container_width=True)
    
    with c2:
        st.plotly_chart(chart_bus_type_box(df), use_container_width=True)
    
    st.markdown("---")
    
    # ===== DATA TABLE WITH FILTERS =====
    st.markdown("<div class='section-header'>📥 ML Training Data Export</div>", unsafe_allow_html=True)
    
    # Filter controls
    with st.container():
        f1, f2, f3, f4 = st.columns(4)
        
        with f1:
            routes_list = ['All Routes'] + sorted(df['route'].unique().tolist())
            selected_routes = st.multiselect("🛤️ Routes", routes_list, default=['All Routes'])
        
        with f2:
            types_list = ['All Types'] + sorted(df['bus_type'].unique().tolist())
            selected_types = st.multiselect("🚌 Coach Type", types_list, default=['All Types'])
        
        with f3:
            min_price, max_price = int(df['base_price'].min()), int(df['base_price'].max())
            price_range = st.slider("💰 Price Range", min_price, max_price, (min_price, max_price))
        
        with f4:
            occ_range = st.slider("📈 Occupancy %", 0, 100, (0, 100))
    
    # Apply filters
    mask = pd.Series([True] * len(df))
    
    if 'All Routes' not in selected_routes and selected_routes:
        mask &= df['route'].isin(selected_routes)
    
    if 'All Types' not in selected_types and selected_types:
        mask &= df['bus_type'].isin(selected_types)
    
    mask &= (df['base_price'] >= price_range[0]) & (df['base_price'] <= price_range[1])
    mask &= (df['occupancy'] >= occ_range[0]) & (df['occupancy'] <= occ_range[1])
    
    filtered_df = df[mask]
    
    # Display columns
    display_cols = ['travel_date', 'day_name', 'route', 'bus_type', 'departure_time',
                    'base_price', 'available_seats', 'sold_seats', 'occupancy', 'days_to_departure']
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    # Stats row
    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} records** | "
                f"Avg Price: ₹{filtered_df['base_price'].mean():,.0f} | "
                f"Avg Occupancy: {filtered_df['occupancy'].mean():.1f}%")
    
    # Data table
    st.dataframe(
        filtered_df[display_cols].sort_values('travel_date', ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "travel_date": st.column_config.DateColumn("Travel Date", format="DD-MMM-YYYY"),
            "base_price": st.column_config.NumberColumn("Price ₹", format="₹%d"),
            "occupancy": st.column_config.ProgressColumn("Occupancy", min_value=0, max_value=100, format="%.1f%%"),
        }
    )
    
    # Export buttons
    st.markdown("#### 📥 Download Data")
    e1, e2, e3 = st.columns([1, 1, 3])
    
    with e1:
        csv_data = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=f"pricing_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with e2:
        excel_data = to_excel(filtered_df[display_cols])
        st.download_button(
            "⬇️ Download Excel",
            data=excel_data,
            file_name=f"pricing_data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"🚌 Vignesh TAT Dynamic Pricing System • Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")


if __name__ == "__main__":
    main()
