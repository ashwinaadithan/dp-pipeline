# Vignesh TAT Dynamic Pricing - Dashboard Documentation

## Overview
/
This dashboard visualizes real-time bus pricing data for **Vignesh TAT** operator, scraped hourly from RedBus via our automated pipeline. All data is stored in Neon PostgreSQL and displayed dynamically - **no hardcoded values**.

---

## Data Pipeline

```
Oracle Cron (Hourly) → RedBus Scraper → Neon PostgreSQL → Streamlit Dashboard
```

- **Scraper**: `src/scraper.py` - Extracts bus data from RedBus operator page
- **Database**: Neon PostgreSQL - Stores all scraped data with timestamps
- **Dashboard**: `dashboard/app.py` - Visualizes data for pricing analysis

---

## Dashboard Charts & Their Purpose

### 1. Price & Occupancy vs Days to Travel
**What it shows**: How prices and seat availability change as departure date approaches.

**Why it matters for DP**: 
- Identifies the "sweet spot" for pricing adjustments
- Shows demand patterns (occupancy rises, price should too)
- Helps set dynamic pricing rules: "If days_to_departure < 3 AND occupancy > 70%, increase price by 15%"

### 2. Price vs Occupancy Scatter
**What it shows**: Relationship between ticket price and booking rate across routes.

**Why it matters for DP**:
- Reveals price elasticity - how sensitive customers are to price changes
- Routes with high occupancy at high prices = inelastic demand (can charge more)
- Routes with low occupancy at any price = need promotions

### 3. Yield Matrix (Route × Day)
**What it shows**: Average price by route and day of week in a heatmap.

**Why it matters for DP**:
- Identifies which route-day combinations command premium pricing
- Weekend vs weekday patterns visible at a glance
- Direct input for dynamic pricing rules per route

### 4. Weekend Premium
**What it shows**: Price difference between weekday and weekend travel.

**Why it matters for DP**:
- Quantifies the weekend demand surge
- Sets baseline for time-based pricing multipliers
- Example: Weekend = +20% base price

### 5. Fleet Occupancy Gauge
**What it shows**: Overall seat utilization across all buses.

**Why it matters for DP**:
- Health indicator for pricing strategy
- Too low = prices too high or demand issues
- Too high = leaving money on table, should price higher

### 6. Route Occupancy Ranking
**What it shows**: Which routes perform best/worst by booking rate.

**Why it matters for DP**:
- Low-performing routes may need promotional pricing
- High-performing routes can sustain premium pricing
- Helps allocate buses to profitable routes

### 7. Price by Coach Type
**What it shows**: Price distribution across bus types (Volvo, Multi-Axle, etc.).

**Why it matters for DP**:
- Premium coaches can charge more
- Shows price variance - opportunity for optimization
- Helps set coach-type multipliers

---

## Building Your Dynamic Pricing Algorithm

### Step 1: Collect Data (Done ✓)
Our scraper collects:
- Route, date, departure time
- Base price, seat availability
- Occupancy rates

### Step 2: Feature Engineering
From this dashboard, you can extract features:
- `days_to_departure` - urgency factor
- `is_weekend` - demand multiplier
- `route_popularity` - route-specific pricing
- `coach_type` - premium tier
- `time_of_day` - departure timing

### Step 3: Train ML Model
```python
# Example features for pricing model
features = ['days_to_departure', 'is_weekend', 'route_demand_score', 
            'current_occupancy', 'coach_tier', 'competitor_price']
target = 'optimal_price'

# Models to try:
# - XGBoost / LightGBM for tabular data
# - Neural network for complex patterns
# - Reinforcement learning for real-time adjustments
```

### Step 4: Pricing Rules (Hybrid Approach)
Combine ML predictions with business rules:

```python
def calculate_dynamic_price(base_price, features):
    # ML prediction
    ml_adjustment = model.predict(features)
    
    # Business rules
    if features['days_to_departure'] < 2 and features['occupancy'] > 80:
        surge_factor = 1.25  # 25% surge
    elif features['is_weekend']:
        surge_factor = 1.15
    else:
        surge_factor = 1.0
    
    return base_price * ml_adjustment * surge_factor
```

---

## Key Metrics to Monitor

| Metric | Target | Action if Off-Target |
|--------|--------|---------------------|
| Avg Occupancy | 70-85% | Adjust base prices |
| Weekend Premium | 15-25% | Check competitor pricing |
| Days-to-Departure Curve | Rising | Implement time-based surge |
| Route Variance | Low | Standardize pricing rules |

---

## Next Steps for DP Development

1. **Export data** using dashboard CSV/Excel buttons
2. **Analyze patterns** in Jupyter/Python
3. **Build baseline model** with XGBoost
4. **Test pricing rules** on historical data
5. **Deploy real-time pricing** API
6. **Monitor and iterate** using this dashboard

---

## Technical: Stable Bus ID System

Starting from January 2026, buses now have **stable IDs** that persist across hourly scrapes:

**Format**: `{OPERATOR}_{FROM}_{TO}_{DATE}_{TIME}`

**Example**: `VIG_CHE_TIR_20260115_2100`
- VIG = Vignesh TAT (first 3 chars)
- CHE = Chennai (from city)
- TIR = Tirunelveli (to city)
- 20260115 = January 15, 2026
- 2100 = 21:00 departure

**Why this matters**:
- Same bus = same ID across all hourly scrapes
- Enables tracking price changes: ₹1000 → ₹1200 → ₹1500
- Critical for dynamic pricing algorithm training
- Price history stored in `price_history` table

**Price History Query**:
```python
from database import get_price_history

# Get price history for a specific bus
history = get_price_history(bus_id="VIG_CHE_TIR_20260115_2100")
# Returns: [{base_price: 1000, scraped_at: "10:00"}, {base_price: 1200, scraped_at: "11:00"}, ...]

# Find buses that had price changes (dynamic pricing in action)
changed = get_buses_with_price_changes()
# Returns buses where max_price != min_price
```

---

*Dashboard data source: Neon PostgreSQL | Updated every 5 minutes | All data from Vignesh TAT operator page on RedBus*
