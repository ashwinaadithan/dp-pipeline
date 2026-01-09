"""
Neon Database Integration
=========================
Handles connection to Neon PostgreSQL and data storage.
FREE tier: 0.5GB storage, always-on serverless.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment or Streamlit secrets
DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")

# Verify if running in Streamlit and try to get secret
if not DATABASE_URL:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NEON_DATABASE_URL" in st.secrets:
            DATABASE_URL = st.secrets["NEON_DATABASE_URL"]
    except:
        pass

SAVE_TO_DB = os.getenv("SAVE_TO_DATABASE", "true").lower() == "true"


def get_connection():
    """Get database connection."""
    if not DATABASE_URL:
        print("⚠️ NEON_DATABASE_URL not set. Database saving disabled.")
        return None
    
    try:
        import psycopg
        conn = psycopg.connect(DATABASE_URL)
        return conn
    except ImportError:
        print("⚠️ psycopg not installed. Database features disabled.")
        return None
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None


def init_database():
    """Create tables if they don't exist."""
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        
        # Create tables
        cur.execute("""
        CREATE TABLE IF NOT EXISTS scrape_runs (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50) UNIQUE,
            operator VARCHAR(100),
            operator_url TEXT,
            scraped_at TIMESTAMP,
            total_buses INTEGER,
            total_routes INTEGER,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS buses (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(50),
            bus_id VARCHAR(50),
            operator VARCHAR(100),
            bus_type VARCHAR(50),
            from_city VARCHAR(100),
            to_city VARCHAR(100),
            travel_date DATE,
            departure_time VARCHAR(10),
            arrival_time VARCHAR(10),
            base_price INTEGER,
            available_seats INTEGER,
            sold_seats INTEGER,
            min_price INTEGER,
            max_price INTEGER,
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE TABLE IF NOT EXISTS seat_prices (
            id SERIAL PRIMARY KEY,
            bus_id VARCHAR(50),
            seat_id VARCHAR(20),
            price INTEGER,
            deck VARCHAR(10),
            is_window BOOLEAN,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_buses_run ON buses(run_id);
        CREATE INDEX IF NOT EXISTS idx_buses_date ON buses(travel_date);
        CREATE INDEX IF NOT EXISTS idx_buses_route ON buses(from_city, to_city);
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database tables initialized")
        return True
        
    except Exception as e:
        print(f"❌ Database init failed: {e}")
        return False


def save_scrape_run(run_data):
    """Save a complete scrape run to database."""
    if not SAVE_TO_DB:
        return False
    
    conn = get_connection()
    if not conn:
        return False
    
    try:
        cur = conn.cursor()
        session = run_data.get("scrape_sessions", [{}])[0]
        
        # Insert run metadata
        cur.execute("""
        INSERT INTO scrape_runs (run_id, operator, operator_url, scraped_at, total_buses, total_routes)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING
        """, (
            session.get("session_id"),
            session.get("operator"),
            session.get("operator_url"),
            session.get("scraped_at"),
            len(session.get("buses", [])),
            len(session.get("expected_routes", {}))
        ))
        
        # Insert buses
        for bus in session.get("buses", []):
            route = bus.get("route", {})
            seats = bus.get("seats", {})
            
            cur.execute("""
            INSERT INTO buses (
                run_id, bus_id, operator, bus_type,
                from_city, to_city, travel_date,
                departure_time, arrival_time, base_price,
                available_seats, sold_seats, min_price, max_price, scraped_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session.get("session_id"),
                bus.get("bus_id"),
                bus.get("operator"),
                bus.get("bus_type"),
                route.get("from_city"),
                route.get("to_city"),
                # Parse date if needed (handle DD-MMM-YYYY from scraper)
                _parse_date(route.get("travel_date")),
                bus.get("departure_time"),
                bus.get("arrival_time"),
                bus.get("base_price"),
                seats.get("available_seats", 0),
                seats.get("sold_seats", 0),
                seats.get("price_range", {}).get("min"),
                seats.get("price_range", {}).get("max"),
                session.get("scraped_at")
            ))
            
            # Insert seat prices
            for seat in seats.get("available_seats_data", []):
                cur.execute("""
                INSERT INTO seat_prices (bus_id, seat_id, price, deck, is_window)
                VALUES (%s, %s, %s, %s, %s)
                """, (
                    bus.get("bus_id"),
                    seat.get("seat_id"),
                    seat.get("price"),
                    seat.get("deck"),
                    seat.get("is_window", False)
                ))
        
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved {len(session.get('buses', []))} buses to database")
        return True
        
    except Exception as e:
        print(f"❌ Database save failed: {e}")
        return False


def get_latest_data(limit=1000):
    """Fetch latest data for dashboard."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        cur.execute("""
        SELECT 
            b.bus_id, b.operator, b.bus_type,
            b.from_city, b.to_city, b.travel_date,
            b.departure_time, b.base_price,
            b.available_seats, b.sold_seats,
            b.min_price, b.max_price, b.scraped_at
        FROM buses b
        ORDER BY b.scraped_at DESC
        LIMIT %s
        """, (limit,))
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
        
    except Exception as e:
        print(f"❌ Database fetch failed: {e}")
        return []


def get_price_trends(route_from=None, route_to=None, days=7):
    """Get price trends for visualization."""
    conn = get_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        
        query = """
        SELECT 
            travel_date,
            from_city,
            to_city,
            AVG(base_price) as avg_price,
            MIN(min_price) as min_price,
            MAX(max_price) as max_price,
            AVG(available_seats) as avg_available,
            AVG(sold_seats) as avg_sold,
            COUNT(*) as bus_count
        FROM buses
        WHERE scraped_at >= NOW() - INTERVAL '%s days'
        """
        
        params = [days]
        
        if route_from:
            query += " AND from_city = %s"
            params.append(route_from)
        if route_to:
            query += " AND to_city = %s"
            params.append(route_to)
        
        query += " GROUP BY travel_date, from_city, to_city ORDER BY travel_date"
        
        cur.execute(query, params)
        
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
        
    except Exception as e:
        print(f"❌ Price trends fetch failed: {e}")
        return []


def _parse_date(date_str):
    """Parse date string to YYYY-MM-DD format."""
    if not date_str:
        return None
    
    try:
        # Try DD-MMM-YYYY (e.g., 10-Jan-2026)
        date_obj = datetime.strptime(date_str, "%d-%b-%Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        try:
            # Try YYYY-MM-DD (already correct)
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            return None


# Initialize on import
if SAVE_TO_DB and DATABASE_URL:
    init_database()
