"""
Multi-Format Data Export
========================
Exports scraped data to JSON, CSV, and Excel formats.
"""

import os
import json
import pandas as pd
from datetime import datetime


def export_to_json(data, filepath):
    """Export data to JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 JSON saved: {filepath}")
        return True
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False


def export_to_csv(data, filepath):
    """Export data to CSV for ML training."""
    try:
        # Flatten bus data for CSV
        rows = []
        
        for session in data.get("scrape_sessions", []):
            for bus in session.get("buses", []):
                route = bus.get("route", {})
                seats = bus.get("seats", {})
                
                # Base row
                base_row = {
                    "bus_id": bus.get("bus_id"),
                    "operator": bus.get("operator"),
                    "bus_type": bus.get("bus_type"),
                    "from_city": route.get("from_city"),
                    "to_city": route.get("to_city"),
                    "travel_date": route.get("travel_date"),
                    "departure_time": bus.get("departure_time"),
                    "arrival_time": bus.get("arrival_time"),
                    "base_price": bus.get("base_price"),
                    "available_seats": seats.get("available_seats", 0),
                    "sold_seats": seats.get("sold_seats", 0),
                    "total_seats": seats.get("total_seats", 0),
                    "occupancy_rate": round(seats.get("sold_seats", 0) / max(seats.get("total_seats", 1), 1) * 100, 2),
                    "min_price": seats.get("price_range", {}).get("min"),
                    "max_price": seats.get("price_range", {}).get("max"),
                    "scraped_at": session.get("scraped_at")
                }
                
                # Add individual seat prices if available
                seat_data = seats.get("available_seats_data", [])
                if seat_data:
                    for seat in seat_data:
                        row = base_row.copy()
                        row["seat_id"] = seat.get("seat_id")
                        row["seat_price"] = seat.get("price")
                        row["deck"] = seat.get("deck")
                        row["is_window"] = seat.get("is_window", False)
                        rows.append(row)
                else:
                    # No seat data, just add base row
                    rows.append(base_row)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            print(f"📊 CSV saved: {filepath} ({len(rows)} rows)")
            return True
        else:
            print("⚠️ No data to export to CSV")
            return False
            
    except Exception as e:
        print(f"❌ CSV export failed: {e}")
        return False


def export_to_excel(data, filepath):
    """Export data to Excel with multiple sheets for clients."""
    try:
        # Prepare dataframes
        buses_rows = []
        seats_rows = []
        routes_summary = {}
        
        for session in data.get("scrape_sessions", []):
            for bus in session.get("buses", []):
                route = bus.get("route", {})
                seats = bus.get("seats", {})
                route_key = f"{route.get('from_city')} → {route.get('to_city')}"
                
                # Bus data
                buses_rows.append({
                    "Bus ID": bus.get("bus_id"),
                    "Operator": bus.get("operator"),
                    "Type": bus.get("bus_type"),
                    "Route": route_key,
                    "Date": route.get("travel_date"),
                    "Departure": bus.get("departure_time"),
                    "Arrival": bus.get("arrival_time"),
                    "Base Price (₹)": bus.get("base_price"),
                    "Available Seats": seats.get("available_seats", 0),
                    "Sold Seats": seats.get("sold_seats", 0),
                    "Min Price (₹)": seats.get("price_range", {}).get("min"),
                    "Max Price (₹)": seats.get("price_range", {}).get("max"),
                })
                
                # Route summary
                if route_key not in routes_summary:
                    routes_summary[route_key] = {
                        "Route": route_key,
                        "Total Buses": 0,
                        "Avg Price (₹)": [],
                        "Avg Occupancy (%)": []
                    }
                routes_summary[route_key]["Total Buses"] += 1
                if bus.get("base_price"):
                    routes_summary[route_key]["Avg Price (₹)"].append(bus.get("base_price"))
                total = seats.get("total_seats", 0)
                sold = seats.get("sold_seats", 0)
                if total > 0:
                    routes_summary[route_key]["Avg Occupancy (%)"].append(sold / total * 100)
                
                # Seat prices
                for seat in seats.get("available_seats_data", []):
                    seats_rows.append({
                        "Bus ID": bus.get("bus_id"),
                        "Seat ID": seat.get("seat_id"),
                        "Price (₹)": seat.get("price"),
                        "Deck": seat.get("deck"),
                        "Window": "Yes" if seat.get("is_window") else "No"
                    })
        
        # Calculate route averages
        summary_rows = []
        for route_key, data in routes_summary.items():
            summary_rows.append({
                "Route": route_key,
                "Total Buses": data["Total Buses"],
                "Avg Price (₹)": round(sum(data["Avg Price (₹)"]) / len(data["Avg Price (₹)"]), 0) if data["Avg Price (₹)"] else 0,
                "Avg Occupancy (%)": round(sum(data["Avg Occupancy (%)"]) / len(data["Avg Occupancy (%)"]), 1) if data["Avg Occupancy (%)"] else 0
            })
        
        # Create Excel with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Route Summary', index=False)
            if buses_rows:
                pd.DataFrame(buses_rows).to_excel(writer, sheet_name='All Buses', index=False)
            if seats_rows:
                pd.DataFrame(seats_rows).to_excel(writer, sheet_name='Seat Prices', index=False)
        
        print(f"📑 Excel saved: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Excel export failed: {e}")
        return False


def export_all(data, output_folder):
    """Export data to all formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_folder, exist_ok=True)
    
    json_path = os.path.join(output_folder, f"data_{timestamp}.json")
    csv_path = os.path.join(output_folder, f"data_ml_{timestamp}.csv")
    excel_path = os.path.join(output_folder, f"report_{timestamp}.xlsx")
    
    results = {
        "json": export_to_json(data, json_path),
        "csv": export_to_csv(data, csv_path),
        "excel": export_to_excel(data, excel_path)
    }
    
    return {
        "json_path": json_path if results["json"] else None,
        "csv_path": csv_path if results["csv"] else None,
        "excel_path": excel_path if results["excel"] else None
    }
