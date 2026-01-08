"""
REDBUS OPERATOR SCRAPER v3.3 - FULLY AUTONOMOUS + SCHEDULED
=============================================================
Works with ANY RedBus operator page - just change OPERATOR_URL.
Runs automatically on a schedule for continuous data collection.

Key features:
1. FULLY AUTONOMOUS: Change OPERATOR_URL to scrape any operator
2. SCHEDULED EXECUTION: Runs every SCRAPE_INTERVAL_HOURS for TOTAL_DURATION_DAYS
3. DYNAMIC ROUTE DISCOVERY: Scrapes routes directly from operator page
4. EXPECTED BUS COUNT TRACKING: Captures how many buses each route should have
5. VERIFICATION: Compares scraped data against expected counts
6. Properly targets seat panel elements for price extraction
7. Each run saves to a new timestamped file in OUTPUT_FOLDER
"""

import asyncio
import json
import os
import re
import random
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
import pandas as pd

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS AS NEEDED
# ============================================================================

# ========================= OPERATOR SETTINGS =========================
# Operator page URL - CHANGE THIS TO SCRAPE ANY OPERATOR
OPERATOR_URL = "https://www.redbus.in/travels/vignesh-tat"

# ========================= SCHEDULING SETTINGS =========================
# How often to run the scraper (in hours)
SCRAPE_INTERVAL_HOURS = 1

# Total duration to keep running the scraper (in days)
# After this many days, the scheduler will stop automatically
TOTAL_DURATION_DAYS = 7

# ========================= SCRAPING SETTINGS =========================
# How many future days to scrape in each run
DAYS_AHEAD = 7

# ========================= OUTPUT SETTINGS =========================
# Output folder for all scraped data (will be created if doesn't exist)
OUTPUT_FOLDER = "scraped_data"

# ============================================================================
# DERIVED SETTINGS (Don't modify these)
# ============================================================================

# Extract operator name from URL for dynamic use
def extract_operator_from_url(url):
    """Extract operator slug from URL like 'vignesh-tat' from /travels/vignesh-tat"""
    match = re.search(r'/travels/([^/\\?]+)', url)
    if match:
        return match.group(1)
    return "unknown-operator"

# Get operator slug and formatted name
OPERATOR_SLUG = extract_operator_from_url(OPERATOR_URL)
OPERATOR_NAME = OPERATOR_SLUG.replace("-", " ").title()  # "vignesh-tat" -> "Vignesh Tat"

# Create output folder if it doesn't exist
import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output files with timestamp and operator name (in output folder)
def get_output_filenames():
    """Generate timestamped output filenames for current run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    operator_clean = OPERATOR_NAME.replace(' ', '')
    return {
        "json": os.path.join(OUTPUT_FOLDER, f"{operator_clean}_{timestamp}.json"),
        "excel": os.path.join(OUTPUT_FOLDER, f"{operator_clean}_{timestamp}.xlsx"),
        "csv": os.path.join(OUTPUT_FOLDER, f"{operator_clean}_ML_{timestamp}.csv")
    }

# Session tracking
_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
_bus_counter = 0

# Global storage for expected bus counts per route (populated from operator page)
_expected_buses_per_route = {}


def gen_bus_id():
    """Generate unique bus ID"""
    global _bus_counter
    _bus_counter += 1
    return f"RB_{_session_id}_{_bus_counter}"


def parse_price(text):
    """Extract numeric price from text like '₹1,078' -> 1078"""
    if not text:
        return None
    match = re.search(r'₹\s*([0-9,]+)', str(text))
    if match:
        return int(match.group(1).replace(',', ''))
    return None


def get_times(text):
    """Extract departure and arrival times"""
    times = re.findall(r'(\d{1,2}:\d{2})', text)
    dep = times[0] if len(times) >= 1 else ""
    arr = times[1] if len(times) >= 2 else ""
    return dep, arr


# ============================================================================
# BROWSER SETUP
# ============================================================================

# Auto-detect if running in CI (GitHub Actions)
IS_CI = os.getenv("CI", "false").lower() == "true" or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"

async def create_browser(pw):
    """Create browser with stealth settings. Uses headless mode in CI."""
    browser = await pw.chromium.launch(
        headless=IS_CI,  # Headless in CI, visible locally
        args=['--start-maximized', '--no-sandbox', '--disable-dev-shm-usage'] if IS_CI else ['--start-maximized']
    )
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    )
    page = await context.new_page()
    return browser, page


# ============================================================================
# ROUTE DISCOVERY FROM OPERATOR PAGE
# ============================================================================

async def discover_routes_from_operator_page(page):
    """
    Navigate to operator page, scroll to load all routes, and extract:
    - Route names (from city -> to city)
    - Expected bus count per route
    
    Returns: List of tuples (from_city, to_city, expected_bus_count)
    """
    global _expected_buses_per_route
    routes_with_counts = []
    
    print(f"\n{'='*60}")
    print(f"🔍 DISCOVERING ROUTES FROM OPERATOR PAGE")
    print(f"{'='*60}")
    
    try:
        # Navigate to operator page
        print(f"  🌐 Loading operator page: {OPERATOR_URL}")
        await page.goto(OPERATOR_URL, timeout=30000)
        await asyncio.sleep(3)
        
        # Dismiss any popups
        try:
            await page.keyboard.press("Escape")
        except:
            pass
        await asyncio.sleep(1)
        
        # Scroll down to load all routes
        print(f"  📜 Scrolling to load all routes...", end="", flush=True)
        
        last_route_count = 0
        no_change_count = 0
        
        for scroll_num in range(30):  # Max 30 scrolls
            await page.evaluate("window.scrollBy(0, 800)")
            await asyncio.sleep(0.8)
            
            # Count current routes
            current_count = await page.evaluate("""
            () => {
                // Look for route links/cards on operator page
                const routeElements = document.querySelectorAll('[class*="route"], [class*="Route"], a[href*="bus-tickets"], [class*="busCardRow"], [class*="srp"]');
                return routeElements.length;
            }
            """)
            
            if current_count == last_route_count:
                no_change_count += 1
                if no_change_count >= 4:
                    print(f" (done at scroll {scroll_num+1})", flush=True)
                    break
            else:
                no_change_count = 0
            
            last_route_count = current_count
        
        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        
        # Extract all routes with bus counts
        print(f"  📊 Extracting routes and bus counts...")
        
        routes_data = await page.evaluate(r"""
        () => {
            const routes = [];
            const pageText = document.body.innerText || '';
            
            // Method 1: Look for route cards (the main structure on operator pages)
            // Each route card has: route name (e.g. "Chennai to Tirunelveli") and "N bus options"
            const routeCards = document.querySelectorAll('[class*="card"], [class*="route"], [class*="travels"]');
            
            routeCards.forEach(card => {
                const cardText = (card.innerText || '').trim();
                
                // Pattern: "City1 to City2" followed by "N bus options"
                const routeMatch = cardText.match(/([A-Z][a-z]+(?:\([^)]+\))?)\s*to\s*([A-Z][a-z]+(?:\([^)]+\))?)/i);
                const countMatch = cardText.match(/(\d+)\s*bus\s*options?/i);
                
                if (routeMatch && countMatch) {
                    // Clean city names - remove parenthetical content like "(Kerala)"
                    let fromCity = routeMatch[1].trim().replace(/\([^)]+\)/g, '').trim();
                    let toCity = routeMatch[2].trim().replace(/\([^)]+\)/g, '').trim();
                    const busCount = parseInt(countMatch[1]);
                    
                    // Capitalize properly
                    fromCity = fromCity.charAt(0).toUpperCase() + fromCity.slice(1).toLowerCase();
                    toCity = toCity.charAt(0).toUpperCase() + toCity.slice(1).toLowerCase();
                    
                    const routeKey = `${fromCity}-${toCity}`;
                    const exists = routes.some(r => `${r.from}-${r.to}` === routeKey);
                    if (!exists && fromCity !== toCity && busCount > 0) {
                        routes.push({
                            from: fromCity,
                            to: toCity,
                            expected_count: busCount,
                            text: `${fromCity} to ${toCity} (${busCount} bus options)`
                        });
                    }
                }
            });
            
            // Method 2: Parse the entire page text for patterns
            // Looking for: "Route Name" ... "N bus options"
            const lines = pageText.split('\n');
            let currentRoute = null;
            
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i].trim();
                
                // Check for route pattern
                const routeMatch = line.match(/^([A-Z][a-z]+(?:\s*\([^)]+\))?)\s+to\s+([A-Z][a-z]+(?:\s*\([^)]+\))?)$/i);
                if (routeMatch) {
                    let fromCity = routeMatch[1].replace(/\([^)]+\)/g, '').trim();
                    let toCity = routeMatch[2].replace(/\([^)]+\)/g, '').trim();
                    fromCity = fromCity.charAt(0).toUpperCase() + fromCity.slice(1).toLowerCase();
                    toCity = toCity.charAt(0).toUpperCase() + toCity.slice(1).toLowerCase();
                    currentRoute = { from: fromCity, to: toCity };
                }
                
                // Check for bus count pattern (usually right after route name)
                const countMatch = line.match(/^(\d+)\s*bus\s*options?$/i);
                if (countMatch && currentRoute) {
                    const busCount = parseInt(countMatch[1]);
                    const routeKey = `${currentRoute.from}-${currentRoute.to}`;
                    const exists = routes.some(r => `${r.from}-${r.to}` === routeKey);
                    if (!exists && currentRoute.from !== currentRoute.to && busCount > 0) {
                        routes.push({
                            from: currentRoute.from,
                            to: currentRoute.to,
                            expected_count: busCount,
                            text: `${currentRoute.from} to ${currentRoute.to} (${busCount} bus options)`
                        });
                    }
                    currentRoute = null;
                }
            }
            
            // Method 3: Fallback - find links with bus-tickets and nearby "N bus options" 
            const allLinks = document.querySelectorAll('a[href*="bus-tickets"]');
            
            allLinks.forEach(link => {
                const href = link.href || '';
                const urlMatch = href.match(/bus-tickets\/([a-z]+)-to-([a-z]+)/i);
                
                if (urlMatch) {
                    let fromCity = urlMatch[1].charAt(0).toUpperCase() + urlMatch[1].slice(1).toLowerCase();
                    let toCity = urlMatch[2].charAt(0).toUpperCase() + urlMatch[2].slice(1).toLowerCase();
                    
                    // Look for bus count in parent or sibling elements
                    const parent = link.closest('div, li, section') || link.parentElement;
                    if (parent) {
                        const parentText = parent.innerText || '';
                        const countMatch = parentText.match(/(\d+)\s*bus\s*options?/i);
                        if (countMatch) {
                            const busCount = parseInt(countMatch[1]);
                            const routeKey = `${fromCity}-${toCity}`;
                            const exists = routes.some(r => `${r.from}-${r.to}` === routeKey);
                            if (!exists && fromCity !== toCity && busCount > 0) {
                                routes.push({
                                    from: fromCity,
                                    to: toCity,
                                    expected_count: busCount,
                                    text: `${fromCity} to ${toCity} (${busCount} bus options)`
                                });
                            }
                        }
                    }
                }
            });
            
            return routes;
        }
        """)
        
        # Process discovered routes
        for route in routes_data:
            from_city = route.get('from', '')
            to_city = route.get('to', '')
            expected = route.get('expected_count', 0)
            
            if from_city and to_city:
                routes_with_counts.append((from_city, to_city, expected))
                route_key = f"{from_city}->{to_city}"
                _expected_buses_per_route[route_key] = expected
                print(f"    ✓ {from_city} → {to_city}: {expected} buses expected")
        
        print(f"\n  ✅ Discovered {len(routes_with_counts)} routes")
        
    except Exception as e:
        print(f"  ❌ Error discovering routes: {str(e)[:50]}")
    
    return routes_with_counts


def verify_scraped_data(scraped_buses_per_route):
    """
    Compare scraped bus counts against expected counts from operator page.
    
    The operator page shows daily bus counts (e.g., "4 bus options"),
    but we scrape for DAYS_AHEAD days, so expected = daily_count * DAYS_AHEAD.
    
    Args:
        scraped_buses_per_route: Dict of route_key -> scraped_count
    
    Returns:
        Verification report dict
    """
    global _expected_buses_per_route
    
    report = {
        "verification_time": datetime.now().isoformat(),
        "days_scraped": DAYS_AHEAD,
        "routes_verified": [],
        "total_expected": 0,
        "total_scraped": 0,
        "match_rate": 0.0,
        "discrepancies": []
    }
    
    print(f"\n{'='*70}")
    print(f"📋 VERIFICATION REPORT (Scraped {DAYS_AHEAD} days)")
    print(f"{'='*70}")
    print(f"  Route                           | Daily | Expected ({DAYS_AHEAD} days) | Scraped | Status")
    print(f"  {'-'*67}")
    
    for route_key, daily_expected in _expected_buses_per_route.items():
        scraped = scraped_buses_per_route.get(route_key, 0)
        
        # Expected for all days = daily_count * DAYS_AHEAD
        total_expected = daily_expected * DAYS_AHEAD
        
        report["total_expected"] += total_expected
        report["total_scraped"] += scraped
        
        # Check if scraped matches expected (with small tolerance for edge cases)
        matched = scraped == total_expected
        partial = scraped > 0 and scraped < total_expected
        over = scraped > total_expected
        
        if matched:
            status = "✅ MATCH"
        elif over:
            status = "📈 OVER"  # Scraped more than expected (could be new buses added)
        elif partial:
            status = "⚠️ PARTIAL"
        else:
            status = "❌ MISSING"
        
        route_report = {
            "route": route_key,
            "daily_expected": daily_expected,
            "total_expected": total_expected,
            "scraped": scraped,
            "matched": matched
        }
        report["routes_verified"].append(route_report)
        
        print(f"  {route_key:<32} | {daily_expected:>5} | {total_expected:>17} | {scraped:>7} | {status}")
        
        if not matched:
            report["discrepancies"].append({
                "route": route_key,
                "daily_expected": daily_expected,
                "total_expected": total_expected,
                "scraped": scraped,
                "difference": scraped - total_expected
            })
    
    # Calculate match rate
    if report["total_expected"] > 0:
        report["match_rate"] = round((report["total_scraped"] / report["total_expected"]) * 100, 2)
    
    print(f"\n  📊 SUMMARY:")
    print(f"     Total Expected ({DAYS_AHEAD} days): {report['total_expected']} buses")
    print(f"     Total Scraped:                {report['total_scraped']} buses")
    print(f"     Match Rate:                   {report['match_rate']}%")
    
    if not report["discrepancies"]:
        print(f"\n     ✅ All routes matched perfectly!")
    else:
        matches = len(report["routes_verified"]) - len(report["discrepancies"])
        print(f"     Routes Matched: {matches}/{len(report['routes_verified'])}")
        print(f"     Discrepancies:  {len(report['discrepancies'])} routes")
    
    return report


# ============================================================================
# SEAT EXTRACTION - FIXED VERSION
# ============================================================================

async def extract_seats_from_panel(page):
    """
    Extract ONLY AVAILABLE seat prices from the open seat panel.
    
    For dynamic pricing ML:
    - Available seats with prices -> These are what we learn from
    - Sold seats -> Only count them (for occupancy insights), don't store individually
    """
    seats_data = {
        "total_seats": 0,
        "available_seats": 0,
        "sold_seats": 0,
        "available_seats_data": [],
        "all_prices": [],
        "price_range": {"min": None, "max": None}
    }
    
    try:
        # Wait for seat panel to fully render
        await asyncio.sleep(3)
        
        # More aggressive seat extraction - look at the ENTIRE visible page
        result = await page.evaluate(r"""
        () => {
            const availableSeats = [];
            let soldCount = 0;
            const allPrices = [];
            
            // Get all text from the page
            const pageText = document.body.innerText || '';
            
            // Check if we have a seat panel open (should have prices and deck info)
            const hasSeatPanel = pageText.includes('₹') && 
                                (pageText.toLowerCase().includes('deck') || 
                                 pageText.toLowerCase().includes('select seat'));
            
            if (!hasSeatPanel) {
                return { found_panel: false, error: 'No seat panel detected on page' };
            }
            
            // Count sold seats from entire visible page
            soldCount = (pageText.match(/\bSold\b/gi) || []).length;
            
            // Extract all prices from the page - filter to reasonable bus price range
            const priceMatches = pageText.match(/₹\s*([0-9,]+)/g) || [];
            priceMatches.forEach(pm => {
                const val = parseInt(pm.replace(/[₹,\s]/g, ''));
                if (val > 100 && val < 10000) {
                    allPrices.push(val);
                }
            });
            
            // Find seat elements with aria-labels (these have structured info)
            const seatElements = document.querySelectorAll('[aria-label*="seat"], [aria-label*="Seat"]');
            
            seatElements.forEach((el, idx) => {
                const ariaLabel = el.getAttribute('aria-label') || '';
                const ariaLower = ariaLabel.toLowerCase();
                
                // Skip sold/booked seats
                if (ariaLower.includes('sold') || ariaLower.includes('booked')) {
                    return;
                }
                
                // Extract price from aria-label: "seat number L1, price 1099"
                const priceMatch = ariaLabel.match(/price\s*(\d+)/i);
                if (priceMatch) {
                    const price = parseInt(priceMatch[1]);
                    if (price > 100 && price < 10000) {
                        // Extract seat number
                        const seatNumMatch = ariaLabel.match(/seat number\s*([A-Z0-9]+)/i);
                        const seatId = seatNumMatch ? seatNumMatch[1] : `S${idx + 1}`;
                        
                        availableSeats.push({
                            seat_id: seatId,
                            price: price,
                            deck: ariaLower.includes('upper') ? 'upper' : 'lower',
                            is_window: ariaLower.includes('window')
                        });
                    }
                }
            });
            
            // If no structured seats found, create pseudo-seats from price counts
            if (availableSeats.length === 0 && allPrices.length > 0) {
                // Estimate: each unique price appears roughly equal times
                const uniquePrices = [...new Set(allPrices)];
                const avgPerPrice = Math.floor(allPrices.length / uniquePrices.length);
                
                uniquePrices.forEach((price, idx) => {
                    for (let i = 0; i < Math.min(avgPerPrice, allPrices.filter(p => p === price).length); i++) {
                        availableSeats.push({
                            seat_id: `S${idx * avgPerPrice + i + 1}`,
                            price: price,
                            deck: 'lower',
                            is_window: false
                        });
                    }
                });
            }
            
            return {
                found_panel: true,
                available_seats: availableSeats,
                sold_count: soldCount,
                all_prices: [...new Set(allPrices)],
                total_price_count: allPrices.length
            };
        }
        """)
        
        if result.get("found_panel"):
            available_seats = result.get("available_seats", [])
            sold_count = result.get("sold_count", 0)
            all_prices = result.get("all_prices", [])
            
            seats_data["available_seats_data"] = available_seats
            seats_data["available_seats"] = len(available_seats) if available_seats else result.get("total_price_count", 0)
            seats_data["sold_seats"] = sold_count
            seats_data["total_seats"] = seats_data["available_seats"] + sold_count
            seats_data["all_prices"] = all_prices
            
            if all_prices:
                seats_data["price_range"]["min"] = min(all_prices)
                seats_data["price_range"]["max"] = max(all_prices)
        
    except Exception as e:
        print(f" [ERR:{str(e)[:20]}]", end="")
    
    return seats_data


async def extract_boarding_dropping_points(page):
    """Extract boarding and dropping points from the panel."""
    boarding = []
    dropping = []
    price_varies = False
    
    try:
        # Click on Boarding point tab if exists
        try:
            await page.click("text=Boarding point", timeout=2000)
            await asyncio.sleep(1)
            
            bp_text = await page.evaluate("""
            () => {
                const points = [];
                const elements = document.querySelectorAll('[class*="boarding"], [class*="Boarding"], [class*="pickup"], [class*="bpdp"]');
                elements.forEach(el => {
                    const text = (el.innerText || '').trim();
                    if (text.length > 5 && text.length < 150 && text.includes(':')) {
                        points.push(text.substring(0, 100));
                    }
                });
                return points.slice(0, 10);
            }
            """)
            boarding = bp_text
        except:
            pass
        
        # Click on Dropping point tab if exists
        try:
            await page.click("text=Dropping point", timeout=2000)
            await asyncio.sleep(1)
            
            dp_result = await page.evaluate("""
            () => {
                const points = [];
                const prices = [];
                const elements = document.querySelectorAll('[class*="drop"], [class*="Drop"], [class*="destination"], [class*="bpdp"]');
                elements.forEach(el => {
                    const text = (el.innerText || '').trim();
                    if (text.length > 5 && text.length < 150) {
                        points.push(text.substring(0, 100));
                        const priceMatch = text.match(/₹\s*([0-9,]+)/);
                        if (priceMatch) {
                            prices.push(parseInt(priceMatch[1].replace(/,/g, '')));
                        }
                    }
                });
                const uniquePrices = [...new Set(prices)];
                return { points: points.slice(0, 10), priceVaries: uniquePrices.length > 1 };
            }
            """)
            dropping = dp_result.get("points", [])
            price_varies = dp_result.get("priceVaries", False)
        except:
            pass
        
    except Exception as e:
        pass
    
    return boarding, dropping, price_varies


async def close_seat_panel(page):
    """Close the seat selection panel."""
    try:
        # Method 1: Press Escape
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)
        
        # Method 2: Click close button
        close_selectors = [
            '[class*="close"]',
            '[class*="Close"]',
            'button[aria-label="Close"]',
            '.ic-close',
            '[class*="dismiss"]'
        ]
        
        for selector in close_selectors:
            try:
                close_btn = page.locator(selector).first
                if await close_btn.is_visible():
                    await close_btn.click(timeout=1000)
                    break
            except:
                continue
        
        # Method 3: Click X button by text
        try:
            await page.click("text=×", timeout=1000)
        except:
            pass
            
        await asyncio.sleep(1)
        
    except Exception as e:
        pass


# ============================================================================
# MAIN SCRAPING LOGIC
# ============================================================================

async def scrape_route(page, from_city, to_city, travel_date, target_day):
    """Scrape all buses for the configured operator on a specific route and date."""
    buses = []
    
    try:
        # 1. Navigate to operator page
        print(f"    🌐 Loading page...", end="", flush=True)
        await page.goto(OPERATOR_URL, timeout=30000)
        await asyncio.sleep(3)
        
        try:
            await page.keyboard.press("Escape")
        except:
            pass
        
        # 2. FROM city
        print(" Cities...", end="", flush=True)
        src_input = page.locator("#txtSource")
        await src_input.click()
        await asyncio.sleep(0.3)
        await src_input.fill("")
        await src_input.type(from_city, delay=100)
        await asyncio.sleep(2)
        
        # Click exact match
        await page.evaluate("""
        (city) => {
            const items = document.querySelectorAll('li');
            for (let item of items) {
                if (!item || !item.innerText) continue;
                const text = item.innerText.toLowerCase().trim();
                if ((text === city || text.startsWith(city + ' (')) && !text.includes(',')) {
                    item.click();
                    return;
                }
            }
        }
        """, from_city.lower())
        await asyncio.sleep(0.5)
        
        # 3. TO city
        dest_input = page.locator("#txtDestination")
        await dest_input.click()
        await asyncio.sleep(0.3)
        await dest_input.fill("")
        await dest_input.type(to_city, delay=100)
        await asyncio.sleep(2)
        
        await page.evaluate("""
        (city) => {
            const items = document.querySelectorAll('li');
            for (let item of items) {
                if (!item || !item.innerText) continue;
                const text = item.innerText.toLowerCase().trim();
                if ((text === city || text.startsWith(city + ' (')) && !text.includes(',')) {
                    item.click();
                    return;
                }
            }
        }
        """, to_city.lower())
        await asyncio.sleep(0.5)
        
        # 4. Calendar
        print(" Date...", end="", flush=True)
        await page.locator("#txtOnwardCalendar").click()
        await asyncio.sleep(1)
        
        await page.evaluate("""
        (day) => {
            const cells = document.querySelectorAll('td, div, span');
            for (let cell of cells) {
                if (!cell || !cell.innerText) continue;
                if (cell.innerText.trim() === day) {
                    if (cell.className && (cell.className.includes('Day') || cell.className.includes('day'))) {
                        cell.click();
                        return;
                    }
                }
            }
        }
        """, target_day)
        await asyncio.sleep(0.5)
        
        # 5. Search
        print(" Search...", end="", flush=True)
        try:
            await page.click("text=SEARCH BUSES", timeout=2000)
        except:
            try:
                await page.locator(".D120_search_btn_v2").click(timeout=2000)
            except:
                await page.evaluate("""
                () => {
                    const btns = document.querySelectorAll('button, a');
                    for (let btn of btns) {
                        if (btn && btn.innerText && btn.innerText.includes('SEARCH')) {
                            btn.click();
                            return;
                        }
                    }
                }
                """)
        
        await asyncio.sleep(5)
        print(" ✓", flush=True)
        
        # Check URL
        url = page.url
        if "bus-tickets" not in url and "search" not in url:
            print(f"    ⚠️ Search failed")
            return buses
        
        if "kovilpatti" in url.lower() or "puram" in url.lower():
            print(f"    ⚠️ Wrong city")
            return buses
        
        # 6. Apply operator filter using dynamic operator name
        print(f"    🔍 Applying {OPERATOR_NAME} filter...", end="", flush=True)
        
        await asyncio.sleep(3)  # Wait for page to fully load
        
        # Dismiss any popups first
        try:
            await page.keyboard.press("Escape")
        except:
            pass
        await asyncio.sleep(1)
        
        # Get operator keywords for matching (split slug by dash)
        # e.g., "vignesh-tat" -> ["vignesh", "tat"]
        operator_keywords = OPERATOR_SLUG.lower().replace("-", " ").split()
        primary_keyword = operator_keywords[0] if operator_keywords else ""
        
        # Click on the "Bus Operator" filter in the sidebar to expand it
        filter_applied = await page.evaluate("""
        (operatorKeywords) => {
            // Split keywords for matching
            const keywords = operatorKeywords.split(' ');
            const primaryKeyword = keywords[0] || '';
            
            // Method 1: Find operator in the filter list and click it
            const checkboxes = document.querySelectorAll('input[type="checkbox"], [class*="checkbox"], [class*="Checkbox"]');
            for (const cb of checkboxes) {
                const parent = cb.closest('label, div, li');
                if (parent) {
                    const text = (parent.innerText || '').toLowerCase();
                    // Match if text contains any of the operator keywords
                    const matches = keywords.some(kw => text.includes(kw));
                    if (matches) {
                        cb.click();
                        return { success: true, method: 'checkbox' };
                    }
                }
            }
            
            // Method 2: Look for operator name as clickable element
            const allElements = document.querySelectorAll('label, span, div, li');
            for (const el of allElements) {
                const text = (el.innerText || '').toLowerCase().trim();
                const matches = keywords.some(kw => text.includes(kw));
                if (matches && text.length < 50) {
                    // Check if it's in the filter area (left side)
                    const rect = el.getBoundingClientRect();
                    if (rect.left < 400 && rect.width < 300) {  // Filter is on left side
                        el.click();
                        return { success: true, method: 'element', text: text };
                    }
                }
            }
            
            // Method 3: Use filter search input if available
            const searchInputs = document.querySelectorAll('input[type="text"][placeholder*="search"], input[type="search"]');
            for (const input of searchInputs) {
                const rect = input.getBoundingClientRect();
                if (rect.left < 400) {  // In filter area
                    input.value = primaryKeyword;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    return { success: true, method: 'search' };
                }
            }
            
            return { success: false, error: 'Could not find operator filter' };
        }
        """, " ".join(operator_keywords))
        
        if filter_applied.get("success"):
            print(f" ✓ (via {filter_applied.get('method')})", flush=True)
            await asyncio.sleep(3)  # Wait for filter to apply
        else:
            print(f" ⚠️ Filter not found, scrolling to load buses...", flush=True)
            
            # Comprehensive scrolling to load ALL buses (up to 50 scrolls for routes with 350+ buses)
            last_bus_count = 0
            no_change_count = 0
            
            for scroll_num in range(50):  # Increased from 15 to 50
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(0.8)
                
                # Check current bus count
                current_count = await page.evaluate("""
                () => document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]').length
                """)
                
                # Check for end of list indicator
                end_of_list = await page.evaluate("""
                () => {
                    const text = document.body.innerText.toLowerCase();
                    return text.includes('end of list') || text.includes('no more buses');
                }
                """)
                
                if end_of_list:
                    print(f" (end at scroll {scroll_num+1}, {current_count} buses)", end="")
                    break
                
                if current_count == last_bus_count:
                    no_change_count += 1
                    if no_change_count >= 4:  # Stop if no new buses after 4 scrolls
                        print(f" (no more at scroll {scroll_num+1}, {current_count} buses)", end="")
                        break
                else:
                    no_change_count = 0
                
                last_bus_count = current_count
            
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)
        
        # 7. Now get all bus cards (should be only operator's buses if filter worked)
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        
        # Count visible operator buses using dynamic keywords
        bus_info = await page.evaluate("""
        (operatorKeywords) => {
            const keywords = operatorKeywords.split(' ');
            const busCards = document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]');
            let operatorCount = 0;
            let totalCount = busCards.length;
            
            busCards.forEach((card) => {
                const text = (card.innerText || '').toLowerCase();
                // Check if any keyword matches
                const matches = keywords.some(kw => text.includes(kw));
                if (matches) {
                    operatorCount++;
                }
            });
            
            return { total: totalCount, operator: operatorCount };
        }
        """, " ".join(operator_keywords))
        
        print(f"    🚌 Found {bus_info.get('operator', 0)} {OPERATOR_NAME} buses (total visible: {bus_info.get('total', 0)})")
        
        if bus_info.get('operator', 0) == 0:
            print(f"    ℹ️ No {OPERATOR_NAME} buses found on this route")
            return buses
        
        # 8. Process buses - iterate through ALL visible bus cards
        # and only scrape the ones that belong to the configured operator
        total_visible = bus_info.get('total', 0)
        max_to_check = min(total_visible, 50)  # Check up to 50 buses
        operator_scraped = 0
        max_operator = 10  # Max operator buses to scrape per route/date
        
        for bus_idx in range(max_to_check):
            if operator_scraped >= max_operator:
                break
            
            # Check if this bus card belongs to the operator
            is_operator_bus = await page.evaluate("""
            (data) => {
                const idx = data.idx;
                const keywords = data.keywords.split(' ');
                const cards = document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]');
                if (idx >= cards.length) return false;
                const text = (cards[idx].innerText || '').toLowerCase();
                // Match if any keyword is present
                return keywords.some(kw => text.includes(kw));
            }
            """, {"idx": bus_idx, "keywords": " ".join(operator_keywords)})
            
            if not is_operator_bus:
                continue  # Skip non-operator buses
            
            operator_scraped += 1
            print(f"\n    🔍 {OPERATOR_NAME} Bus {operator_scraped} (card #{bus_idx})")
            
            try:
                # First scroll to the bus card
                await page.evaluate("""
                (idx) => {
                    const cards = document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]');
                    if (cards[idx]) {
                        cards[idx].scrollIntoView({ behavior: 'instant', block: 'center' });
                    }
                }
                """, bus_idx)
                await asyncio.sleep(1)
                
                # Use PLAYWRIGHT's native click - much more reliable!
                # Find View Seats buttons and click the one at this index
                view_seats_btns = page.locator('[class*="viewSeats"], button:has-text("View Seats"), button:has-text("VIEW SEATS")')
                btn_count = await view_seats_btns.count()
                
                clicked = False
                
                # Try to match the button index to the bus index
                # Since not all buses have View Seats, we need to find the nth Vignesh bus's button
                if bus_idx < btn_count:
                    try:
                        btn = view_seats_btns.nth(bus_idx)
                        await btn.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        await btn.click(force=True, timeout=5000)
                        clicked = True
                    except Exception as e:
                        print(f" (click1 failed: {str(e)[:15]})", end="")
                
                # Fallback: try clicking the Nth button matching current vignesh count
                if not clicked and vignesh_scraped <= btn_count:
                    try:
                        btn = view_seats_btns.nth(vignesh_scraped - 1)
                        await btn.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)
                        await btn.click(force=True, timeout=5000)
                        clicked = True
                    except Exception as e:
                        print(f" (click2 failed: {str(e)[:15]})", end="")
                
                # Last resort: JavaScript click
                if not clicked:
                    clicked = await page.evaluate("""
                    (idx) => {
                        const cards = document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]');
                        if (idx >= cards.length) return false;
                        const card = cards[idx];
                        const btn = card.querySelector('[class*="viewSeats"], button');
                        if (btn) { btn.click(); return true; }
                        return false;
                    }
                    """, bus_idx)
                
                if not clicked:
                    print(f"      ⚠️ Could not click View Seats")
                    continue
                
                print(f"      👆 Clicked View Seats... ✓", flush=True)
                
                # CRITICAL: Wait longer for seat panel to fully render
                await asyncio.sleep(5)
                
                # Extract seat data
                print(f"      🪑 Extracting seats...", end="", flush=True)
                seats_data = await extract_seats_from_panel(page)
                avail = seats_data.get("available_seats", 0)
                sold = seats_data.get("sold_seats", 0)
                prices = seats_data.get("all_prices", [])
                print(f" {avail} available, {sold} sold, prices: {prices[:5]}{'...' if len(prices) > 5 else ''}", flush=True)
                
                # Get bus card text for metadata
                bus_card_text = await page.evaluate("""
                (idx) => {
                    const cards = document.querySelectorAll('[class*="bus-item"], [class*="tuple"], li[class*="row-sec"]');
                    if (cards[idx]) return cards[idx].innerText.substring(0, 300);
                    return '';
                }
                """, bus_idx)
                
                dep, arr = get_times(bus_card_text)
                base_price = parse_price(bus_card_text)
                
                bus_type = ""
                for pattern in ["A/C Sleeper", "A/C Seater", "Sleeper", "Seater", "2+1"]:
                    if pattern.lower() in bus_card_text.lower():
                        bus_type = pattern
                        break
                
                # Build bus data
                bus_data = {
                    "bus_id": gen_bus_id(),
                    "operator": OPERATOR_NAME.upper(),
                    "bus_type": bus_type,
                    "departure_time": dep,
                    "arrival_time": arr,
                    "route": {
                        "from_city": from_city,
                        "to_city": to_city,
                        "travel_date": travel_date
                    },
                    "seats": seats_data,
                    "base_price": base_price
                }
                
                buses.append(bus_data)
                
                # Close panel
                print(f"      ❌ Closing panel...", end="", flush=True)
                await close_seat_panel(page)
                await asyncio.sleep(2)
                print(" ✓", flush=True)
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:40]}")
                try:
                    await close_seat_panel(page)
                except:
                    pass
                continue
    
    except Exception as e:
        print(f"    ❌ Route error: {str(e)[:40]}")
    
    return buses


async def run_scraper():
    """Main scraper function with dynamic route discovery and verification."""
    global _session_id, _bus_counter, _expected_buses_per_route
    
    # Reset session variables for each run
    _session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    _bus_counter = 0
    _expected_buses_per_route = {}
    
    # Get new output filenames for this run
    output_files = get_output_filenames()
    json_file = output_files["json"]
    
    all_data = {
        "scrape_sessions": []
    }
    
    session = {
        "session_id": f"RB_{_session_id}",
        "scraped_at": datetime.now().isoformat(),
        "operator": OPERATOR_NAME,
        "operator_url": OPERATOR_URL,
        "buses": [],
        "expected_routes": {},
        "verification_report": None
    }
    
    # Track scraped buses per route for verification
    scraped_buses_per_route = {}
    
    print(f"\n🕐 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output file: {json_file}")
    
    async with async_playwright() as pw:
        browser, page = await create_browser(pw)
        
        try:
            # Step 1: Discover routes from operator page
            discovered_routes = await discover_routes_from_operator_page(page)
            
            if not discovered_routes:
                print("\n⚠️ No routes discovered from operator page!")
                print("   Check if the operator page structure has changed.")
                await browser.close()
                return all_data, json_file
            
            # Store expected routes in session
            session["expected_routes"] = dict(_expected_buses_per_route)
            
            # Step 2: Scrape each discovered route
            for from_city, to_city, expected_count in discovered_routes:
                route_key = f"{from_city}->{to_city}"
                scraped_buses_per_route[route_key] = 0
                
                print(f"\n{'='*50}")
                print(f"📍 Route: {from_city} → {to_city} (Expected: {expected_count} buses)")
                print(f"{'='*50}")
                
                for day in range(DAYS_AHEAD):
                    target_date = datetime.now() + timedelta(days=day + 1)
                    date_str = target_date.strftime("%d-%b-%Y")
                    target_day = str(target_date.day)
                    
                    print(f"\n  📅 Date: {date_str}")
                    
                    route_buses = await scrape_route(page, from_city, to_city, date_str, target_day)
                    session["buses"].extend(route_buses)
                    
                    # Update scraped count for verification
                    scraped_buses_per_route[route_key] += len(route_buses)
                    
                    print(f"  ✅ Scraped {len(route_buses)} buses (Total for route: {scraped_buses_per_route[route_key]})")
                    
                    # Save after each date
                    all_data["scrape_sessions"] = [session]
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(all_data, f, indent=2, ensure_ascii=False)
                    print(f"  📁 Saved: {json_file}")
            
            # Step 3: Verification - Compare scraped vs expected
            verification_report = verify_scraped_data(scraped_buses_per_route)
            session["verification_report"] = verification_report
            
            # Save final data with verification report
            all_data["scrape_sessions"] = [session]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"\n📁 Final data saved to: {json_file}")
        
        finally:
            await browser.close()
    
    return all_data, json_file


def run_scheduled_scraper():
    """
    Run the scraper on a schedule.
    
    - Runs immediately on start
    - Then runs every SCRAPE_INTERVAL_HOURS
    - Stops after TOTAL_DURATION_DAYS
    """
    import time
    
    start_time = datetime.now()
    end_time = start_time + timedelta(days=TOTAL_DURATION_DAYS)
    interval_seconds = SCRAPE_INTERVAL_HOURS * 3600  # Convert hours to seconds
    
    run_count = 0
    
    print(f"\n{'='*60}")
    print(f"📅 SCHEDULED SCRAPING STARTED")
    print(f"{'='*60}")
    print(f"   Start Time:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End Time:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Interval:       Every {SCRAPE_INTERVAL_HOURS} hour(s)")
    print(f"   Total Duration: {TOTAL_DURATION_DAYS} day(s)")
    print(f"   Operator:       {OPERATOR_NAME}")
    print(f"   Output Folder:  {OUTPUT_FOLDER}")
    print(f"{'='*60}")
    
    while datetime.now() < end_time:
        run_count += 1
        current_time = datetime.now()
        time_remaining = end_time - current_time
        
        print(f"\n{'#'*60}")
        print(f"# RUN #{run_count}")
        print(f"# Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Time Remaining: {time_remaining}")
        print(f"{'#'*60}")
        
        try:
            # Run the scraper
            result, output_file = asyncio.run(run_scraper())
            print(f"\n✅ Run #{run_count} completed successfully!")
            print(f"   Output: {output_file}")
        except Exception as e:
            print(f"\n❌ Run #{run_count} failed with error: {str(e)}")
        
        # Check if we should continue
        if datetime.now() >= end_time:
            print(f"\n🏁 Scheduled duration ({TOTAL_DURATION_DAYS} days) completed!")
            break
        
        # Calculate next run time
        next_run = current_time + timedelta(seconds=interval_seconds)
        if next_run >= end_time:
            print(f"\n🏁 Next run would exceed duration. Stopping scheduler.")
            break
        
        # Wait until next run
        wait_seconds = (next_run - datetime.now()).total_seconds()
        if wait_seconds > 0:
            print(f"\n⏳ Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Waiting {wait_seconds/3600:.2f} hours...")
            
            # Sleep in smaller intervals to allow for keyboard interrupt
            sleep_interval = 60  # Check every minute
            while wait_seconds > 0 and datetime.now() < end_time:
                sleep_time = min(sleep_interval, wait_seconds)
                time.sleep(sleep_time)
                wait_seconds -= sleep_time
    
    print(f"\n{'='*60}")
    print(f"📊 SCHEDULING COMPLETE")
    print(f"{'='*60}")
    print(f"   Total Runs:     {run_count}")
    print(f"   Started:        {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Ended:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output Folder:  {OUTPUT_FOLDER}")
    print(f"{'='*60}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for --single-run flag (used by GitHub Actions)
    single_run = "--single-run" in sys.argv
    
    # Try to import export and database modules
    try:
        from export import export_all
        EXPORT_ENABLED = True
    except ImportError:
        EXPORT_ENABLED = False
        print("⚠️ export module not found - will only save JSON")
    
    try:
        from database import save_scrape_run, SAVE_TO_DB
        DB_ENABLED = SAVE_TO_DB
    except ImportError:
        DB_ENABLED = False
        print("⚠️ database module not found - database saving disabled")
    
    print(f"""
    ╔════════════════════════════════════════════════════════╗
    ║  REDBUS OPERATOR SCRAPER v3.4                          ║
    ║  Fully Autonomous + Multi-Format Export                ║
    ║  Dynamic Route Discovery + Neon DB Support             ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    print(f"📌 Operator:        {OPERATOR_NAME}")
    print(f"🔗 URL:             {OPERATOR_URL}")
    print(f"📁 Output Folder:   {OUTPUT_FOLDER}")
    print(f"📄 Export Formats:  JSON" + (" + CSV + Excel" if EXPORT_ENABLED else ""))
    print(f"🗄️  Database:        {'Enabled' if DB_ENABLED else 'Disabled'}")
    
    if single_run:
        print(f"\n🔄 Mode: SINGLE RUN (for GitHub Actions)")
        print(f"📅 Scraping {DAYS_AHEAD} days ahead")
        
        try:
            # Run single scrape
            result, json_file = asyncio.run(run_scraper())
            
            # Export to all formats
            if EXPORT_ENABLED and result.get("scrape_sessions"):
                print(f"\n📤 Exporting to all formats...")
                export_all(result, OUTPUT_FOLDER)
            
            # Save to database
            if DB_ENABLED and result.get("scrape_sessions"):
                print(f"\n🗄️  Saving to Neon database...")
                save_scrape_run(result)
            
            print(f"\n✅ Single run completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)
    else:
        print(f"\n⏱️  Mode: SCHEDULED (every {SCRAPE_INTERVAL_HOURS} hour(s))")
        print(f"📅 Duration:        {TOTAL_DURATION_DAYS} day(s)")
        
        try:
            run_scheduled_scraper()
        except KeyboardInterrupt:
            print("\n\n⛔ Stopped by user")
    
    print(f"\n✅ All done! Check {OUTPUT_FOLDER}/ for scraped data files.")


