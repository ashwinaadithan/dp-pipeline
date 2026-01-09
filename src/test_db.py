
import os
import sys
from dotenv import load_dotenv

# Load env variables from the src folder
load_dotenv(".env")

url = os.getenv("NEON_DATABASE_URL")
print(f"Connection String found: {'Yes' if url else 'No'}")
if url:
    # Mask password for display
    safe_url = url.split("@")[-1] if "@" in url else "..."
    print(f"Target Host: {safe_url}")

try:
    import psycopg
    print("Library: psycopg (v3) detected")
except ImportError:
    try:
        import psycopg2 as psycopg
        print("Library: psycopg2 (v2) detected")
    except ImportError:
        print("❌ No postgres library found!")
        sys.exit(1)

try:
    print("\n1. Connecting to database...")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://")
        
    conn = psycopg.connect(url)
    print("✅ Connection successful!")
    
    cur = conn.cursor()
    
    print("\n2. Creating test table...")
    cur.execute("CREATE TABLE IF NOT EXISTS test_connection (id SERIAL PRIMARY KEY, note TEXT, created_at TIMESTAMP DEFAULT NOW())")
    conn.commit()
    print("✅ Table created!")
    
    print("\n3. Inserting test data...")
    cur.execute("INSERT INTO test_connection (note) VALUES ('Hello from Oracle VM')")
    conn.commit()
    print("✅ Insertion successful!")
    
    print("\n4. Reading data back...")
    cur.execute("SELECT * FROM test_connection ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    print(f"✅ Read back: {row}")
    
    cur.close()
    conn.close()
    print("\n🎉 DIAGNOSIS: Database connection is WORKING!")
    
except Exception as e:
    print(f"\n❌ FAILURE: {e}")
    print("\nHints:")
    print("- Check if IP is allowed in Neon (usually allow 0.0.0.0/0)")
    print("- Check if password is correct")
    print("- Check if database name 'neondb' exists")
