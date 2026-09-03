import sqlite3
import pandas as pd
import glob
from pathlib import Path
import os

# Create data folder if it doesn't exist
os.makedirs('database', exist_ok=True)

# Step 1: Create database
db_path = 'database/New_database.db'
conn = sqlite3.connect(db_path)

# Step 2: Load ALL CSVs → tables (auto‑detects filenames)
csv_files = glob.glob('data/*.csv')
if not csv_files:
    print("No CSVs found in 'data/' folder. Please add your CSV files there and run this script again.")
else:
    for csv_path in csv_files:
        table_name = Path(csv_path).stem    
        print(f"Loading {csv_path}...")
        
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Print schema preview
        row_count = len(df)
        col_info = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
        print(f"  → {table_name}: {row_count:,} rows, {len(col_info)} columns")
    
    conn.commit()
    conn.close()
    
    # Step 3: Verify
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print(f"\n✅ Database created: {db_path}")
    print("Tables loaded:")
    for table in tables['name']:
        rows = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {table}", conn).iloc[0,0]
        print(f"  {table}: {rows:,} rows")
    
    conn.close()
