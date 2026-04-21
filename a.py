import sqlite3
import pandas as pd
import time

def measure_query(conn, query, iterations=10):
    """Execute the query repeatedly and return the average execution time."""
    cursor = conn.cursor()
    times = []
    for _ in range(iterations):
        start = time.time()
        cursor.execute(query)
        cursor.fetchall() 
        end = time.time()
        times.append(end - start)
    return sum(times) / iterations

def run_full_dataset_tests(db_path="premier_league.db", csv_path="matches.csv"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    matches_df = pd.read_csv(csv_path, parse_dates=["Date"])
    
    matches_df.to_sql('matches', conn, if_exists='replace', index=False)
    
    query_full = "SELECT * FROM matches WHERE Date BETWEEN '2020-01-01' AND '2024-12-31';"
    
    # 1. Drop indexes if they exist, then measure query time without indexes
    cursor.execute("DROP INDEX IF EXISTS idx_date")
    cursor.execute("DROP INDEX IF EXISTS idx_team")
    cursor.execute("DROP INDEX IF EXISTS idx_opponent")
    conn.commit()
    
    avg_time_no_index = measure_query(conn, query_full, iterations=10)
    print("Average query time without indexes: {:.4f} seconds".format(avg_time_no_index))
    
    # 2. Create indexes on key columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON matches(Date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_team ON matches(Team)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_opponent ON matches(Opponent)")
    conn.commit()
    
    # 3. Measure query time with indexes
    avg_time_with_index = measure_query(conn, query_full, iterations=10)
    print("Average query time with indexes:    {:.4f} seconds".format(avg_time_with_index))
    
    # 4. Calculate and print the performance improvement percentage
    improvement = ((avg_time_no_index - avg_time_with_index) / avg_time_no_index) * 100
    print("Performance improvement: {:.2f}%".format(improvement))
    
    conn.close()

if __name__ == "__main__":
    run_full_dataset_tests()
