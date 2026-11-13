
import sqlite3

# Connect to the project database
conn = sqlite3.connect("telemetry_audit.db")
cursor = conn.cursor()

# Display available tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("DATABASE TABLES:")
for table in tables:
    print(table[0])

# Display telemetry records
print("\nTELEMETRY RECORDS:")
cursor.execute("SELECT * FROM telemetry_log LIMIT 5")

for record in cursor.fetchall():
    print(record)

conn.close()
