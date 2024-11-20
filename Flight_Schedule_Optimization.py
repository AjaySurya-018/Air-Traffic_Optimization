import sqlite3
import pandas as pd
from scipy.optimize import minimize
import streamlit as st

# Initialize the database and store CSV data in SQLite
def initialize_database(csv_file, db_file="flights.db"):
    conn = sqlite3.connect(db_file)
    data = pd.read_csv(csv_file)
    
    # Create flights table
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flights (
        FlightNum TEXT,
        Origin TEXT,
        Dest TEXT,
        CRSElapsedTime REAL,
        ActualElapsedTime REAL,
        DepDelay REAL,
        ArrDelay REAL
    )
    """)
    conn.commit()
    
    # Insert data into the table
    data.to_sql("flights", conn, if_exists="replace", index=False)
    conn.close()

# Optimization function
def optimize_schedule(buffer_min, buffer_max, airport=None, airport_type="origin", db_file="flights.db"):
    conn = sqlite3.connect(db_file)
    query = "SELECT * FROM flights"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Filter data for selected airport
    if airport:
        if airport_type == "origin":
            df = df[df["Origin"] == airport]
        elif airport_type == "destination":
            df = df[df["Dest"] == airport]
    
    # Ensure there is data to optimize
    if df.empty:
        return None

    # Optimization variables (CSR adjustments)
    adjustments = [0] * len(df)

    # Objective function: minimize total arrival delay
    def objective(adjustments):
        total_delay = 0
        for i, adj in enumerate(adjustments):
            new_csr = df["CRSElapsedTime"].iloc[i] + adj
            actual_time = df["ActualElapsedTime"].iloc[i]
            arr_delay = max(0, actual_time - new_csr)
            total_delay += arr_delay
        return total_delay

    # Bounds for adjustments
    bounds = [(buffer_min, buffer_max) for _ in range(len(df))]
    
    # Solve the optimization problem
    result = minimize(objective, adjustments, bounds=bounds)
    df["AdjustedCRS"] = df["CRSElapsedTime"] + result.x
    df["NewArrDelay"] = df["AdjustedCRS"] - df["ActualElapsedTime"]
    return df

# Streamlit app
def main():
    st.title("Flight Schedule Optimization")
    st.write("Optimize flight delays by adjusting CSR elapsed times.")

    # Initialize database
    csv_file = "data/Final_data.csv"  # Replace with your CSV file
    initialize_database(csv_file)

    # Connect to SQLite
    conn = sqlite3.connect("flights.db")
    df = pd.read_sql_query("SELECT * FROM flights", conn)
    conn.close()

    # Display original data
    st.write("Original Data:", df)

    # Sidebar inputs
    st.sidebar.header("Optimization Settings")
    buffer_min = st.sidebar.slider("Minimum Adjustment (minutes)", -30, 0, -10)
    buffer_max = st.sidebar.slider("Maximum Adjustment (minutes)", 0, 30, 10)

    # Dropdown for filtering by airport
    airports = pd.concat([df["Origin"], df["Dest"]]).unique()
    selected_airport = st.sidebar.selectbox("Select Airport", ["All"] + list(airports))
    airport_type = st.sidebar.radio("Airport Type", ["Origin", "Destination"])

    # Optimize button
    if st.button("Optimize Schedule"):
        airport_filter = selected_airport if selected_airport != "All" else None
        result_df = optimize_schedule(buffer_min, buffer_max, airport_filter, airport_type.lower())

        if result_df is not None:
            st.write(f"Optimized Data for {selected_airport} ({airport_type}):", result_df)

            # Display metrics
            total_delay_before = df["ArrDelay"].sum()
            total_delay_after = result_df["NewArrDelay"].sum()
            st.write(f"Total Arrival Delay Before Optimization: {total_delay_before} minutes")
            st.write(f"Total Arrival Delay After Optimization: {total_delay_after} minutes")
        else:
            st.write("No data available for the selected airport.")

if __name__ == "__main__":
    main()
