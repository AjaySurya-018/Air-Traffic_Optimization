import pandas as pd
import sqlite3
import pulp
import streamlit as st

# Create a SQLite connection
conn = sqlite3.connect("flights.db")
cur = conn.cursor()

# Load CSV into SQLite for simplicity
df = pd.read_csv("data/Final_data.csv")
df.to_sql("final_data", conn, if_exists="replace", index=False)

# Load the data from SQLite
query = "SELECT * FROM final_data"
data = pd.read_sql_query(query, conn)

# Streamlit UI
st.title("Taxi Time Optimization")

# Dropdowns for airport constraints
st.sidebar.header("Filters")
source_airport = st.sidebar.selectbox("Select Source Airport", data['Origin'].unique())
dest_airport = st.sidebar.selectbox("Select Destination Airport", data['Dest'].unique())
max_gates = st.sidebar.number_input("Number of Available Gates", min_value=1, max_value=10, value=5)

# Filter data based on user input
filtered_data = data[(data['Origin'] == source_airport) & (data['Dest'] == dest_airport)]

# Show filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Optimization Problem
if st.button("Run Optimization"):
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Prepare data for optimization
        flights = filtered_data['FlightNum'].unique()
        gates = range(max_gates)

        # TaxiIn and TaxiOut times
        taxi_in = filtered_data.set_index('FlightNum')['TaxiIn'].to_dict()
        taxi_out = filtered_data.set_index('FlightNum')['TaxiOut'].to_dict()

        # Decision Variables
        flight_gate = pulp.LpVariable.dicts("GateAssignment", ((f, g) for f in flights for g in gates), cat="Binary")

        # Optimization Model
        model = pulp.LpProblem("TaxiTimeOptimization", pulp.LpMinimize)

        # Objective Function: Minimize TaxiIn + TaxiOut
        model += pulp.lpSum(
            [
                flight_gate[f, g] * (taxi_in[f] + taxi_out[f])
                for f in flights
                for g in gates
            ]
        )

        # Constraints
        # 1. Each flight assigned to one gate
        for f in flights:
            model += pulp.lpSum(flight_gate[f, g] for g in gates) == 1

        # 2. Gate capacity: A gate can only handle one flight at a time
        for g in gates:
            model += pulp.lpSum(flight_gate[f, g] for f in flights) <= 1

        # Solve the optimization problem
        model.solve()

        # Results
        st.subheader("Optimization Results")
        if pulp.LpStatus[model.status] == "Optimal":
            result_data = []
            for f in flights:
                for g in gates:
                    if pulp.value(flight_gate[f, g]) == 1:
                        result_data.append(
                            {
                                "FlightNum": f,
                                "AssignedGate": g,
                                "TaxiInTime": taxi_in[f],
                                "TaxiOutTime": taxi_out[f],
                            }
                        )
            result_df = pd.DataFrame(result_data)
            st.write(result_df)

            # Total optimized taxi time
            total_taxi_time = result_df["TaxiInTime"].sum() + result_df["TaxiOutTime"].sum()
            st.success(f"Total Optimized Taxi Time: {total_taxi_time} minutes")
        else:
            st.error("Optimization Problem is Infeasible.")

# Close SQLite connection
conn.close()