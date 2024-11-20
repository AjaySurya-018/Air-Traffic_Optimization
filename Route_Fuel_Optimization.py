import pandas as pd
import streamlit as st
from pulp import LpProblem, LpVariable, LpMinimize, lpSum

# Load dataset (simulate with CSV for this example)
@st.cache_data
def load_data():
    return pd.read_csv("data/Final_data.csv")  # Replace with your dataset file

# Filter dataset based on user inputs
def filter_routes(data, origin, dest, max_distance):
    return data[(data["Origin"] == origin) &
                (data["Dest"] == dest) &
                (data["Distance"] <= max_distance)]

# Optimization function
def optimize_routes(data, max_flights):
    # Create a problem instance
    problem = LpProblem("Route_Fuel_Optimization", LpMinimize)

    # Variables: 1 if flight is selected, 0 otherwise
    flights = data["FlightNum"].unique()
    flight_vars = {f: LpVariable(f"Flight_{f}", 0, 1, cat="Binary") for f in flights}

    # Objective: Minimize total distance
    problem += lpSum(flight_vars[f] * data[data["FlightNum"] == f]["Distance"].values[0] for f in flights)

    # Constraint: Limit the number of selected flights
    problem += lpSum(flight_vars[f] for f in flights) <= max_flights

    # Solve the problem
    problem.solve()

    # Collect optimized flights
    selected_flights = [f for f in flights if flight_vars[f].value() == 1]
    return data[data["FlightNum"].isin(selected_flights)]

# Main Streamlit App
def main():
    st.title("Airline Route Fuel Optimization")

    # Load dataset
    data = load_data()

    # User inputs
    st.sidebar.header("User Inputs")
    origin = st.sidebar.selectbox("Select Source Airport", data["Origin"].unique())
    dest = st.sidebar.selectbox("Select Destination Airport", data["Dest"].unique())
    max_distance = st.sidebar.slider("Maximum Route Distance (miles)", 0, int(data["Distance"].max()), 3000)
    max_flights = st.sidebar.number_input("Maximum Flights Allowed on Route", min_value=1, step=1, value=5)

    # Filtered data based on user inputs
    filtered_data = filter_routes(data, origin, dest, max_distance)

    # Display historical data
    st.subheader("Filtered Historical Data")
    st.write(filtered_data)

    # Perform optimization if filtered data is not empty
    if not filtered_data.empty:
        optimized_data = optimize_routes(filtered_data, max_flights)

        # Display optimization results
        st.subheader("Optimized Route Plan")
        st.write(optimized_data)

        # Summary metrics
        st.subheader("Optimization Summary")
        st.write(f"**Total Flights Selected**: {len(optimized_data)}")
        st.write(f"**Total Distance**: {optimized_data['Distance'].sum()} miles")

        # Visualize routes
        st.subheader("Route Visualization")
        st.map(optimized_data[["Origin", "Dest"]])  # Simple map plot

    else:
        st.warning("No data available for the selected filters.")

# Run the Streamlit app
if __name__ == "__main__":
    main()