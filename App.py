import sqlite3
import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt
import pulp
import os
import seaborn as sns

# Function to fetch distinct origin and destination airport codes
def fetch_airport_codes():
    conn = sqlite3.connect('flights.db')
    query = "SELECT DISTINCT Origin, Dest FROM final_data"
    data = pd.read_sql(query, conn)
    conn.close()
    origins = data['Origin'].unique().tolist()
    destinations = data['Dest'].unique().tolist()
    return origins, destinations

# Function to fetch data for the given route
def fetch_route_data(origin, dest):
    conn = sqlite3.connect('flights.db')
    query = f"""
    SELECT Year, AVG(ArrDelay) AS AvgArrDelay
    FROM final_data
    WHERE Origin = '{origin}' AND Dest = '{dest}'
    GROUP BY Year
    HAVING Year BETWEEN 2003 AND 2008
    """
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Function to prepare and train the forecasting model
def forecast_delays(route_data):
    route_data['ds'] = pd.to_datetime(route_data['Year'], format='%Y')
    route_data = route_data.rename(columns={'AvgArrDelay': 'y'})
    route_data = route_data[['ds', 'y']]
    model = Prophet(yearly_seasonality=True, interval_width=0.95)
    model.fit(route_data)
    future_years = pd.date_range(start='2008', end='2015', freq='Y')
    future = pd.DataFrame({'ds': future_years})
    forecast = model.predict(future)
    return forecast, model

# Load the data from SQLite
def load_data():
    conn = sqlite3.connect("flights.db")
    query = "SELECT * FROM final_data"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

# Attribute Descriptions
attribute_descriptions = {
    "UniqueCarrier": "The unique code of the airline carrier. Helps assess carrier performance and analyze delays.",
    "Flight Num": "The flight number. Tracks individual flight performance and correlates delays with routes.",
    "Origin": "Departure airport code. Useful for analyzing route delays.",
    "Dest": "Arrival airport code. Useful for understanding destination-related delay patterns.",
    "Distance": "Distance in miles. Useful for correlating delays with distance.",
    "DayOfWeek": "Day of the week (1 for Monday, 7 for Sunday). Highlights weekday delay patterns.",
    "DayofMonth": "Day of the month (1–31). Useful for monthly trend analysis.",
    "Month": "Month of the year (1–12). Analyzes seasonal delay trends.",
    "Year": "Year of operation. Useful for long-term trend analysis.",
    "CRSElapsedTime": "Scheduled flight duration in minutes. Compares scheduled vs. actual durations.",
    "ActualElapsedTime": "Actual flight duration in minutes. Measures deviation from scheduled time.",
    "ArrDelay": "Arrival delay in minutes. Key for passenger satisfaction and efficiency.",
    "DepDelay": "Departure delay in minutes. Analyzes initial delay impact on later segments.",
    "TaxiIn": "Taxi-in time in minutes. Highlights airport congestion.",
    "TaxiOut": "Taxi-out time in minutes. Optimizes ground operations.",
    "CarrierDelay": "Minutes delayed due to airline operations.",
    "WeatherDelay": "Minutes delayed due to weather conditions.",
    "NASDelay": "Minutes delayed by National Air System issues.",
    "SecurityDelay": "Minutes delayed by security measures.",
    "LateAircraftDelay": "Minutes delayed by late arrival of the previous aircraft.",
}

# Main Interface
st.title("Airline Traffic Analysis and Optimization Dashboard")

# Sidebar Option Selector
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a module to explore:", ["Dataset Overview", "Optimization", "Forecasting", "Analysis"])

# Load the dataset
df = load_data()

if option == "Dataset Overview":
    # Dataset Section
    st.header("Dataset Preview")
    max_rows = len(df)
    rows_to_display = st.slider("Number of rows to display", min_value=5, max_value=max_rows, value=10)
    st.dataframe(df.head(rows_to_display))

    # Attributes Section
    st.header("Attribute Descriptions")
    for attr, desc in attribute_descriptions.items():
        st.subheader(attr)
        st.write(desc)

elif option == "Optimization":
    st.title("Taxi Time Optimization")

    # Dropdowns for airport constraints
    source_airport = st.selectbox("Select Source Airport", df['Origin'].unique())
    dest_airport = st.selectbox("Select Destination Airport", df['Dest'].unique())
    max_gates = st.number_input("Number of Available Gates", min_value=1, max_value=10, value=5)

    # Filter data based on user input
    filtered_data = df[(df['Origin'] == source_airport) & (df['Dest'] == dest_airport)]

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

elif option == "Forecasting":
    st.title("Flight Delay Forecasting")

    origins, destinations = fetch_airport_codes()

    # Dropdown menus for source and destination
    origin = st.selectbox("Select Origin Airport Code", origins)
    dest = st.selectbox("Select Destination Airport Code", destinations)

    if origin and dest:
        # Fetch historical data for the selected route
        route_data = fetch_route_data(origin, dest)

        if not route_data.empty:
            # Display the historical data table
            st.subheader(f"Historical Data (2003-2008) for {origin} to {dest}")
            st.write(route_data)

            # Plot historical delays
            st.subheader("Historical Average Delays (in min) Over the Years")
            fig, ax = plt.subplots()
            ax.plot(route_data['Year'], route_data['AvgArrDelay'], marker='o')
            ax.set_title(f"Historical Average Delays for {origin} to {dest}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Arrival Delay (minutes)")
            st.pyplot(fig)

            # Forecast delays for 2009-2014
            forecast, model = forecast_delays(route_data)

            # Plot the forecast with predicted values annotated
            st.subheader("Forecasted Average Delays (in min) for next 5 years")
            fig = model.plot(forecast)
            ax = fig.gca()

            # Annotate predicted values (yhat) on the plot
            for i, row in forecast.tail(7).iterrows():
                ax.text(row['ds'], row['yhat'], f"{row['yhat']:.2f}", color='blue', fontsize=10)

            # Adjust the x-axis limits to display only 2009-2014
            ax.set_xlim(pd.Timestamp('2009-01-01'), pd.Timestamp('2014-12-31'))
            st.pyplot(fig)
        else:
            st.write(f"No historical data available for the route {origin} to {dest}.")

elif option == "Analysis":
    st.title("Analysis")

        # Connect to SQLite database
    conn = sqlite3.connect("flights.db")

    # Load data from SQLite
    query = "SELECT * FROM flights"
    data = pd.read_sql_query(query, conn)

    # Inferences corresponding to each analysis
    inferences = [
        "Carriers with higher arrival delays tend to have higher departure delays as well. This suggests operational inefficiencies or external factors that impact both arrival and departure times.",
        "Delays are highest on Sundays and lowest on Tuesdays. Arrival delays are consistently higher than departure delays on all days.",
        "Longer flight times for HA and CO could be due to factors like operating longer-haul routes or having a fleet with lower average cruising speeds.",
        "The increasing trend in arrival delays suggests potential systemic issues in airport operations or air traffic control that need to be addressed to improve overall flight punctuality.",
        "Improving aircraft turnaround times and addressing issues within the air traffic control system could have a substantial impact on reducing flight delays.",
        "Seasonal weather patterns, increased travel demand during summer, and potential operational challenges during peak travel periods might contribute to the observed seasonal variation in flight delays.",
        "Longer routes might be more susceptible to delays due to factors like increased exposure to weather disturbances, longer travel times, and higher chances of cascading delays. However, the relationship is not very strong, suggesting other factors like airport congestion, airline operations, and air traffic control play a significant role in determining delays.",
        "The variations in taxi times across airports could be attributed to factors like airport layout, air traffic congestion, ground handling efficiency, and airline operational practices. Identifying and addressing the root causes of these variations can help improve airport efficiency and reduce passenger wait times.",
        "The high frequency of flights on these routes indicates strong demand between these city pairs. This could be due to factors like business travel, tourism, or geographic proximity. Airlines might consider increasing capacity on these routes to meet the high demand.",
        "These routes are experiencing significant delays, which could be due to various factors such as weather conditions, air traffic congestion, or operational issues at specific airports. Identifying the root causes of these delays and implementing targeted solutions can help improve the reliability of these routes.",
        "Flights which have a departure delay, are also highly likely to have an arrival delay. This is clearly shown in the correlation map.",
        "Understand how consistent or unpredictable delays are for carriers, routes, or airports."
    ]

    # Streamlit app

    # Dropdown menu for selecting analysis
    options = [
        "Carrier Delays",
        "Day-wise Delays",
        "Flight Time vs Carrier",
        "Year-wise Arrival Delays",
        "Turnaround Time Impact",
        "Seasonal Delays",
        "Distance vs Delay",
        "Airport Taxi Times",
        "Top Routes by Frequency",
        "Top Routes with Delays",
        "Correlation of Delays",
        "Variability in Delays",
    ]
    selected_option = st.selectbox("Select Analysis", options)

    # Helper function to display plot and inference
    def display_analysis(plot_function, inference_index):
        plot_function()
        st.write(f"Inference: {inferences[inference_index]}")

    # 1. Carrier Delays Analysis
    if selected_option == "Carrier Delays":
        def plot_carrier_delays():
            query = """
            SELECT UniqueCarrier, AVG(ArrDelay) AS avg_arrival_delay, AVG(DepDelay) AS avg_departure_delay
            FROM flights
            GROUP BY UniqueCarrier
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.barplot(data=result, x="UniqueCarrier", y="avg_arrival_delay", color="skyblue", ax=ax, label="Arrival Delay")
            sns.barplot(data=result, x="UniqueCarrier", y="avg_departure_delay", color="orange", ax=ax, label="Departure Delay")
            ax.legend()
            ax.set_title("Average Delays by Carrier")
            st.pyplot(fig)

        display_analysis(plot_carrier_delays, 0)

    # 2. Day-wise Delays Analysis
    elif selected_option == "Day-wise Delays":
        def plot_day_wise_delays():
            query = """
            SELECT DayOfWeek, AVG(ArrDelay) AS avg_arrival_delay, AVG(DepDelay) AS avg_departure_delay
            FROM flights
            GROUP BY DayOfWeek
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.lineplot(data=result, x="DayOfWeek", y="avg_arrival_delay", marker="o", label="Arrival Delay", ax=ax)
            sns.lineplot(data=result, x="DayOfWeek", y="avg_departure_delay", marker="o", label="Departure Delay", ax=ax)
            ax.set_title("Average Delays by Day of the Week")
            st.pyplot(fig)

        display_analysis(plot_day_wise_delays, 1)

    # 3. Flight Time vs Carrier Analysis
    elif selected_option == "Flight Time vs Carrier":
        def plot_flight_time_vs_carrier():
            query = """
            SELECT UniqueCarrier, AVG(CRSElapsedTime) AS avg_crs_elapsed_time
            FROM flights
            GROUP BY UniqueCarrier
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.barplot(data=result, x="UniqueCarrier", y="avg_crs_elapsed_time", ax=ax, palette="coolwarm")
            ax.set_title("Average Flight Time by Carrier")
            st.pyplot(fig)

        display_analysis(plot_flight_time_vs_carrier, 2)

    # 4. Year-wise Arrival Delays Analysis
    elif selected_option == "Year-wise Arrival Delays":
        def plot_year_wise_arrival_delays():
            query = """
            SELECT Year, AVG(ArrDelay) AS avg_arrival_delay
            FROM flights
            GROUP BY Year
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.lineplot(data=result, x="Year", y="avg_arrival_delay", marker="o", ax=ax)
            ax.set_title("Year-wise Average Arrival Delays")
            st.pyplot(fig)

        display_analysis(plot_year_wise_arrival_delays, 3)

    # 5. Turnaround Time Impact Analysis
    elif selected_option == "Turnaround Time Impact":
        def plot_turnaround_time_impact():
            query = """
            SELECT TaxiIn + TaxiOut AS turnaround_time, AVG(ArrDelay) AS avg_arrival_delay
            FROM flights
            GROUP BY turnaround_time
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.scatterplot(data=result, x="turnaround_time", y="avg_arrival_delay", ax=ax)
            ax.set_title("Impact of Turnaround Time on Arrival Delays")
            st.pyplot(fig)

        display_analysis(plot_turnaround_time_impact, 4)

    # 6. Seasonal Delays Analysis
    elif selected_option == "Seasonal Delays":
        def plot_seasonal_delays():
            query = """
            SELECT Month, AVG(ArrDelay) AS avg_arrival_delay
            FROM flights
            GROUP BY Month
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.lineplot(data=result, x="Month", y="avg_arrival_delay", marker="o", ax=ax)
            ax.set_title("Seasonal Variation in Arrival Delays")
            st.pyplot(fig)

        display_analysis(plot_seasonal_delays, 5)

    # 7. Distance vs Delay Analysis
    elif selected_option == "Distance vs Delay":
        def plot_distance_vs_delay():
            query = """
            SELECT Distance, AVG(ArrDelay) AS avg_arrival_delay
            FROM flights
            GROUP BY Distance
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.scatterplot(data=result, x="Distance", y="avg_arrival_delay", ax=ax)
            ax.set_title("Distance vs Arrival Delays")
            st.pyplot(fig)

        display_analysis(plot_distance_vs_delay, 6)

    # 8. Airport Taxi Times Analysis
    elif selected_option == "Airport Taxi Times":
        def plot_airport_taxi_times():
            query = """
            SELECT Origin, AVG(TaxiIn + TaxiOut) AS avg_taxi_time
            FROM flights
            GROUP BY Origin
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.barplot(data=result, x="Origin", y="avg_taxi_time", ax=ax, palette="viridis")
            ax.set_title("Average Taxi Times by Airport")
            st.pyplot(fig)

        display_analysis(plot_airport_taxi_times, 7)

    # 9. Top Routes by Frequency
    elif selected_option == "Top Routes by Frequency":
        def plot_top_routes_by_frequency():
            query = """
            SELECT Origin || '-' || Dest AS route, COUNT(*) AS flight_count
            FROM flights
            GROUP BY route
            ORDER BY flight_count DESC
            LIMIT 10
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.barplot(data=result, x="flight_count", y="route", ax=ax, palette="rocket")
            ax.set_title("Top 10 Routes by Flight Frequency")
            st.pyplot(fig)

        display_analysis(plot_top_routes_by_frequency, 8)

    # 10. Top Routes with Delays
    elif selected_option == "Top Routes with Delays":
        def plot_top_routes_with_delays():
            query = """
            SELECT Origin || '-' || Dest AS route, AVG(ArrDelay) AS avg_arrival_delay
            FROM flights
            GROUP BY route
            ORDER BY avg_arrival_delay DESC
            LIMIT 10
            """
            result = pd.read_sql_query(query, conn)
            fig, ax = plt.subplots()
            sns.barplot(data=result, x="avg_arrival_delay", y="route", ax=ax, palette="magma")
            ax.set_title("Top 10 Routes with Highest Arrival Delays")
            st.pyplot(fig)

        display_analysis(plot_top_routes_with_delays, 9)

    elif selected_option == "Correlation of Delays":
        def plot_correlation_analysis():
            query = """
            SELECT Distance, ArrDelay, DepDelay, TaxiIn, TaxiOut
            FROM flights
            """
            # Fetch the data
            correlation_data = pd.read_sql_query(query, conn)

            # Compute correlation matrix
            correlation_matrix = correlation_data.corr()

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix of Flight Variables")
            st.pyplot(fig)

        display_analysis(plot_correlation_analysis, 10)

    elif selected_option == "Variability in Delays":
        def plot_variance_analysis():
            # Variance and Standard Deviation of delays grouped by UniqueCarrier
            query_carrier = """
            SELECT UniqueCarrier,
                   AVG(ArrDelay) AS avg_arrival_delay,
                   AVG(DepDelay) AS avg_departure_delay,
                   -- Variance and Standard Deviation for Arrival Delay
                   AVG((ArrDelay - (SELECT AVG(ArrDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)) * 
                       (ArrDelay - (SELECT AVG(ArrDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier))) 
                       AS var_arrival_delay,
                   SQRT(AVG((ArrDelay - (SELECT AVG(ArrDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)) * 
                            (ArrDelay - (SELECT AVG(ArrDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)))) 
                       AS std_arrival_delay,
                   -- Variance and Standard Deviation for Departure Delay
                   AVG((DepDelay - (SELECT AVG(DepDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)) * 
                       (DepDelay - (SELECT AVG(DepDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier))) 
                       AS var_departure_delay,
                   SQRT(AVG((DepDelay - (SELECT AVG(DepDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)) * 
                            (DepDelay - (SELECT AVG(DepDelay) FROM flights AS f WHERE f.UniqueCarrier = flights.UniqueCarrier)))) 
                       AS std_departure_delay
            FROM flights
            GROUP BY UniqueCarrier;
            """
            carrier_result = pd.read_sql_query(query_carrier, conn)

            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=carrier_result, x="UniqueCarrier", y="std_arrival_delay", label="Arrival Delay STD", ax=ax)
            sns.barplot(data=carrier_result, x="UniqueCarrier", y="std_departure_delay", label="Departure Delay STD", ax=ax, alpha=0.7)
            ax.set_title("Standard Deviation of Delays by UniqueCarrier")
            ax.legend()
            st.pyplot(fig)

        display_analysis(plot_variance_analysis, 11)

