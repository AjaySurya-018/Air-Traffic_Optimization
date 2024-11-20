import sqlite3
import pandas as pd
import streamlit as st
from prophet import Prophet
import matplotlib.pyplot as plt

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
    # Prepare the data for Prophet
    route_data['ds'] = pd.to_datetime(route_data['Year'], format='%Y')
    route_data = route_data.rename(columns={'AvgArrDelay': 'y'})
    route_data = route_data[['ds', 'y']]

    # Initialize and train the Prophet model
    model = Prophet(yearly_seasonality=True, interval_width=0.95)
    model.fit(route_data)

    # Create a future dataframe for 2009-2014
    future_years = pd.date_range(start='2008', end='2015', freq='Y')
    future = pd.DataFrame({'ds': future_years})
    forecast = model.predict(future)

    # Return the forecasted results and the model for plotting
    return forecast, model

# Streamlit UI
st.title("Flight Delay Forecasting")

# Fetch distinct origins and destinations for dropdown menus
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