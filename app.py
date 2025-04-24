
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Water Quality Monitoring Explorer")

# Upload CSV files
station_file = st.file_uploader("Upload station.csv (Part 1)", type=["csv"])
result_file = st.file_uploader("Upload narrowresult.csv (Part 2)", type=["csv"])

if station_file and result_file:
    # Load data
    station_df = pd.read_csv(station_file)
    result_df = pd.read_csv(result_file)

    # Ensure correct data types
    result_df['ActivityStartDate'] = pd.to_datetime(result_df['ActivityStartDate'], errors='coerce')
    result_df['ResultMeasureValue'] = pd.to_numeric(result_df['ResultMeasureValue'], errors='coerce')

    # Sidebar filters
    st.sidebar.header("Filters")

    # Filter by contaminant
    contaminants = result_df['CharacteristicName'].dropna().unique()
    selected_contaminant = st.sidebar.selectbox("Select a Contaminant", sorted(contaminants))

    filtered_results = result_df[result_df['CharacteristicName'] == selected_contaminant]

    # Filter by date range
    min_date = filtered_results['ActivityStartDate'].min()
    max_date = filtered_results['ActivityStartDate'].max()

    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    # Filter by value range
    min_val = float(filtered_results['ResultMeasureValue'].min())
    max_val = float(filtered_results['ResultMeasureValue'].max())
    val_range = st.sidebar.slider("Select Value Range", min_val, max_val, (min_val, max_val))

    # Apply all filters
    mask = (
        (filtered_results['ActivityStartDate'] >= pd.to_datetime(start_date)) &
        (filtered_results['ActivityStartDate'] <= pd.to_datetime(end_date)) &
        (filtered_results['ResultMeasureValue'] >= val_range[0]) &
        (filtered_results['ResultMeasureValue'] <= val_range[1])
    )
    filtered_results = filtered_results[mask]

    # Join with station data
    merged = pd.merge(
        filtered_results,
        station_df,
        how='left',
        left_on='MonitoringLocationIdentifier',
        right_on='MonitoringLocationIdentifier'
    )
    merged = merged.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])

    # Map of stations
    st.subheader("Map of Monitoring Stations")
    map_center = [merged["LatitudeMeasure"].mean(), merged["LongitudeMeasure"].mean()]
    m = folium.Map(location=map_center, zoom_start=7)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in merged.iterrows():
        popup_text = f"""
        <b>Station:</b> {row['MonitoringLocationName']}<br>
        <b>Value:</b> {row['ResultMeasureValue']}<br>
        <b>Date:</b> {row['ActivityStartDate'].date()}
        """
        folium.Marker(
            location=[row["LatitudeMeasure"], row["LongitudeMeasure"]],
            popup=popup_text,
            tooltip=row["MonitoringLocationIdentifier"]
        ).add_to(marker_cluster)

    st_data = st_folium(m, width=900, height=500)

    # Trend chart
    st.subheader(f"{selected_contaminant} Trend Over Time")
    time_series = merged[['ActivityStartDate', 'ResultMeasureValue']].sort_values(by='ActivityStartDate')

    plt.figure(figsize=(12, 6))
    plt.plot(time_series['ActivityStartDate'], time_series['ResultMeasureValue'], marker='o', linestyle='-')
    plt.xlabel("Date")
    plt.ylabel("Result Measure Value")
    plt.title(f"{selected_contaminant} Levels Across All Stations")
    plt.grid(True)
    st.pyplot(plt)

else:
    st.info("Please upload both `station.csv` and `narrowresult.csv` to begin.")
