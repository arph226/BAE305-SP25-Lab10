# BAE305-SP25-Lab10
# Laboratory Report for Lab 10 of BAE305 Spring 2025
# Lab 10 - Python and AI: Surrender your Data to our Future AI Helpers
* By: Abby Phillips
* Submitted: April 23rd, 2025


# Summary  
In Lab 10, we explored the use of Python and Streamlit to manipulate, visualize, and interpret water quality datasets. The la


# Materials
- Computer with internet access
- Python 3+ environment (Anaconda recommended)
- station.csv – water station metadata
- narrowresult.csv – water quality measurements
- GitHub and Streamlit accounts
- Libraries: pandas, matplotlib, folium, streamlit, streamlit-folium

# Assembly Procedures 
## Part 1: Map It 
1. Data Exploration
2. We downloaded the station.csv file from the Canvas assignment and opened it in Excel to explore the dataset. We examined column headers such as MonitoringLocationName, MonitoringLocationTypeName, LatitudeMeasure, and LongitudeMeasure to identify relevant station information.
3. Environment Setup
4. We opened a Jupyter Notebook within a Python environment (Anaconda) and imported required libraries: pandas for data manipulation and folium for map visualization.
5. AI-Assisted Code Generation
6. Using GPT-4, we prompted the model to create a Python function that would:
    - Read the CSV file
    - Filter for monitoring locations relevant to water quality (streams and rivers)
    - Drop any duplicate or irrelevant rows
    - Return a clean DataFrame of unique stations
7. Mapping with Folium
8. We requested a follow-up prompt to generate an interactive map showing the location of each station. The code used mean latitude and longitude to center the map and placed a marker for each valid site.

## Part 2: What’s Normal 
1. Understanding the Dataset
2. We downloaded the narrowresult.csv file from Canvas and opened it in Excel to identify key data columns:
    - CharacteristicName (e.g., pH, Turbidity)
    - ActivityStartDate (timestamp of measurement)
    - ResultMeasureValue (the numeric result)
3. We uploaded the file to our GitHub Codespaces environment, ensuring compatibility with Python. We ensured headers were properly formatted and consistent with the column names used in our code.
4. We prompted GPT-4 with to create a Python code that accesses Narrowresult, filters for a desired water quality characteristic, and plots the results with pH. We also tested with turbidity. Initial File Review
5. We further prompted GPT-4 to generate dual-characteristic plots, displaying pH and turbidity using two y-axes on a shared timeline.
6. We debugged plot formatting (e.g., axis labeling, legends, grid) and added customizations such as markers and figure sizing for clarity and readability.

## Part 3: Streamlit App Development
1. Streamlit and GitHub Integration Setup
2. We signed up at streamlit.io using our GitHub credentials to link accounts. After email and authorization verifications, we created a new public GitHub repository named BAE305-SP25-Lab10.
3. Streamlit App File Setup
4. Within the GitHub repo, we added a new file: streamlit_app.py. We initialized Streamlit deployment by selecting:
    - Branch: main
    - Main file path: streamlit_app.py
    - Custom App URL for easy access
5. Prompting GPT for Full App Code to request
    - File uploaders for station.csv and narrowresult.csv
    - Sidebar selection of CharacteristicName, date range, and value range
    - Filtering and merging logic
    - Interactive folium map of valid station locations
    - Time-series plot of selected contaminant using matplotlib
6. We edited the code in GitHub Codespaces. Upon every session restart, we reinstalled required libraries using pip install commands in the terminal. These commands were connected to the app through the file requirements.txt. It was important to add this to the same repository as the streamlit app


# Test Equipment
- Computer running Python and web browser
- GitHub and Streamlit portals
- Local and cloud IDEs (Anaconda, VS Code, GitHub Codespaces)


# Test Procedures
## Part 1: Map It 
1. We verified that the Python function correctly filtered station types by checking output in the console. Each site was verified to be unique and relevant (e.g., labeled as "Stream" or "River").
2. We confirmed that all markers were displayed on the map and that each popup provided readable site names and identifiers. We tested marker interactivity by hovering and clicking on them.
3. We intentionally removed key columns and reran the code to ensure that missing values were properly excluded and the map remained functional.

## Part 2: What’s Normal
1. We tested the filter function with several parameters including "pH", "Turbidity", and "Nitrate" to ensure proper data isolation and plot rendering.
2. We validated the order of data points along the x-axis and ensured that non-numeric entries were dropped. We also tested small and large datasets to check for rendering performance.
3. We plotted both pH and turbidity simultaneously and ensured each had a distinct y-axis with readable tick marks. Overlapping points were visualized with different colors.

## Part 3: Streamlit App
1. We uploaded several versions of station.csv and narrowresult.csv to confirm compatibility. Files with different formats (e.g., missing headers or altered delimiters) were tested to ensure error messages were shown when appropriate.
2. We used the sidebar to select various contaminants, adjust the result value slider, and modify the date range. The app was verified to update both the map and plot in real-time.
3. We cross-checked map markers with actual data points from the narrowresult.csv and confirmed only valid entries within selected filters were shown.
4. We varied the selected contaminant and date range to test plot updates. Chart titles and axes were checked for accuracy and clarity.

# Test Results
## Part 1: Map It
| **Goal**   | Access the station database, filter for water quality measurement site names, and show unique location info |
|------------|------------------------------------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Generate a Python function that accesses the database, filters for the names of water quality measurement sites, and displays the location information for all sites without repetition. |

```ruby
import pandas as pd

def get_unique_water_quality_sites(data):
    # Filter for rows where the MonitoringLocationTypeName contains water quality-related sites (e.g., stream, river, etc.)
    water_quality_sites = data[data['MonitoringLocationTypeName'].str.contains('Stream|River', na=False)]

    # Select only the relevant columns to display location information
    location_info = water_quality_sites[['MonitoringLocationName', 'MonitoringLocationIdentifier', 
                                         'MonitoringLocationTypeName', 'MonitoringLocationDescriptionText']]
    
    # Drop duplicate rows to ensure no repetition
    unique_location_info = location_info.drop_duplicates()

    # Display the filtered and unique location information
    import ace_tools as tools; tools.display_dataframe_to_user(name="Unique Water Quality Sites", dataframe=unique_location_info)

    return unique_location_info

# Call the function to display the results
unique_sites = get_unique_water_quality_sites(data)
```
---

| **Goal**   | Create an interactive map showing the location of all stations from the database |
|------------|----------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Create a map that pinpoints the location of every station in the database. |

![image](https://github.com/user-attachments/assets/9d97ea56-b106-4b37-88f3-6c0384f871d7)

---

| **Goal**   | Request the full Python code used to generate the station map |
|------------|---------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Show Python code to generate the above map |

```ruby
import pandas as pd
import folium

# Load the station data
station_df = pd.read_csv("station.csv")  # Replace with your file path if needed

# Drop rows with missing latitude or longitude
valid_locations = station_df.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])

# Create a base map centered around the mean coordinates
m = folium.Map(
    location=[valid_locations["LatitudeMeasure"].mean(), 
              valid_locations["LongitudeMeasure"].mean()],
    zoom_start=7
)

# Add a marker for each station
for _, row in valid_locations.iterrows():
    folium.Marker(
        location=[row["LatitudeMeasure"], row["LongitudeMeasure"]],
        popup=row["MonitoringLocationName"],
        tooltip=row["MonitoringLocationIdentifier"]
    ).add_to(m)

# Save the map to an HTML file
m.save("station_map.html")
```

```ruby


```
<p align="left"><em>  </em></p>

## Part 2: What's Normal
---

| **Goal**   | Generate a Python function that accesses the Narrowresult dataset, filters for a desired water quality characteristic, and plots the results |
|------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Generate a Python function that accesses Narrowresult, filters for a desired water quality characteristic, and plots the results with pH. Can you please generate with turbidity? |

![image](https://github.com/user-attachments/assets/df79e2bc-9eb3-45f8-8c94-dd47d2362183)

![image](https://github.com/user-attachments/assets/4337a42d-8614-436e-91a9-0b206515c816)

| **Goal**   | Generate a Python function that accesses the Narrowresult dataset and plots multiple datasets on the same plot|
|------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Can you please modify the python code so I can ask for two characteristics at the same time with 2 y axis? |

![image](https://github.com/user-attachments/assets/cae75868-c4f5-48ce-97e8-56d883d6fa08)



---

| **Goal**   | Request the full Python code used to generate the turbidity and pH comparison graphs |
|------------|----------------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Can you please show me the full Python code you used to generate these graphs? |

```ruby
import pandas as pd
import matplotlib.pyplot as plt

def filter_and_plot_water_quality(dataframe, characteristic_name):
    """
    Filters the dataframe for a specific water quality characteristic
    and plots the result values over time.
    """
    filtered_df = dataframe[dataframe['CharacteristicName'] == characteristic_name].copy()
    filtered_df['ActivityStartDate'] = pd.to_datetime(filtered_df['ActivityStartDate'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['ActivityStartDate', 'ResultMeasureValue'])
    filtered_df['ResultMeasureValue'] = pd.to_numeric(filtered_df['ResultMeasureValue'], errors='coerce')
    filtered_df = filtered_df.sort_values(by='ActivityStartDate')

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['ActivityStartDate'], filtered_df['ResultMeasureValue'], marker='o', linestyle='-')
    plt.title(f"{characteristic_name} Levels Over Time")
    plt.xlabel("Date")
    plt.ylabel("Result Measure Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def filter_and_plot_multiple_characteristics(dataframe, characteristic_names):
    """
    Filters the dataframe for multiple water quality characteristics
    and plots each characteristic's result values over time.
    """
    plt.figure(figsize=(14, 7))

    for characteristic in characteristic_names:
        filtered_df = dataframe[dataframe['CharacteristicName'] == characteristic].copy()
        filtered_df['ActivityStartDate'] = pd.to_datetime(filtered_df['ActivityStartDate'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['ActivityStartDate', 'ResultMeasureValue'])
        filtered_df['ResultMeasureValue'] = pd.to_numeric(filtered_df['ResultMeasureValue'], errors='coerce')
        filtered_df = filtered_df.sort_values(by='ActivityStartDate')

        plt.plot(filtered_df['ActivityStartDate'], filtered_df['ResultMeasureValue'],
                 marker='o', linestyle='-', label=characteristic)

    plt.title("Water Quality Characteristics Over Time")
    plt.xlabel("Date")
    plt.ylabel("Result Measure Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def filter_and_plot_dual_axis(dataframe, characteristic_1, characteristic_2):
    """
    Plots two water quality characteristics over time with separate y-axes.
    """
    df1 = dataframe[dataframe['CharacteristicName'] == characteristic_1].copy()
    df1['ActivityStartDate'] = pd.to_datetime(df1['ActivityStartDate'], errors='coerce')
    df1 = df1.dropna(subset=['ActivityStartDate', 'ResultMeasureValue'])
    df1['ResultMeasureValue'] = pd.to_numeric(df1['ResultMeasureValue'], errors='coerce')
    df1 = df1.sort_values(by='ActivityStartDate')

    df2 = dataframe[dataframe['CharacteristicName'] == characteristic_2].copy()
    df2['ActivityStartDate'] = pd.to_datetime(df2['ActivityStartDate'], errors='coerce')
    df2 = df2.dropna(subset=['ActivityStartDate', 'ResultMeasureValue'])
    df2['ResultMeasureValue'] = pd.to_numeric(df2['ResultMeasureValue'], errors='coerce')
    df2 = df2.sort_values(by='ActivityStartDate')

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(df1['ActivityStartDate'], df1['ResultMeasureValue'], 'b-', label=characteristic_1)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{characteristic_1}', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(df2['ActivityStartDate'], df2['ResultMeasureValue'], 'r-', label=characteristic_2)
    ax2.set_ylabel(f'{characteristic_2}', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title(f"{characteristic_1} and {characteristic_2} Over Time")
    fig.tight_layout()
    plt.grid(True)
    plt.show()
```

## Part 3: Streamlit
| **Goal**   | Develop a Streamlit app that allows uploading station and result databases, filtering for a contaminant and range, and displaying a map and trend graph |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model**  | GPT-4 (ChatGPT-4, April 2025 version) |
| **Prompt** | Develop a Streamlit app that allows the user to upload both databases used in Part 1 and 2, to search for a contaminant in the databases. Once a contaminant has been selected you should be able to define the range of values and dates that you want to show. After modifying the ranges, update the map showing the location of the stations with the contaminant within that range and measured 

```ruby
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
```
```ruby
streamlit 
pandas 
folium 
streamlit-folium 
matplotlib
```
requirements.txt

https://bookish-telegram-g4r6prjrxrwx2966-8501.app.github.dev/ 
https://bae305-sp25-lab10-abbyphillips.streamlit.app/

# Discussion



# Conclusion



