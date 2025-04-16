# Car Statistics Dashboard - Streamlit Application

This project is a Streamlit web application designed for analyzing personal driving data, particularly focused on hybrid or plug-in hybrid vehicles (PHEVs). It processes data from a CSV file (likely exported from a car manufacturer's app or service) containing metrics like electricity/fuel consumption, distance, driving time, and speed.

The application provides interactive visualizations and calculations to help users understand their driving patterns, energy consumption, running costs, and potential savings compared to conventional gasoline and fully electric vehicles.

*(Optional: Add a screenshot or GIF of the application in action here)*

## Project Structure

```
car-stats-dashboard/
├── src/
│   ├── main.py             # Main script for the Streamlit application
│   ├── components/         # UI components
│   │   └── menu.py         # Creates the Streamlit sidebar menu and filters
│   ├── utils/              # Utility functions
│   │   └── data_processing.py # Contains functions for data loading, cleaning, and transformation
│   └── assets/
│       └── car_stats.csv     # Default location for driving data
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Key Features

- **Overview Dashboard:** Displays key summary statistics like total trips, distance, driving time, average consumption, and average speed over the selected period.
- **Cost & Savings Analysis:**
  - Calculates the estimated running cost based on actual electricity and fuel consumption.
  - Compares costs against hypothetical gasoline-only and fully electric vehicles.
  - Shows potential monetary savings.
  - Uses configurable prices (€/L, €/kWh) and comparison vehicle consumption rates (L/100km, kWh/100km) via sidebar inputs.
- **Consumption Analysis:** Visualizes electricity vs. fuel consumption per trip and the overall energy source distribution. Includes scatter plots showing consumption relative to distance.
- **Trip Details:** Allows selecting individual trips to view detailed metrics and compare them against the average for the selected period.
- **Performance Metrics:** Includes charts analyzing relationships between speed, consumption, distance, and driving time (e.g., Speed vs. Electricity Consumption, Distance vs. Driving Time with trendline). Shows performance trends over time.
- **Data Explorer:** Provides a table view of the raw trip data with filtering options for distance and consumption rates. Allows downloading the filtered data.
- **Interactive Filtering:** Filter the displayed data by date range or specific selected dates using sidebar controls.
- **Visualizations:** Uses Plotly for interactive charts and graphs.

## Data Source & Format

The application expects a CSV file named `car_stats.csv` located in the `src/assets/` directory by default.

The CSV file should contain columns representing trip data. Based on the typical processing steps, the expected columns (or similar) after initial loading are:

1. `Date/Time`: Timestamp of the trip (e.g., `YYYY-MM-DDTHH:MM:SS+ZZ:ZZ` or similar parseable format).
2. `Average Electricity Consumption (kWh/100 km)`: Average electricity consumption for the trip.
3. `Average Combustion Engine Consumption (L/100 km)`: Average fuel consumption for the trip.
4. `Distance (km)`: Distance of the trip in kilometers.
5. `Driving Time`: Driving time for the trip (e.g., `H:MM` format).
6. `Average Speed (km/h)`: Average speed for the trip.

*Note: The data cleaning functions in `src/utils/data_processing.py` attempt to handle variations like comma decimals (`,` -> `.`) and different datetime formats, but adherence to a consistent format is recommended.*

## Technology Stack

- **Python:** Core programming language.
- **Streamlit:** Web application framework for creating the interactive dashboard.
- **Pandas:** Data manipulation and analysis.
- **Plotly:** Generating interactive charts and visualizations.

## Installation

To run this project locally, ensure you have Python 3.8+ installed.

1. **Clone the repository:**
   ```bash
   git clone <repository-url> # Replace <repository-url> with your repo URL
   cd your-repo-name # Replace your-repo-name with the cloned directory name
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # Activate it:
   # Windows: .\venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Ensure your `car_stats.csv` file is placed in the `src/assets/` directory.

Start the Streamlit application by running the following command in your terminal from the project's root directory:

```bash
streamlit run src/main.py
```

This will typically launch the application automatically in your default web browser.

## Configuration

Most configurations are handled directly within the application's sidebar:

- **Date Filters:** Select the time period for analysis.
- **Cost/Savings Parameters:** Input your local average prices for gasoline (€/L) and electricity (€/kWh).
- **Comparison Consumption:** Input the average consumption figures (L/100km, kWh/100km) for the gasoline and electric vehicles you want to compare against.

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue to discuss changes or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.