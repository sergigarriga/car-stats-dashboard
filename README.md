# Car Statistics Dashboard - Streamlit Application

This project is a Streamlit web application designed for analyzing personal driving data, particularly focused on hybrid or plug-in hybrid vehicles (PHEVs). It processes data from a CSV file (likely exported from a car manufacturer's app or service) containing metrics like electricity/fuel consumption, distance, driving time, and speed.

The application provides interactive visualizations and calculations to help users understand their driving patterns, energy consumption, running costs, and potential savings compared to conventional gasoline and fully electric vehicles. It also includes experimental future projections based on historical data and AI-powered explanations for charts.

## Project Structure

```
car-stats-dashboard/
├── src/
│   ├── main.py               # Main script for the Streamlit application
│   ├── components/           # UI components
│   │   ├── __init__.py
│   │   └── menu.py           # Creates the Streamlit sidebar menu and filters
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── data_processing.py # Data loading, cleaning, and transformation
│   │   ├── ai_caller.py      # Function(s) to interact with the AI model
│   │   └── predictions.py    # Functions for future projections
│   └── assets/               # Static assets like the data file
│       └── car_stats.csv     # Default location for driving data
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_ai_caller.py
│   └── test_predictions.py
├── requirements.txt          # Project dependencies
├── .gitignore                # Git ignore file
├── LICENSE                   # License file (e.g., MIT) - Add one if needed
└── README.md                 # This file
```

## Key Features

- **Overview Dashboard:** Displays key summary statistics (total trips, distance, time, avg. consumption, avg. speed). Includes charts for distance per trip and cumulative distance over time.
- **Cost & Savings Analysis:** Calculates estimated running costs based on actual consumption and configurable energy prices. Compares costs against hypothetical gasoline/electric vehicles and shows potential savings.
- **Consumption Analysis:** Visualizes electric vs. fuel consumption per trip (bar chart), overall energy source distribution (pie chart), and consumption relative to trip distance (scatter plot).
- **Trip Details:** Allows selecting individual trips to view detailed metrics and compare them against the period average.
- **Performance Metrics:** Includes charts analyzing relationships between speed, consumption, distance, and time. Shows performance trends over time with dual axes.
- **Experimental Future Projections:** Provides simple predictions for the next month's distance, energy usage, and cost based on historical monthly averages. Includes historical trend plots.
- **Data Explorer:** Offers a table view of the trip data with interactive filtering and an option to download the filtered data as CSV.
- **Interactive Filtering:** Filters all displayed data and analyses by date range or specific selected dates via sidebar controls.
- **AI-Generated Explanations:** Provides optional, easy-to-understand explanations for most charts, generated by an AI model (requires API setup in `ai_caller.py`).
- **Visualizations:** Uses Plotly for interactive and informative charts.

## Data Source & Format

The application expects a CSV file named `car_stats.csv` located in the `src/assets/` directory by default (this path is configured in `src/utils/data_processing.py`).

The CSV file should contain columns representing trip data. The `load_data` function expects columns with headers similar to these (originally in Spanish, but the function attempts to handle or rename them to English internally for processing):

1. `Fecha/Hora` -> `Date/Time`: Timestamp (e.g., `YYYY-MM-DDTHH:MM:SS`, `DD.MM.YYYY HH:MM`, etc.)
2. `Consumo medio de electricidad (kWh/100 km)` -> `Average Electricity Consumption (kWh/100 km)`
3. `Consumo medio del motor de combustión (l/100 km)` -> `Average Combustion Engine Consumption (L/100 km)`
4. `Trayecto (km)` -> `Distance (km)`
5. `Tiempo de conducción` -> `Driving Time` (e.g., `H:MM` format)
6. `Velocidad media (km/h)` -> `Average Speed (km/h)`

*Note: The data cleaning functions in `src/utils/data_processing.py` attempt to handle variations like comma decimals (`,` -> `.`) and multiple common datetime formats. However, using a consistent format is recommended for best results.*

## Technology Stack

- **Python:** Core programming language (3.8+ recommended).
- **Streamlit:** Web application framework.
- **Pandas:** Data manipulation and analysis.
- **Plotly:** Interactive charting.
- **Numpy:** Numerical operations.
- **httpx & httpx-auth:** For making API calls in `ai_caller.py` (if using AI features).
- **pytest & pytest-mock:** For running unit tests.

## Installation

To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone <repository-url> # Replace <repository-url> with your repo URL
   cd your-repo-name         # Replace with the actual folder name
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(If using AI Features):** Configure API credentials needed by `src/utils/ai_caller.py`. It currently expects environment variables `CLIENT_ID` and `CLIENT_SECRET`. **Do not commit your actual secrets to Git.** Consider using a `.env` file and `python-dotenv` package, or other secrets management practices.

## Running the Application

1. Ensure your driving data CSV file (e.g., `car_stats.csv`) is placed in the `src/assets/` directory.
2. Make sure your virtual environment is activated.
3. Start the Streamlit application from the project's root directory:
   ```bash
   streamlit run src/main.py
   ```
4. The application should open automatically in your web browser.

## Configuration

Runtime configurations are handled via the application's sidebar:

- **Date Filters:** Select the analysis period.
- **Cost/Savings Parameters:** Input local average prices for gasoline (€/L) and electricity (€/kWh).
- **Comparison Consumption:** Input average consumption figures (L/100km, kWh/100km) for comparison vehicles.

## Running Tests

Unit tests are included to verify the data processing and utility functions.

1. Make sure your virtual environment is activated.
2. Ensure developer dependencies (like `pytest`, `pytest-mock`) are installed (they should be if included in `requirements.txt`).
3. Run pytest from the project's **root directory**:
   ```bash
   pytest
   ```

## Contributing

Contributions, issues, and feature requests are welcome! Please check the existing issues or open a new one to discuss changes or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.