import os

import pandas as pd
import streamlit as st


def convert_time_to_minutes(time_str):
    """Convert time string in 'H:MM' format to minutes."""
    try:
        if pd.isna(time_str):
            return 0
        parts = str(time_str).split(':')
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 1 and parts[0].isdigit():
            return int(parts[0])  # Handle cases with only minutes
        return 0
    except (ValueError, TypeError, AttributeError):
        return 0


def load_data(file_path="src/data/car_stats.csv"):
    """Load car statistics data from a CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Error: File '{os.path.basename(file_path)}' not found.")
        st.error(f"Path searched: {os.path.abspath(file_path)}")
        st.stop()

    try:
        data = pd.read_csv(file_path, encoding='utf-8',
                           delimiter=',', header=0)
        expected_cols = ["Date/Time", "Average Electricity Consumption (kWh/100 km)",
                         "Average Combustion Engine Consumption (L/100 km)",
                         "Distance (km)", "Driving Time", "Average Speed (km/h)"]

        if list(data.columns) != expected_cols:
            st.warning(
                f"CSV columns ({list(data.columns)}) do not match expected columns ({expected_cols}). Attempting to rename.")
            if len(data.columns) == len(expected_cols):
                data.columns = expected_cols
            else:
                st.error("Critical error: Column count mismatch. Cannot proceed.")
                st.stop()
    except FileNotFoundError:
        st.error(f"Error: File not found at path: {file_path}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("Error: CSV file is empty.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading CSV: {e}")
        st.stop()

    return data


def clean_data(data):
    """Clean and transform car statistics data."""
    new_columns = ['datetime', 'electricity_consumption', 'fuel_consumption',
                   'distance', 'driving_time', 'average_speed']
    if len(data.columns) == len(new_columns):
        data.columns = new_columns
    else:
        st.error("Critical error during column renaming.")
        st.stop()

    # Convert decimal commas to decimal points
    cols_to_convert = ['electricity_consumption', 'fuel_consumption']
    for col in cols_to_convert:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col].str.replace(
                ',', '.', regex=False), errors='coerce')
        elif pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].astype(float)
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Convert integer columns, handling errors
    int_cols = ['average_speed', 'distance']
    for col in int_cols:
        data[col] = pd.to_numeric(
            data[col], errors='coerce').fillna(0).astype('Int64')

    # Parse datetime column
    datetime_col = data['datetime'].copy()
    if datetime_col.dtype == 'object':
        datetime_col = datetime_col.str.split(
            '+').str[0].str.strip().str.replace('Z', '', regex=False)

    formats_to_try = ['%Y-%m-%dT%H:%M:%S', '%d.%m.%Y %H:%M',
                      '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M']
    parsed_datetime = pd.NaT
    for fmt in formats_to_try:
        try:
            parsed_datetime = pd.to_datetime(
                datetime_col, format=fmt, errors='coerce')
            if parsed_datetime.notna().all():
                break
        except ValueError:
            continue

    if parsed_datetime.isna().all():
        st.warning(
            "Failed to parse datetime with specific formats. Attempting automatic inference.")
        parsed_datetime = pd.to_datetime(
            datetime_col, errors='coerce', infer_datetime_format=True)

    data['datetime'] = parsed_datetime
    if data['datetime'].isna().any():
        st.warning(
            f"Warning: {data['datetime'].isna().sum()} datetime values could not be parsed.")

    # Convert driving time to minutes
    data['driving_minutes'] = data['driving_time'].astype(
        str).apply(convert_time_to_minutes).fillna(0).astype(int)

    # Fill NaN values in key numeric columns
    data['electricity_consumption'] = data['electricity_consumption'].fillna(0)
    data['fuel_consumption'] = data['fuel_consumption'].fillna(0)
    data['distance'] = data['distance'].fillna(0)

    # Remove rows with zero distance
    initial_rows = len(data)
    data = data[data['distance'] > 0].copy()
    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        print(f"Removed {rows_dropped} records with 0 km distance.")

    return data


def transform_data(data):
    """Perform calculations and add derived metrics."""
    if data.empty or 'distance' not in data.columns or data['distance'].isnull().all():
        st.warning("No valid data available for transformations.")
        expected_cols = data.columns.tolist(
        ) + ['electricity_used_kwh', 'fuel_used_liters', 'km_per_kwh', 'km_per_liter']
        return pd.DataFrame(columns=expected_cols)

    data_valid = data[data['distance'] > 0].copy()
    if data_valid.empty:
        return data_valid

    # Calculate energy used per trip
    data_valid['electricity_used_kwh'] = data_valid['electricity_consumption'] * \
        data_valid['distance'] / 100
    data_valid['fuel_used_liters'] = data_valid['fuel_consumption'] * \
        data_valid['distance'] / 100

    # Calculate efficiency metrics
    data_valid['km_per_kwh'] = 0.0
    data_valid['km_per_liter'] = 0.0

    mask_elec = data_valid['electricity_consumption'] > 0
    data_valid.loc[mask_elec, 'km_per_kwh'] = 100 / \
        data_valid.loc[mask_elec, 'electricity_consumption']

    mask_fuel = data_valid['fuel_consumption'] > 0
    data_valid.loc[mask_fuel, 'km_per_liter'] = 100 / \
        data_valid.loc[mask_fuel, 'fuel_consumption']

    # Fill NaN or Inf values
    data_valid['km_per_kwh'] = data_valid['km_per_kwh'].fillna(0)
    data_valid['km_per_liter'] = data_valid['km_per_liter'].fillna(0)

    return data_valid
