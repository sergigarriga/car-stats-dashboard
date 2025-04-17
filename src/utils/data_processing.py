# src/utils/data_processing.py
import os

import pandas as pd
import streamlit as st

# --- Helper and Data Processing Functions ---


def convert_time_to_minutes(time_str):
    """Converts a time string in the format 'H:MM' to minutes."""
    if not isinstance(time_str, str):
        return 0
    try:
        if pd.isna(time_str) or time_str == "":
            return 0
        parts = time_str.split(':')
        if len(parts) != 2:
            if len(parts) == 1 and parts[0].isdigit():
                return int(parts[0])
            return 0  # Invalid format if not H:MM or M
        if parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return 0  # Invalid format if H or MM are not digits
    except (ValueError, TypeError, AttributeError):
        return 0  # Handle unexpected errors during conversion


def load_data(file_path="src/assets/car_stats.csv"):
    """Loads car statistics data from a CSV file."""
    abs_file_path = os.path.abspath(file_path)

    if not os.path.exists(abs_file_path):
        error_message = f"Error: File '{os.path.basename(file_path)}' not found at: {abs_file_path}"
        st.error(error_message)
        st.stop()

    try:
        data = pd.read_csv(abs_file_path, encoding='utf-8',
                           delimiter=',', header=0)

        expected_cols = ["Date/Time", "Average Electricity Consumption (kWh/100 km)",
                         "Average Combustion Engine Consumption (L/100 km)", "Distance (km)",
                         "Driving Time", "Average Speed (km/h)"]

        if list(data.columns) != expected_cols:
            warning_message = f"Warning: CSV columns ({list(data.columns)}) do not match expected columns ({expected_cols}). Attempting to rename."
            st.warning(warning_message)

            if len(data.columns) == len(expected_cols):
                data.columns = expected_cols
            else:
                error_message = "Critical Error: Column count mismatch. Cannot proceed."
                st.error(error_message)
                st.stop()

    except FileNotFoundError:
        error_message = f"Error: CSV file not found at: {abs_file_path}"
        st.stop()
        st.stop()
    except pd.errors.EmptyDataError:
        error_message = "Error: The CSV file is empty."
        st.error(error_message)
        st.stop()
    except Exception as e:
        error_message = f"Unexpected error while loading CSV: {e}"
        st.error(error_message)
        st.stop()

    return data


def clean_data(data):
    """Cleans and transforms car statistics data."""
    new_columns = ['datetime', 'electricity_consumption', 'fuel_consumption',
                   'distance', 'driving_time', 'average_speed']
    if len(data.columns) == len(new_columns):
        data.columns = new_columns
    else:
        error_message = "Critical Error: Column count mismatch during renaming."
        st.error(error_message)
        st.stop()

    cols_to_convert_float = ['electricity_consumption', 'fuel_consumption']
    for col in cols_to_convert_float:
        if col in data.columns:
            if data[col].dtype == 'object':
                data[col] = pd.to_numeric(data[col].astype(str).str.replace(
                    ',', '.', regex=False), errors='coerce').astype('Float64')
            else:
                data[col] = pd.to_numeric(
                    data[col], errors='coerce').astype('Float64')
        else:
            print(f"Warning: Expected column '{col}' not found for cleaning.")

    int_cols = ['average_speed', 'distance']
    for col in int_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col], errors='coerce').fillna(0).astype('Int64')
        else:
            print(f"Warning: Expected column '{col}' not found for cleaning.")

    if 'datetime' in data.columns:
        datetime_col_obj = data['datetime'].copy()

        if pd.api.types.is_string_dtype(datetime_col_obj) or datetime_col_obj.dtype == 'object':
            datetime_col_obj = datetime_col_obj.astype(str).str.split(
                '+').str[0].str.strip().str.replace('Z', '', regex=False)

        # Initialize the final datetime column with NaT
        data['datetime'] = pd.NaT
        needs_parsing_mask = data['datetime'].isna()

        # Define formats
        formats_to_try = [
            '%Y-%m-%dT%H:%M:%S', # ISO 8601 with T and seconds
            '%Y-%m-%dT%H:%M',    # <<< --- ADD THIS FORMAT --- ISO 8601 with T, NO seconds
            '%d.%m.%Y %H:%M',    # Format like '25.01.2024 15:30'
            '%Y-%m-%d %H:%M:%S', # ISO 8601 with space and seconds
            '%Y-%m-%d %H:%M',    # ISO 8601 with space, NO seconds
            '%d/%m/%Y %H:%M',    # Format like '25/01/2024 15:30'
        ]

        for fmt in formats_to_try:
            if not needs_parsing_mask.any():
                break
            parsed_chunk = pd.to_datetime(
                datetime_col_obj[needs_parsing_mask], format=fmt, errors='coerce')
            success_in_chunk_mask = parsed_chunk.notna()
            original_indices = data.index[needs_parsing_mask][success_in_chunk_mask]
            if not original_indices.empty:
                data.loc[original_indices,
                         'datetime'] = parsed_chunk[success_in_chunk_mask].values
            needs_parsing_mask = data['datetime'].isna()

        if needs_parsing_mask.any():
            final_attempt = pd.to_datetime(
                datetime_col_obj[needs_parsing_mask], errors='coerce', format='%Y-%m-%d %H:%M')
            success_in_final_mask = final_attempt.notna()
            if success_in_final_mask.any():
                original_indices = data.index[needs_parsing_mask][success_in_final_mask]
                if not original_indices.empty:
                    data.loc[original_indices,
                             'datetime'] = final_attempt[success_in_final_mask].values

        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        unparsed_count = data['datetime'].isna().sum()
        if unparsed_count > 0:
            warning_message = f"Warning: {unparsed_count} datetime values could not be parsed."
            st.warning(warning_message)
    else:
        print("Warning: 'datetime' column not found.")
        data['datetime'] = pd.NaT

    if 'driving_time' in data.columns:
        data['driving_minutes'] = data['driving_time'].apply(
            convert_time_to_minutes)
    else:
        print("Warning: 'driving_time' column not found.")
        data['driving_minutes'] = 0
    data['driving_minutes'] = data['driving_minutes'].fillna(0).astype(int)

    initial_rows = len(data)
    if 'distance' in data.columns:
        data['distance'] = pd.to_numeric(data['distance'], errors='coerce')
        data = data.dropna(subset=['distance'])
        data = data[data['distance'] > 0].copy()
    else:
        print("Error: Required column 'distance' not found.")
        return pd.DataFrame(columns=data.columns)

    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        print(
            f"Info: {rows_dropped} records with invalid or zero distance were removed.")

    return data


def transform_data(data):
    """Performs calculations and adds derived metrics."""
    required_cols = ['distance', 'electricity_consumption', 'fuel_consumption']
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        error_message = f"Error: Missing required columns for transformation: {missing}"
        st.error(error_message)
        st.stop()
        return data

    if data.empty:
        return pd.DataFrame(columns=data.columns.tolist() + [
            'electricity_used_kwh', 'fuel_used_liters', 'km_per_kwh', 'km_per_liter'
        ])

    data = data.copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce').astype('Float64')

    data['electricity_used_kwh'] = data['electricity_consumption'] * \
        data['distance'] / 100
    data['fuel_used_liters'] = data['fuel_consumption'] * data['distance'] / 100

    data['km_per_kwh'] = pd.Series(dtype='Float64')
    data['km_per_liter'] = pd.Series(dtype='Float64')

    mask_elec = data['electricity_consumption'].notna() & (
        data['electricity_consumption'] > 0)
    if mask_elec.any():
        data.loc[mask_elec, 'km_per_kwh'] = 100 / \
            data.loc[mask_elec, 'electricity_consumption']

    mask_fuel = data['fuel_consumption'].notna() & (
        data['fuel_consumption'] > 0)
    if mask_fuel.any():
        data.loc[mask_fuel, 'km_per_liter'] = 100 / \
            data.loc[mask_fuel, 'fuel_consumption']

    return data
