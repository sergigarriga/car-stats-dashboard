import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.utils.data_processing import (
    clean_data,
    convert_time_to_minutes,
    load_data,
    transform_data,
)

# --- Tests for convert_time_to_minutes ---


@pytest.mark.parametrize("time_str, expected_minutes", [
    ("1:30", 90),
    ("0:45", 45),
    ("10:00", 600),
    ("0:00", 0),
    ("2:05", 125),
    ("15", 15),
    (np.nan, 0),
    (None, 0),
    ("", 0),
    (":", 0),
    ("1:", 0),
    (":30", 0),
    ("1:MM", 0),
    ("HH:15", 0),
    ("1:2:3", 0),
    ("invalid", 0),
    (120, 0),
])
def test_convert_time_to_minutes(time_str, expected_minutes):
    """Test various inputs for convert_time_to_minutes."""
    assert convert_time_to_minutes(time_str) == expected_minutes


# --- Fixture for mocking Streamlit functions ---

@pytest.fixture
def mock_st_functions(mocker):
    """Mock common Streamlit functions used in data processing."""
    return {
        'warning': mocker.patch('streamlit.warning'),
        'error': mocker.patch('streamlit.error'),
        'stop': mocker.patch('streamlit.stop', side_effect=SystemExit("Streamlit stopped")),
        'cache_data': mocker.patch('streamlit.cache_data', lambda func: func),
        'cache_resource': mocker.patch('streamlit.cache_resource', lambda func: func),
    }


# --- Tests for load_data ---

@patch('pandas.read_csv')
@patch('os.path.exists')
def test_load_data_success(mock_os_exists, mock_read_csv, mock_st_functions):
    """Tests successful data loading and verifies expected warning on column mismatch."""
    mock_os_exists.return_value = True
    # Mocked read_csv output (Spanish cols)
    mock_csv_output_spanish_cols = pd.DataFrame({
        "Fecha/Hora": ["2024-01-01T10:00:00"],
        "Consumo medio de electricidad (kWh/100 km)": ["15,5"],
        "Consumo medio del motor de combustión (l/100 km)": ["5,2"],
        "Trayecto (km)": [100],
        "Tiempo de conducción": ["1:30"],
        "Velocidad media (km/h)": [67]
    })
    mock_read_csv.return_value = mock_csv_output_spanish_cols.copy()

    # Define the expected ENGLISH column names (corrected 'fuel' -> 'Combustion Engine')
    expected_english_cols = pd.Index([
        'Date/Time', 'Average Electricity Consumption (kWh/100 km)',
        'Average Combustion Engine Consumption (L/100 km)', # <-- FIX HERE
        'Distance (km)', 'Driving Time',
        'Average Speed (km/h)'
    ])

    df_returned = load_data("dummy_path.csv")

    # --- Assertions ---
    mock_read_csv.assert_called_once()
    call_args, _ = mock_read_csv.call_args
    assert os.path.isabs(call_args[0])

    # Check returned columns are English (should pass now)
    pd.testing.assert_index_equal(df_returned.columns, expected_english_cols)

    # Check content matches
    expected_content_df = mock_csv_output_spanish_cols.copy()
    expected_content_df.columns = expected_english_cols # Use corrected English names
    assert_frame_equal(df_returned, expected_content_df, check_like=True)

    # Check warning WAS called once (due to Esp->Eng rename needed by *your* load_data)
    mock_st_functions['warning'].assert_called_once()
    mock_st_functions['error'].assert_not_called()
    mock_st_functions['stop'].assert_not_called()

@patch('os.path.exists')
def test_load_data_file_not_found(mock_os_exists, mock_st_functions):
    """Test behavior when the CSV file is not found."""
    mock_os_exists.return_value = False

    with pytest.raises(SystemExit):
        load_data("nonexistent_path.csv")

    mock_st_functions['error'].assert_called()
    mock_st_functions['stop'].assert_called_once()


@patch('pandas.read_csv')
@patch('os.path.exists')
def test_load_data_wrong_column_count(mock_os_exists, mock_read_csv, mock_st_functions):
    """Test loading data with incorrect column count."""
    mock_os_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame({
        "Date/Time": ["2024-01-01T10:00:00"],
        "Average Electricity Consumption (kWh/100 km)": ["15.5"],
        "Distance (km)": [100]
    })

    with pytest.raises(SystemExit):
        load_data("dummy_path.csv")

    mock_st_functions['warning'].assert_called_once()
    mock_st_functions['error'].assert_called_once()
    mock_st_functions['stop'].assert_called_once()


@patch('pandas.read_csv')
@patch('os.path.exists')
def test_load_data_empty_csv(mock_os_exists, mock_read_csv, mock_st_functions):
    """Test loading an empty CSV file."""
    mock_os_exists.return_value = True
    mock_read_csv.side_effect = pd.errors.EmptyDataError(
        "No columns to parse from file")

    with pytest.raises(SystemExit):
        load_data("empty.csv")

    mock_st_functions['error'].assert_called()
    mock_st_functions['stop'].assert_called_once()


# --- Tests for clean_data ---

def test_clean_data_basic(mock_st_functions):
    """Test basic cleaning: renaming, type conversion, and decimal handling."""
    raw_data = pd.DataFrame({
        "Date/Time": ["2024-01-01T10:00:00+01:00", "2024-01-25T15:30"],
        "Average Electricity Consumption (kWh/100 km)": ["15.5", "16.0"],
        "Average Combustion Engine Consumption (L/100 km)": ["5.2", "0"],
        "Distance (km)": ["100", 50],
        "Driving Time": ["1:30", "0:45"],
        "Average Speed (km/h)": ["67", 66]
    })

    cleaned = clean_data(raw_data.copy())

    expected_data = pd.DataFrame({
        'datetime': pd.to_datetime(["2024-01-01T10:00:00", "2024-01-25T15:30:00"], errors='coerce'),
        'electricity_consumption': pd.Series([15.5, 16.0], dtype='Float64'),
        'fuel_consumption': pd.Series([5.2, 0.0], dtype='Float64'),
        'distance': pd.Series([100, 50], dtype='Int64'),
        'driving_time': ["1:30", "0:45"],
        'average_speed': pd.Series([67, 66], dtype='Int64'),
        'driving_minutes': [90, 45]
    })

    assert_frame_equal(cleaned, expected_data, check_dtype=True)
    mock_st_functions['warning'].assert_not_called()


def test_clean_data_with_na_and_errors(mock_st_functions):
    """Test cleaning with missing values, parsing errors, and zero distance."""
    raw_data = pd.DataFrame({
        "Date/Time": ["2024-01-01T10:00:00", None, "Invalid Date", "2024-01-04T12:00:00"],
        "Average Electricity Consumption (kWh/100 km)": ["15.5", "abc", "14.0", "12.0"],
        "Average Combustion Engine Consumption (L/100 km)": ["5.2", "6.0", "", "4.5"],
        "Distance (km)": ["100", 50, 0, 25],
        "Driving Time": ["1:30", "0:45", "0:00", "invalid"],
        "Average Speed (km/h)": ["67", None, 0, 50]
    })

    cleaned = clean_data(raw_data.copy())

    expected_data = pd.DataFrame({
        'datetime': pd.to_datetime(["2024-01-01T10:00:00", pd.NaT, "2024-01-04T12:00:00"]),
        'electricity_consumption': pd.Series([15.5, np.nan, 12.0], dtype='Float64'),
        'fuel_consumption': pd.Series([5.2, 6.0, 4.5], dtype='Float64'),
        'distance': pd.Series([100, 50, 25], dtype='Int64'),
        'driving_time': ["1:30", "0:45", "invalid"],
        'average_speed': pd.Series([67, 0, 50], dtype='Int64'),
        'driving_minutes': [90, 45, 0]
    })
    expected_data.index = [0, 1, 3]

    mock_st_functions['warning'].assert_called()
    assert_frame_equal(cleaned, expected_data,
                       check_dtype=True, check_like=True)


# --- Tests for transform_data ---

def test_transform_data_basic():
    """Test calculation of derived metrics."""
    cleaned_data = pd.DataFrame({
        'datetime': pd.to_datetime(["2024-01-01T10:00:00", "2024-01-25T15:30:00"]),
        'electricity_consumption': pd.Series([15.0, 20.0], dtype='Float64'),
        'fuel_consumption': pd.Series([5.0, 4.0], dtype='Float64'),
        'distance': pd.Series([100, 50], dtype='Int64'),
        'driving_time': ["1:30", "0:45"],
        'average_speed': pd.Series([67, 66], dtype='Int64'),
        'driving_minutes': [90, 45]
    })

    transformed = transform_data(cleaned_data.copy())

    expected_cols = [
        'datetime', 'electricity_consumption', 'fuel_consumption', 'distance',
        'driving_time', 'average_speed', 'driving_minutes',
        'electricity_used_kwh', 'fuel_used_liters', 'km_per_kwh', 'km_per_liter'
    ]
    assert list(transformed.columns) == expected_cols

    assert_series_equal(transformed['electricity_used_kwh'], pd.Series(
        [15.0, 10.0], dtype='Float64'), check_names=False, check_dtype=True)
    assert_series_equal(transformed['fuel_used_liters'], pd.Series(
        [5.0, 2.0], dtype='Float64'), check_names=False, check_dtype=True)
    assert_series_equal(transformed['km_per_kwh'], pd.Series(
        [100/15.0, 100/20.0], dtype='Float64'), check_names=False, atol=0.01, check_dtype=True)
    assert_series_equal(transformed['km_per_liter'], pd.Series(
        [100/5.0, 100/4.0], dtype='Float64'), check_names=False, atol=0.01, check_dtype=True)


def test_transform_data_zero_consumption():
    """Test derived metrics when consumption is zero."""
    cleaned_data = pd.DataFrame({
        'datetime': pd.to_datetime(["2024-01-01T10:00:00"]),
        'electricity_consumption': pd.Series([0.0], dtype='Float64'),
        'fuel_consumption': pd.Series([5.0], dtype='Float64'),
        'distance': pd.Series([100], dtype='Int64'),
        'driving_time': ["1:30"],
        'average_speed': pd.Series([67], dtype='Int64'),
        'driving_minutes': [90]
    })

    transformed = transform_data(cleaned_data.copy())

    assert pd.isna(transformed['km_per_kwh'].iloc[0])
    assert transformed['km_per_liter'].iloc[0] == pytest.approx(100 / 5.0)
    assert transformed['electricity_used_kwh'].iloc[0] == 0.0
    assert transformed['fuel_used_liters'].iloc[0] == 5.0


def test_transform_data_empty_input(mock_st_functions):
    """Test transform_data with empty input DataFrame."""
    empty_df = pd.DataFrame(columns=[
        'datetime', 'electricity_consumption', 'fuel_consumption', 'distance',
        'driving_time', 'average_speed', 'driving_minutes'
    ])
    transformed = transform_data(empty_df)

    assert transformed.empty
    assert 'electricity_used_kwh' in transformed.columns
    assert 'fuel_used_liters' in transformed.columns
    assert 'km_per_kwh' in transformed.columns
    assert 'km_per_liter' in transformed.columns
    mock_st_functions['error'].assert_not_called()
