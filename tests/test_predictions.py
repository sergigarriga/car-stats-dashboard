# tests/test_predictions.py

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Adjust import path as necessary
from src.utils.predictions import predict_next_month_stats

# --- Test Cases for predict_next_month_stats ---

# Use mocker fixture to disable @st.cache_data during tests
@pytest.fixture(autouse=True)
def mock_st_cache(mocker):
    """Disable Streamlit caching decorators for all tests in this module."""
    mocker.patch('streamlit.cache_data', lambda func=None, **kwargs: (func if func else lambda f: f))

def test_empty_dataframe():
    """Test prediction with an empty DataFrame."""
    empty_df = pd.DataFrame(columns=[
        'datetime', 'distance', 'electricity_used_kwh', 'fuel_used_liters'
    ])
    result = predict_next_month_stats(empty_df, 1.5, 0.15)
    assert result is None

def test_missing_datetime_column():
    """Test prediction with DataFrame missing the datetime column."""
    df_no_datetime = pd.DataFrame({
        'distance': [100], 'electricity_used_kwh': [10], 'fuel_used_liters': [5]
    })
    result = predict_next_month_stats(df_no_datetime, 1.5, 0.15)
    assert result is None

def test_no_valid_datetime():
    """Test prediction with invalid datetime entries."""
    df_invalid_date = pd.DataFrame({
        'datetime': ["invalid date", None, ""],
        'distance': [100, 50, 30],
        'electricity_used_kwh': [10, 8, 6],
        'fuel_used_liters': [5, 4, 3]
    })
    result = predict_next_month_stats(df_invalid_date, 1.5, 0.15)
    assert result is None

def test_insufficient_data_one_month():
    """Test prediction with data from only one month."""
    df_one_month = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-03-05', '2024-03-15', '2024-03-25']),
        'distance': [100, 50, 30],
        'electricity_used_kwh': [10, 8, 6],
        'fuel_used_liters': [5, 4, 3]
    })
    result = predict_next_month_stats(df_one_month, 1.5, 0.15)
    assert result is None # Expect None because only 1 month of data

def test_sufficient_data_two_months():
    """Test prediction with exactly two months of data."""
    df_two_months = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-02-10', '2024-02-20', '2024-03-05', '2024-03-15', '2024-03-25']),
        'distance':              [  50,        60,         100,       50,         30 ], # Feb: 110, Mar: 180 -> Avg: 145
        'electricity_used_kwh':  [  10,        12,          10,        8,          6 ], # Feb: 22,  Mar: 24  -> Avg: 23
        'fuel_used_liters':      [  2.5,       3.0,         5,         4,          3 ]  # Feb: 5.5, Mar: 12  -> Avg: 8.75
    })
    gas_price = 1.60
    elec_price = 0.18

    result = predict_next_month_stats(df_two_months, gas_price, elec_price)

    assert result is not None
    assert isinstance(result, dict)
    assert result['num_months_used'] == 2
    assert result['predicted_distance'] == pytest.approx(145.0)
    assert result['predicted_electricity_kwh'] == pytest.approx(23.0)
    assert result['predicted_fuel_l'] == pytest.approx(8.75)

    expected_elec_cost = 23.0 * elec_price
    expected_fuel_cost = 8.75 * gas_price
    expected_total_cost = expected_elec_cost + expected_fuel_cost
    assert result['predicted_electricity_cost'] == pytest.approx(expected_elec_cost)
    assert result['predicted_fuel_cost'] == pytest.approx(expected_fuel_cost)
    assert result['predicted_total_cost'] == pytest.approx(expected_total_cost)

def test_sufficient_data_multiple_months():
    """Test prediction with three months of data."""
    df_three_months = pd.DataFrame({
        'datetime': pd.to_datetime([
            '2024-01-15', '2024-01-25', # Jan
            '2024-02-10', '2024-02-20', # Feb
            '2024-03-05', '2024-03-15', '2024-03-25' # Mar
        ]),
        'distance':              [ 50, 150,   50,  60,   100,  50,  30], # Jan: 200, Feb: 110, Mar: 180 -> Avg: 163.33
        'electricity_used_kwh':  [ 10,  20,   10,  12,    10,   8,   6], # Jan: 30,  Feb: 22,  Mar: 24  -> Avg: 25.33
        'fuel_used_liters':      [  2,   8,    2.5, 3.0,   5,    4,   3]  # Jan: 10,  Feb: 5.5, Mar: 12  -> Avg: 9.166
    })
    gas_price = 1.55
    elec_price = 0.14

    result = predict_next_month_stats(df_three_months, gas_price, elec_price)

    assert result is not None
    assert isinstance(result, dict)
    assert result['num_months_used'] == 3
    assert result['predicted_distance'] == pytest.approx(490 / 3) # 163.33...
    assert result['predicted_electricity_kwh'] == pytest.approx(76 / 3) # 25.33...
    assert result['predicted_fuel_l'] == pytest.approx(27.5 / 3) # 9.166...

    expected_elec_cost = (76 / 3) * elec_price
    expected_fuel_cost = (27.5 / 3) * gas_price
    expected_total_cost = expected_elec_cost + expected_fuel_cost
    assert result['predicted_electricity_cost'] == pytest.approx(expected_elec_cost)
    assert result['predicted_fuel_cost'] == pytest.approx(expected_fuel_cost)
    assert result['predicted_total_cost'] == pytest.approx(expected_total_cost)

def test_data_with_nans_in_usage():
    """Test prediction handles NaNs in usage columns (sum treats NaN as 0)."""
    df_nans = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-02-10', '2024-02-20', '2024-03-05', '2024-03-15', '2024-03-25']),
        'distance':              [  50,        60,         100,       50,         30 ], # Feb: 110, Mar: 180 -> Avg: 145
        'electricity_used_kwh':  [  10,      None,          10,        8,          6 ], # Feb: 10,  Mar: 24  -> Avg: 17
        'fuel_used_liters':      [ np.nan,     3.0,         5,       None,         3 ]  # Feb: 3.0, Mar: 8   -> Avg: 5.5
    })
    gas_price = 1.60
    elec_price = 0.18

    result = predict_next_month_stats(df_nans, gas_price, elec_price)

    assert result is not None
    assert result['num_months_used'] == 2
    assert result['predicted_distance'] == pytest.approx(145.0)
    assert result['predicted_electricity_kwh'] == pytest.approx(17.0) # NaN treated as 0 in sum
    assert result['predicted_fuel_l'] == pytest.approx(5.5) # NaN treated as 0 in sum

    expected_elec_cost = 17.0 * elec_price
    expected_fuel_cost = 5.5 * gas_price
    expected_total_cost = expected_elec_cost + expected_fuel_cost
    assert result['predicted_electricity_cost'] == pytest.approx(expected_elec_cost)
    assert result['predicted_fuel_cost'] == pytest.approx(expected_fuel_cost)
    assert result['predicted_total_cost'] == pytest.approx(expected_total_cost)

def test_zero_usage_months():
    """Test prediction when some months have zero usage for one fuel type."""
    df_zero = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-02-10', '2024-02-20', '2024-03-05', '2024-03-15']),
        'distance':              [  50,        60,         100,        50], # Feb: 110, Mar: 150 -> Avg: 130
        'electricity_used_kwh':  [  10,        12,           0,         0], # Feb: 22,  Mar: 0   -> Avg: 11
        'fuel_used_liters':      [   0,         0,           5,         4]  # Feb: 0,   Mar: 9   -> Avg: 4.5
    })
    gas_price = 1.60
    elec_price = 0.18

    result = predict_next_month_stats(df_zero, gas_price, elec_price)

    assert result is not None
    assert result['num_months_used'] == 2
    assert result['predicted_distance'] == pytest.approx(130.0)
    assert result['predicted_electricity_kwh'] == pytest.approx(11.0)
    assert result['predicted_fuel_l'] == pytest.approx(4.5)

    expected_elec_cost = 11.0 * elec_price
    expected_fuel_cost = 4.5 * gas_price
    expected_total_cost = expected_elec_cost + expected_fuel_cost
    assert result['predicted_electricity_cost'] == pytest.approx(expected_elec_cost)
    assert result['predicted_fuel_cost'] == pytest.approx(expected_fuel_cost)
    assert result['predicted_total_cost'] == pytest.approx(expected_total_cost)