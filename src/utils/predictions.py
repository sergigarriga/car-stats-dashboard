import pandas as pd
import streamlit as st


@st.cache_data
def predict_next_month_stats(df: pd.DataFrame, gas_price: float, elec_price: float):
    """
    Predicts next month's stats based on historical monthly averages.

    Args:
        df: DataFrame with historical data, must include 'datetime', 'distance',
            'electricity_used_kwh', 'fuel_used_liters'.
        gas_price: Current price per liter of gasoline.
        elec_price: Current price per kWh of electricity.

    Returns:
        A dictionary with predicted stats for the next month,
        or None if insufficient data exists.
    """
    if df.empty or 'datetime' not in df.columns:
        print("Prediction Error: DataFrame is empty or missing 'datetime' column.")
        return None

    # Ensure datetime is the correct type
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])

    if df.empty:
        print("Prediction Error: No valid datetime entries found.")
        return None

    # Set datetime as index for resampling or use Period for grouping
    df['year_month'] = df['datetime'].dt.to_period('M')

    # Calculate monthly sums
    monthly_stats = df.groupby('year_month').agg(
        total_distance=pd.NamedAgg(column='distance', aggfunc='sum'),
        total_elec_kwh=pd.NamedAgg(column='electricity_used_kwh', aggfunc='sum'),
        total_fuel_L=pd.NamedAgg(column='fuel_used_liters', aggfunc='sum')
    )

    # We need at least 2 full months of data for a meaningful average
    if len(monthly_stats) < 2:
        print(f"Prediction Info: Insufficient historical data ({len(monthly_stats)} months found, need >= 2).")
        return None

    # Calculate average monthly figures
    avg_monthly_distance = monthly_stats['total_distance'].mean()
    avg_monthly_elec_kwh = monthly_stats['total_elec_kwh'].mean()
    avg_monthly_fuel_L = monthly_stats['total_fuel_L'].mean()

    # Predict next month's cost based on average usage and *current* prices
    predicted_electricity_cost = avg_monthly_elec_kwh * elec_price
    predicted_fuel_cost = avg_monthly_fuel_L * gas_price
    predicted_monthly_cost = predicted_electricity_cost + predicted_fuel_cost

    predictions = {
        "predicted_distance": avg_monthly_distance,
        "predicted_electricity_kwh": avg_monthly_elec_kwh,
        "predicted_fuel_l": avg_monthly_fuel_L,
        "predicted_electricity_cost": predicted_electricity_cost,
        "predicted_fuel_cost": predicted_fuel_cost,
        "predicted_total_cost": predicted_monthly_cost,
        "num_months_used": len(monthly_stats)  # Store how many months were averaged
    }

    return predictions
