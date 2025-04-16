import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.menu import create_menu
from utils.data_processing import clean_data, load_data, transform_data


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


def calculate_and_display_savings(data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp):
    """Calculate and display cost estimations and savings."""
    st.header("Cost and Savings Estimation")

    if data.empty:
        st.warning(
            "No data available for the selected period to calculate savings.")
        return

    total_distance = data['distance'].sum()
    total_elec_kwh = data['electricity_used_kwh'].sum()
    total_fuel_L = data['fuel_used_liters'].sum()

    if total_distance == 0:
        st.warning(
            "Total distance for the selected period is 0 km. Savings cannot be calculated.")
        return

    # Cost calculations
    your_cost = (total_elec_kwh * elec_price) + (total_fuel_L * gas_price)
    gas_car_cost = (total_distance / 100) * gas_consumption_comp * gas_price
    elec_car_cost = (total_distance / 100) * elec_consumption_comp * elec_price

    # Savings
    savings_vs_gas = gas_car_cost - your_cost
    savings_vs_elec = elec_car_cost - your_cost

    # Display results
    st.subheader(f"Analysis for {total_distance} km traveled:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Hybrid Car Cost", f"{your_cost:.2f} €")
    with col2:
        st.metric(
            f"Estimated Gasoline Cost ({gas_consumption_comp} L/100km)", f"{gas_car_cost:.2f} €")
    with col3:
        st.metric(
            f"Estimated Electric Cost ({elec_consumption_comp} kWh/100km)", f"{elec_car_cost:.2f} €")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.metric("Savings vs. Gasoline Car", f"{savings_vs_gas:.2f} €")
        st.success(f"You saved {savings_vs_gas:.2f} € compared to a gasoline car.") if savings_vs_gas > 0 else st.warning(
            f"You spent {abs(savings_vs_gas):.2f} € more compared to a gasoline car.")
    with col5:
        st.metric("Savings vs. Electric Car", f"{savings_vs_elec:.2f} €")
        st.success(f"You saved {savings_vs_elec:.2f} € compared to an electric car.") if savings_vs_elec > 0 else st.warning(
            f"You spent {abs(savings_vs_elec):.2f} € more compared to an electric car.")

    st.caption(
        f"Calculations based on prices: Gasoline = {gas_price:.3f} €/L, Electricity = {elec_price:.3f} €/kWh.")


def show_overview(data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp):
    """Display an overview of statistics and savings."""
    st.header("Usage Overview")

    if data.empty:
        st.warning("No data available for the selected period.")
        calculate_and_display_savings(
            data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp)
        return

    col1, col2, col3 = st.columns(3)
    total_distance = data['distance'].sum()
    total_minutes = data['driving_minutes'].sum()
    avg_elec_consumption = data['electricity_consumption'][data['electricity_consumption'] > 0].mean(
    ) or 0
    avg_fuel_consumption = data['fuel_consumption'][data['fuel_consumption'] > 0].mean(
    ) or 0
    avg_speed = data['average_speed'].mean() or 0

    with col1:
        st.metric("Total Trips", len(data))
        st.metric("Total Distance", f"{total_distance} km")
    with col2:
        st.metric("Total Driving Time",
                  f"{total_minutes // 60}h {total_minutes % 60}min")
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
    with col3:
        st.metric("Avg Electric Consumption",
                  f"{avg_elec_consumption:.2f} kWh/100km")
        st.metric("Avg Fuel Consumption",
                  f"{avg_fuel_consumption:.2f} L/100km")

    st.markdown("---")
    calculate_and_display_savings(
        data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp)

    # Visualizations
    st.markdown("---")
    st.subheader("Visualizations")
    if not data.empty and 'datetime' in data.columns and data['datetime'].notna().any():
        data_sorted = data.sort_values(by='datetime')
        timeline_fig = px.line(data_sorted, x='datetime', y='distance', markers=True,
                               title="Distance per Trip Over Time", labels={'datetime': 'Date', 'distance': 'Distance (km)'})
        timeline_fig.update_traces(marker=dict(size=8))
        st.plotly_chart(timeline_fig, use_container_width=True)

        data_sorted['cumulative_distance'] = data_sorted['distance'].cumsum()
        cumulative_fig = px.line(data_sorted, x='datetime', y='cumulative_distance', markers=True, title="Cumulative Distance Over Time", labels={
                                 'datetime': 'Date', 'cumulative_distance': 'Cumulative Distance (km)'})
        cumulative_fig.update_traces(
            line=dict(color='green', width=2), marker=dict(size=8))
        st.plotly_chart(cumulative_fig, use_container_width=True)
    else:
        st.info("Not enough datetime data to display temporal charts.")

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.dataframe(data[numeric_cols].describe().T.round(
            2), use_container_width=True)
    else:
        st.info("No numeric columns available for descriptive statistics.")


def main():
    st.set_page_config(page_title="Hybrid Car Statistics",
                       page_icon="🚗", layout="wide")

    try:
        file_path = "src/data/car_stats.csv"
        raw_data = load_data(file_path)
        cleaned_data = clean_data(raw_data.copy())
        data = transform_data(cleaned_data)
    except Exception as e:
        st.error(f"Fatal error during data loading or processing: {e}")
        st.stop()

    selection, filtered_data, gas_price, elec_price, gas_comp, elec_comp = create_menu(
        data)

    if selection == "Overview":
        show_overview(filtered_data, gas_price,
                      elec_price, gas_comp, elec_comp)
    elif selection == "Consumption Analysis":
        show_fuel_consumption(filtered_data)
    elif selection == "Trip Details":
        show_trip_details(filtered_data)
    elif selection == "Performance Metrics":
        show_performance(filtered_data)
    elif selection == "Data Explorer":
        show_data_explorer(filtered_data)


if __name__ == "__main__":
    main()
