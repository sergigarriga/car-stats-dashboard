import os
from datetime import timedelta

import pandas as pd
import streamlit as st


def create_menu(data):
    """Create the sidebar menu and filters for the Streamlit app."""
    st.sidebar.title("Control Panel 🚗")

    # --- File Upload ---
    st.sidebar.subheader("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file", type=["csv"], key="file_uploader"
    )
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.to_csv(
                os.path.join("src", "data", "car_stats.csv"), index=False
            )
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            data = pd.DataFrame()  # Reset to empty DataFrame if error occurs

    # --- Navigation ---
    options = ["Overview", "Consumption Analysis",
               "Trip Details", "Performance Metrics", "Predictions", "Data Explorer"]
    selection = st.sidebar.radio("Navigation", options)

    # --- Date Filter ---
    st.sidebar.subheader("Filter by Date")

    filtered_data = data.copy()  # Start with all data

    # Check if data exists and the datetime column is valid
    if data.empty or 'datetime' not in data.columns or not pd.api.types.is_datetime64_any_dtype(data['datetime']) or data['datetime'].isnull().all():
        st.sidebar.warning("No valid date data available for filtering.")
    else:
        min_date = data['datetime'].min().date()
        max_date = data['datetime'].max().date()

        filter_type = st.sidebar.radio(
            "Date Filter Type",
            ["All Dates", "Date Range", "Specific Days"],
            key="date_filter_type"  # Add key to avoid state issues
        )

        if filter_type == "Date Range":
            start_date = st.sidebar.date_input(
                "From", min_date, min_value=min_date, max_value=max_date, key="start_date")
            end_date = st.sidebar.date_input(
                "To", max_date, min_value=start_date, max_value=max_date, key="end_date")

            if start_date and end_date:
                start_datetime = pd.to_datetime(start_date)
                # Adjust end_datetime to include the entire final day
                end_datetime = pd.to_datetime(
                    end_date) + timedelta(days=1) - timedelta(seconds=1)

                filtered_data = data[
                    (data['datetime'] >= start_datetime) & (
                        data['datetime'] <= end_datetime)
                ]

        elif filter_type == "Specific Days":
            available_dates = sorted(data['datetime'].dt.date.unique())
            if not available_dates:
                st.sidebar.warning("No unique dates available.")
            else:
                selected_dates = st.sidebar.multiselect(
                    "Select Days:", available_dates, default=[], key="multi_date_select"
                )
                if selected_dates:
                    filtered_data = data[data['datetime'].dt.date.isin(
                        selected_dates)]
                else:
                    st.sidebar.info("Select at least one day to view data.")
                    filtered_data = data.iloc[0:0]  # Empty DataFrame

        # If "All Dates" is selected, filtered_data remains the full copy

    # Display the number of selected trips
    st.sidebar.info(f"Showing {len(filtered_data)} of {len(data)} trips")

    # --- Savings Calculation Parameters ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("Savings Comparison Parameters", expanded=False):
        st.subheader("Savings Comparison Parameters")

        # Initialize session state for parameters if not already set
        if "gas_price" not in st.session_state:
            st.session_state["gas_price"] = 1.55
        if "elec_price" not in st.session_state:
            st.session_state["elec_price"] = 0.18
        if "gas_comp" not in st.session_state:
            st.session_state["gas_comp"] = 6.5
        if "elec_comp" not in st.session_state:
            st.session_state["elec_comp"] = 17.0

        # Use session state for inputs and update session state on change
        st.session_state["gas_price"] = st.number_input(
            "Gasoline Price (€/L)",
            min_value=0.1, max_value=3.0,
            value=st.session_state["gas_price"],
            step=0.01, format="%.2f",
            key="gas_price_input"
        )

        st.session_state["elec_price"] = st.number_input(
            "Electricity Price (€/kWh)",
            min_value=0.01, max_value=1.0,
            value=st.session_state["elec_price"],
            step=0.01, format="%.3f",
            key="elec_price_input"
        )

        st.session_state["gas_comp"] = st.number_input(
            "Gasoline Car Consumption (L/100km)",
            min_value=1.0, max_value=20.0,
            value=st.session_state["gas_comp"],
            step=0.1, format="%.1f",
            key="gas_comp_input"
        )

        st.session_state["elec_comp"] = st.number_input(
            "Electric Car Consumption (kWh/100km)",
            min_value=5.0, max_value=40.0,
            value=st.session_state["elec_comp"],
            step=0.1, format="%.1f",
            key="elec_comp_input"
        )

    # Return all necessary values for main.py
    return selection, filtered_data, st.session_state["gas_price"], st.session_state["elec_price"], st.session_state["gas_comp"], st.session_state["elec_comp"]
