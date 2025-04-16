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


def show_fuel_consumption(data):
    """Display fuel consumption analysis."""
    st.header("Consumption Analysis")

    if data.empty:
        st.warning(
            "No data available to display for the selected period.")
        return

    # Ensure necessary columns exist and have data
    required_cols = ['electricity_consumption', 'fuel_consumption',
                     'electricity_used_kwh', 'fuel_used_liters', 'distance']
    if not all(col in data.columns for col in required_cols):
        st.error("Missing columns required for consumption analysis.")
        return
    if not all(data[col].notna().any() for col in required_cols):
        st.warning(
            "Some columns required for consumption analysis have no data.")
        # Optionally return or continue with available data

    col1, col2 = st.columns(2)

    with col1:
        # Filter out zero consumption trips for the bar chart unless they are the only data
        plot_data = data[(data['electricity_consumption'] > 0)
                         | (data['fuel_consumption'] > 0)]
        if plot_data.empty:
            plot_data = data  # Show all if filtering leaves nothing

        if not plot_data.empty:
            # Bar chart: Electric vs Fuel consumption per trip
            consumption_fig = px.bar(plot_data.reset_index(), x='index',
                                     y=['electricity_consumption',
                                        'fuel_consumption'],
                                     title='Electric vs. Fuel Consumption per Trip',
                                     labels={'index': 'Trip Number',
                                             'value': 'Consumption', 'variable': 'Type'},
                                     barmode='group')
            consumption_fig.update_layout(
                yaxis_title='Consumption (kWh or L / 100km)')
            st.plotly_chart(consumption_fig, use_container_width=True)
        else:
            st.info("No consumption data available for bar chart.")

    with col2:
        # Pie chart showing proportion of energy types used (in kWh equivalent)
        total_elec_kwh = data['electricity_used_kwh'].sum()
        # Approx conversion: 1 Liter gasoline ~ 9-10 kWh. Using 9.5 for estimation.
        total_fuel_kwh_eq = data['fuel_used_liters'].sum() * 9.5

        if total_elec_kwh > 0 or total_fuel_kwh_eq > 0:
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Electricity (kWh)', 'Fuel (kWh eq.)'],
                values=[total_elec_kwh, total_fuel_kwh_eq],
                hole=.3,
                hoverinfo='label+percent+value',
                textinfo='percent+label'
            )])
            pie_fig.update_layout(
                title_text='Total Energy Usage Distribution (Estimated)')
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No total consumption data available for pie chart.")

    # Consumption vs Distance Scatter Chart
    st.markdown("---")
    st.subheader("Consumption vs Trip Distance")
    if not data.empty:
        scatter_fig = px.scatter(data, x='distance', y=['electricity_consumption', 'fuel_consumption'],
                                 title='Consumption vs Distance',
                                 labels={
            'distance': 'Trip Distance (km)', 'value': 'Consumption (/100km)', 'variable': 'Energy Type'},
            hover_data=['datetime'])  # Add datetime to hover
        # Improve legend names
        scatter_fig.for_each_trace(lambda t: t.update(name=t.name.replace(
            "electricity_consumption", "Electric (kWh/100km)").replace("fuel_consumption", "Fuel (L/100km)")))
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.info(
            "No data available to show Consumption vs Distance scatter chart.")


def show_trip_details(data):
    """Display trip details."""
    st.header("Trip Details")

    if data.empty or 'datetime' not in data.columns:
        st.warning(
            "No trip data available or missing 'datetime' column for the selected period.")
        return

    # Sort data by datetime for selection consistency
    data_sorted = data.sort_values(
        by='datetime', ascending=False).reset_index(drop=True)

    # Format options for the selectbox

    def format_trip_option(index):
        trip_row = data_sorted.iloc[index]
        date_str = trip_row['datetime'].strftime(
            '%Y-%m-%d %H:%M') if pd.notna(trip_row['datetime']) else "Unknown date"
        dist_str = f"{trip_row['distance']} km" if pd.notna(
            trip_row['distance']) else "Unknown dist."
        return f"{index}: {date_str} - {dist_str}"

    trip_index = st.selectbox("Select a Trip (sorted by date descending)",
                              data_sorted.index,  # Use the index of the sorted dataframe
                              format_func=format_trip_option)

    if trip_index is not None and trip_index in data_sorted.index:
        trip = data_sorted.iloc[trip_index]

        st.subheader(
            f"Details for Trip #{trip_index} ({trip['datetime'].strftime('%Y-%m-%d %H:%M')})")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Distance", f"{trip.get('distance', 'N/A')} km")
            st.metric("Driving Time", f"{trip.get('driving_time', 'N/A')}" + (
                f" ({trip.get('driving_minutes', 0)} min)" if trip.get('driving_minutes') else ""))

        with col2:
            st.metric("Average Speed",
                      f"{trip.get('average_speed', 'N/A')} km/h")
            st.metric("Electricity Consumption", f"{trip.get('electricity_consumption', 'N/A'):.2f} kWh/100km" if pd.notna(
                trip.get('electricity_consumption')) else 'N/A')
            st.metric("Electricity Used", f"{trip.get('electricity_used_kwh', 'N/A'):.2f} kWh" if pd.notna(
                trip.get('electricity_used_kwh')) else 'N/A')

        with col3:
            st.metric("Fuel Consumption", f"{trip.get('fuel_consumption', 'N/A'):.2f} L/100km" if pd.notna(
                trip.get('fuel_consumption')) else 'N/A')
            st.metric("Fuel Used", f"{trip.get('fuel_used_liters', 'N/A'):.2f} L" if pd.notna(
                trip.get('fuel_used_liters')) else 'N/A')

        # Compare this trip to average (using the full original dataset for average calculation for context)
        # Or use the filtered dataset average? Let's use filtered dataset avg.
        st.markdown("---")
        st.subheader(
            "Comparison of this Trip with the Selected Period Average")

        # Calculate averages from the 'data' (filtered dataset) passed to this function
        avg_distance = data['distance'].mean()
        avg_speed = data['average_speed'].mean()
        # Avg only of trips using electricity
        avg_elec_cons = data['electricity_consumption'][data['electricity_consumption'] > 0].mean(
        )
        # Avg only of trips using fuel
        avg_fuel_cons = data['fuel_consumption'][data['fuel_consumption'] > 0].mean(
        )
        avg_driving_mins = data['driving_minutes'].mean()

        # Handle cases where average might be NaN or zero (e.g., no electric trips in selection)
        avg_elec_cons = avg_elec_cons if pd.notna(avg_elec_cons) else 0
        avg_fuel_cons = avg_fuel_cons if pd.notna(avg_fuel_cons) else 0

        comparison_data = {
            'Metric': ['Distance (km)', 'Average Speed (km/h)',
                       'Electricity Consumption (kWh/100km)', 'Fuel Consumption (L/100km)',
                       'Driving Time (min)'],
            'This Trip': [trip.get('distance', 0), trip.get('average_speed', 0),
                          trip.get('electricity_consumption', 0), trip.get(
                'fuel_consumption', 0),
                trip.get('driving_minutes', 0)],
            'Period Average': [avg_distance, avg_speed, avg_elec_cons, avg_fuel_cons, avg_driving_mins]
        }

        comp_df = pd.DataFrame(comparison_data)
        comp_df['This Trip'] = pd.to_numeric(
            comp_df['This Trip'], errors='coerce').fillna(0)
        comp_df['Period Average'] = pd.to_numeric(
            comp_df['Period Average'], errors='coerce').fillna(0)

        comp_fig = go.Figure()

        comp_fig.add_trace(go.Bar(
            y=comp_df['Metric'],
            x=comp_df['This Trip'],
            name='This Trip',
            orientation='h',
            marker=dict(color='rgba(58, 71, 80, 0.8)'),  # Darker color
            text=comp_df['This Trip'].round(2),
            textposition='auto'
        ))

        comp_fig.add_trace(go.Bar(
            y=comp_df['Metric'],
            x=comp_df['Period Average'],
            name='Period Average',
            orientation='h',
            marker=dict(color='rgba(246, 78, 139, 0.6)'),  # Lighter color
            text=comp_df['Period Average'].round(2),
            textposition='auto'
        ))

        comp_fig.update_layout(barmode='group', title='Trip vs Period Average Comparison',
                               xaxis_title='Value', yaxis_title='Metric', height=400)
        st.plotly_chart(comp_fig, use_container_width=True)

    else:
        st.info("Select a trip from the list to see its details.")


def show_performance(data):
    """Display performance metrics."""
    st.header("Performance Metrics")

    if data.empty:
        st.warning(
            "No data available to display for the selected period.")
        return

    # Ensure necessary columns exist
    required_cols = ['average_speed', 'electricity_consumption',
                     'fuel_consumption', 'distance', 'driving_minutes', 'datetime']
    if not all(col in data.columns and data[col].notna().any() for col in required_cols):
        st.warning(
            "Missing columns or data required to show performance metrics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # --- Chart 1: Speed vs Electric Consumption ---
        # Prepare data for fig1
        data_fig1 = data.copy()  # Start with a fresh copy

        # Convert essential columns to numeric, coercing errors
        for col in ['average_speed', 'electricity_consumption', 'distance', 'fuel_consumption']:
            if col in data_fig1.columns:
                data_fig1[col] = pd.to_numeric(data_fig1[col], errors='coerce')

        # Drop rows where essential plotting values are NaN
        essential_plot1_cols = [
            'average_speed', 'electricity_consumption', 'distance', 'fuel_consumption']
        data_fig1 = data_fig1.dropna(subset=essential_plot1_cols)

        # *** KEY STEP: Ensure float types for x, y, size, color ***
        if not data_fig1.empty:
            data_fig1['average_speed'] = data_fig1['average_speed'].astype(
                float)
            data_fig1['electricity_consumption'] = data_fig1['electricity_consumption'].astype(
                float)
            data_fig1['distance'] = data_fig1['distance'].astype(
                float)  # Crucial for 'size'
            data_fig1['fuel_consumption'] = data_fig1['fuel_consumption'].astype(
                float)  # Crucial for 'color'

        # Check if data remains after cleaning
        if not data_fig1.empty:
            fig1 = px.scatter(data_fig1, x='average_speed', y='electricity_consumption',
                              size='distance',  # Should now work with float type
                              color='fuel_consumption',
                              color_continuous_scale=px.colors.sequential.Viridis_r,
                              # Ensure these columns exist in data_fig1 if needed
                              hover_data=['datetime', 'driving_time'],
                              title='Speed vs Electric Consumption',
                              labels={'average_speed': 'Average Speed (km/h)',
                                      'electricity_consumption': 'Electric Consumption (kWh/100km)',
                                      'fuel_consumption': 'Fuel Consumption (L/100km)',
                                      'distance': 'Distance (km)'})  # Add label for size if desired
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info(
                "No valid data to display the Speed vs Electric Consumption chart after cleaning.")

    with col2:
        # --- Chart 2: Distance vs Driving Time with Trendline ---
        data_numeric = data[['distance', 'driving_minutes',
                             'average_speed', 'datetime']].copy()
        data_numeric['distance'] = pd.to_numeric(
            data_numeric['distance'], errors='coerce')
        data_numeric['driving_minutes'] = pd.to_numeric(
            data_numeric['driving_minutes'], errors='coerce')
        data_numeric['average_speed'] = pd.to_numeric(
            data_numeric['average_speed'], errors='coerce')
        data_numeric = data_numeric.dropna(
            subset=['distance', 'driving_minutes', 'average_speed'])

        if not data_numeric.empty:
            data_numeric['distance'] = data_numeric['distance'].astype(float)
            data_numeric['driving_minutes'] = data_numeric['driving_minutes'].astype(
                float)

        if len(data_numeric) > 1:
            fig2 = px.scatter(data_numeric, x='distance', y='driving_minutes',
                              trendline='ols',
                              color='average_speed',
                              color_continuous_scale=px.colors.sequential.Plasma,
                              hover_data=['datetime'],
                              title='Distance vs Driving Time',
                              labels={'distance': 'Distance (km)',
                                        'driving_minutes': 'Driving Time (minutes)',
                                        'average_speed': 'Average Speed (km/h)'})
            st.plotly_chart(fig2, use_container_width=True)
        elif not data_numeric.empty:
            st.info(
                "Not enough data (need > 1) to calculate trendline. Showing points only.")
            fig2 = px.scatter(data_numeric, x='distance', y='driving_minutes',
                              color='average_speed',
                              color_continuous_scale=px.colors.sequential.Plasma,
                              hover_data=['datetime'],
                              title='Distance vs Driving Time (no trendline)',
                              labels={'distance': 'Distance (km)',
                                      'driving_minutes': 'Driving Time (minutes)',
                                      'average_speed': 'Average Speed (km/h)'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(
                "No valid numeric data to display the Distance vs Time chart.")

    # --- Chart 3: Performance Trends Over Time ---
    st.markdown("---")
    st.subheader("Performance Trends Over Time")

    data_sorted = data.copy()
    data_sorted['datetime'] = pd.to_datetime(
        data_sorted['datetime'], errors='coerce')
    data_sorted = data_sorted.dropna(subset=['datetime'])
    data_sorted = data_sorted.sort_values('datetime')

    perf_cols = ['electricity_consumption',
                 'fuel_consumption', 'average_speed']
    for col in perf_cols:
        if col in data_sorted.columns:
            data_sorted[col] = pd.to_numeric(data_sorted[col], errors='coerce')

    if not data_sorted.empty and data_sorted['datetime'].notna().any():
        performance_fig = go.Figure()

        if 'electricity_consumption' in data_sorted.columns:
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'],
                y=data_sorted['electricity_consumption'],
                mode='lines+markers',
                name='Electric (kWh/100km)',
                yaxis='y1'
            ))

        if 'fuel_consumption' in data_sorted.columns:
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'],
                y=data_sorted['fuel_consumption'],
                mode='lines+markers',
                name='Fuel (L/100km)',
                yaxis='y1'
            ))

        if 'average_speed' in data_sorted.columns:
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'],
                y=data_sorted['average_speed'],
                mode='lines+markers',
                name='Average Speed (km/h)',
                yaxis='y2'
            ))

        performance_fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Date',
            yaxis=dict(
                title='Consumption (kWh or L / 100km)',
                tickfont=dict(color="#1f77b4")
            ),
            yaxis2=dict(
                title='Average Speed (km/h)',
                tickfont=dict(color="#ff7f0e"),
                overlaying='y',
                side='right',
                showgrid=False,
            ),
            legend=dict(x=0.01, y=1.1, orientation="h")
        )

        st.plotly_chart(performance_fig, use_container_width=True)
    else:
        st.info(
            "Not enough data to display the performance trends chart.")


def show_data_explorer(data):
    """Display data explorer with filtering options."""
    st.header("Data Explorer")

    if data.empty:
        st.warning(
            "No data available to explore for the selected period.")
        return

    st.write("Filter and explore the raw trip data.")

    # Create filter options in columns
    col1, col2, col3 = st.columns(3)

    # Use min/max from the actual data being explored
    min_dist = int(data['distance'].min()) if not data.empty else 0
    max_dist = int(data['distance'].max()) if not data.empty else 1
    with col1:
        distance_range = st.slider(
            "Distance Range (km)", min_dist, max_dist, (min_dist, max_dist))

    min_elec = float(data['electricity_consumption'].min()
                     ) if not data.empty else 0.0
    max_elec = float(data['electricity_consumption'].max()
                     ) if not data.empty else 1.0
    with col2:
        # Add a small epsilon to max_value if min and max are the same
        max_elec_adj = max_elec if max_elec > min_elec else min_elec + 0.1
        elec_range = st.slider("Electric Consumption Range (kWh/100km)",
                               min_elec, max_elec_adj, (min_elec, max_elec), step=0.1)

    min_fuel = float(data['fuel_consumption'].min()) if not data.empty else 0.0
    max_fuel = float(data['fuel_consumption'].max()) if not data.empty else 1.0
    with col3:
        max_fuel_adj = max_fuel if max_fuel > min_fuel else min_fuel + 0.1
        fuel_range = st.slider("Fuel Consumption Range (L/100km)",
                               min_fuel, max_fuel_adj, (min_fuel, max_fuel), step=0.1)

    # Apply filters
    filtered_df = data[
        (data['distance'] >= distance_range[0]) &
        (data['distance'] <= distance_range[1]) &
        (data['electricity_consumption'] >= elec_range[0]) &
        (data['electricity_consumption'] <= elec_range[1]) &
        (data['fuel_consumption'] >= fuel_range[0]) &
        (data['fuel_consumption'] <= fuel_range[1])
    ]

    # Display filtered data
    st.subheader(f"Filtered Data ({len(filtered_df)} records)")

    # Allow choosing columns to display
    # Make default selection more robust, check if columns exist first
    default_cols_options = ['datetime', 'distance', 'electricity_consumption',
                            'fuel_consumption', 'average_speed', 'driving_time',
                            'electricity_used_kwh', 'fuel_used_liters']
    default_cols = [
        col for col in default_cols_options if col in filtered_df.columns]

    all_cols = filtered_df.columns.tolist()
    display_cols = st.multiselect(
        "Select columns to display:",
        all_cols,
        default=default_cols
    )

    if display_cols:
        # Display with selected columns, format datetime nicely
        display_df = filtered_df[display_cols].copy()
        if 'datetime' in display_df.columns:
            display_df['datetime'] = display_df['datetime'].dt.strftime(
                '%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df)
    else:
        # Display all columns if none selected, format datetime
        display_df_all = filtered_df.copy()
        if 'datetime' in display_df_all.columns:
            display_df_all['datetime'] = display_df_all['datetime'].dt.strftime(
                '%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df_all)

    # Provide download button for the filtered data
    if not filtered_df.empty:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="car_stats_filtered.csv",
            mime="text/csv"
        )


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
