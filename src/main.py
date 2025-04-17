import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.menu import create_menu
from utils.data_processing import clean_data, load_data, transform_data
from utils.predictions import predict_next_month_stats

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def initialize_ai_model():
    """Initialize the AI model and handle its availability."""
    try:
        from utils.ai_caller import ask_ia_model
        AI_AVAILABLE = True
    except ImportError:
        AI_AVAILABLE = False
        st.warning(
            "AI model is not available. Some features may be limited.", icon="⚠️")
        logger.warning("AI model is not available.")

        def ask_ia_model(prompt: str, model: str = "gpt-4o", max_tokens: int = 1000, temperature: float = 0.1):
            return {
                "error": "Function ask_ia_model is not available.",
                "choices": [{"message": {"content": "Error: The AI function is not available."}}]
            }
    return AI_AVAILABLE, ask_ia_model


@st.cache_data
def generate_explanation(AI_AVAILABLE: bool, chart_description: str, summary_data: str, chart_type: str = "this chart"):
    """Generates an explanation for a chart using the AI model."""
    logger.info(f"Generating explanation for {chart_type}.")
    if not AI_AVAILABLE:
        logger.warning("AI explanations are not available.")
        return "AI explanations are not available."

    prompt = f"""
You are an expert assistant in analyzing hybrid car driving data.
A user is viewing a dashboard with their driving statistics.
Below {chart_type}, a simple and clear explanation is needed for a non-technical audience.

Chart description: {chart_description}
Key data or summary statistics displayed in the chart:
{summary_data}

Please generate a concise explanation (2-4 sentences) about what the chart shows and the main conclusions the user can draw about their driving habits or their car's performance based on this data.
Focus on the perspective of a hybrid car owner. Use plain language.
Additionally, if real data insights are provided in the summary data, incorporate them into the explanation to make it more personalized and actionable.
Format: Plain text or light Markdown.
"""
    try:
        with st.spinner(f"🧠 Analyzing {chart_type}..."):
            response = ask_ia_model(
                prompt=prompt, model="gpt-4o", max_tokens=200, temperature=0.2)

        if "error" in response:
            logger.error(f"AI API Error: {response['error']}")
            st.error(f"AI API Error: {response['error']}")
            return "Unable to generate the explanation due to an error."
        elif response and 'choices' in response and len(response['choices']) > 0:
            explanation = response['choices'][0]['message']['content']
            logger.info(
                f"Explanation generated successfully for {chart_type}.")
            return explanation
        else:
            logger.warning("The AI response did not have the expected format.")
            st.warning("The AI response did not have the expected format.")
            return "Could not retrieve a valid explanation."

    except Exception as e:
        logger.exception("Error while calling the AI model.")
        st.error(f"Error while calling the AI model: {e}")
        return "An error occurred while generating the explanation."


def calculate_and_display_savings(data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp):
    """Calculate and display cost estimations, savings, and CO2 emissions."""
    logger.info("Calculating and displaying savings and CO2 emissions.")
    st.header("Cost, Savings, and CO2 Emissions Estimation")

    if data.empty:
        logger.warning(
            "No data available for the selected period to calculate savings and emissions.")
        st.warning(
            "No data available for the selected period to calculate savings and emissions.")
        return

    total_distance = data['distance'].sum()
    logger.info(f"Total distance: {total_distance} km.")
    total_elec_kwh = data['electricity_used_kwh'].sum()
    total_fuel_L = data['fuel_used_liters'].sum()

    if total_distance == 0:
        logger.warning(
            "Total distance for the selected period is 0 km. Savings and emissions cannot be calculated.")
        st.warning(
            "Total distance for the selected period is 0 km. Savings and emissions cannot be calculated.")
        return

    # Cost calculations
    your_cost = (total_elec_kwh * elec_price) + (total_fuel_L * gas_price)
    gas_car_cost = (total_distance / 100) * gas_consumption_comp * gas_price
    elec_car_cost = (total_distance / 100) * elec_consumption_comp * elec_price

    # Savings
    savings_vs_gas = gas_car_cost - your_cost
    savings_vs_elec = elec_car_cost - your_cost

    # CO2 emissions calculations
    CO2_PER_LITER_GASOLINE = 2.31  # kg CO2 per liter of gasoline
    CO2_PER_KWH_ELECTRICITY = 0.25  # kg CO2 per kWh (average for Spain)

    hybrid_emissions = (total_fuel_L * CO2_PER_LITER_GASOLINE) + \
        (total_elec_kwh * CO2_PER_KWH_ELECTRICITY)
    gas_car_emissions = (total_distance / 100) * \
        gas_consumption_comp * CO2_PER_LITER_GASOLINE
    emissions_saved = gas_car_emissions - hybrid_emissions

    logger.info(
        f"Hybrid emissions: {hybrid_emissions:.2f} kg CO2, Gasoline car emissions: {gas_car_emissions:.2f} kg CO2, Emissions saved: {emissions_saved:.2f} kg CO2.")

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
        if savings_vs_gas > 0:
            st.success(
                f"You saved {savings_vs_gas:.2f} € compared to a gasoline car.")
        else:
            st.warning(
                f"You spent {abs(savings_vs_gas):.2f} € more compared to a gasoline car.")
    with col5:
        st.metric("Savings vs. Electric Car", f"{savings_vs_elec:.2f} €")
        if savings_vs_elec > 0:
            st.success(
                f"You saved {savings_vs_elec:.2f} € compared to an electric car.")
        else:
            st.warning(
                f"You spent {abs(savings_vs_elec):.2f} € more compared to an electric car.")

    st.markdown("---")
    st.subheader("CO2 Emissions")
    col6, col7, col8 = st.columns(3)
    with col6:
        st.metric("Hybrid Car Emissions", f"{hybrid_emissions:.2f} kg CO2")
    with col7:
        st.metric("Gasoline Car Emissions", f"{gas_car_emissions:.2f} kg CO2")
    with col8:
        st.metric("Emissions Saved", f"{emissions_saved:.2f} kg CO2")
        if emissions_saved > 0:
            st.success(
                f"You saved {emissions_saved:.2f} kg CO2 by using your hybrid car.")
        else:
            st.warning(
                f"You emitted {abs(emissions_saved):.2f} kg CO2 more compared to a gasoline car.")

    st.caption(
        f"CO2 calculations based on: {CO2_PER_LITER_GASOLINE} kg CO2/L for gasoline and {CO2_PER_KWH_ELECTRICITY} kg CO2/kWh for electricity.")


def show_overview(data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp):
    """Display an overview of statistics and savings."""
    logger.info("Displaying usage overview.")
    st.header("Usage Overview")

    if data.empty:
        logger.warning("No data available for the selected period.")
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

    logger.info(
        f"Total distance: {total_distance} km, Total driving time: {total_minutes} minutes, Average speed: {avg_speed:.1f} km/h.")
    logger.info(
        f"Average electric consumption: {avg_elec_consumption:.2f} kWh/100km, Average combustion engine consumption: {avg_fuel_consumption:.2f} L/100km.")

    with col1:
        st.metric("Total Trips", len(data))
        st.metric("Total Distance", f"{total_distance} km")
    with col2:
        st.metric("Total Driving Time",
                  f"{total_minutes // 60}h {total_minutes % 60}min")
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
    with col3:
        st.metric("Avg. Electric Consumption",
                  f"{avg_elec_consumption:.2f} kWh/100km")
        st.metric("Avg. Fuel Consumption",
                  f"{avg_fuel_consumption:.2f} L/100km")

    st.markdown("---")
    calculate_and_display_savings(
        data, gas_price, elec_price, gas_consumption_comp, elec_consumption_comp)

    st.markdown("---")
    st.subheader("Visualizations")
    if not data.empty and 'datetime' in data.columns and data['datetime'].notna().any():
        data_sorted = data.sort_values(by='datetime')

        chart_desc_1 = "Line chart showing the distance (Y-axis, in km) of each individual trip over time (X-axis)."
        summary_data_1 = f"Number of trips: {len(data_sorted)}. Average trip distance: {data_sorted['distance'].mean():.1f} km. Max distance: {data_sorted['distance'].max()} km. Min distance: {data_sorted['distance'].min()} km."
        timeline_fig = px.line(data_sorted, x='datetime', y='distance', markers=True,
                               title="Distance per Trip Over Time", labels={'datetime': 'Date', 'distance': 'Distance (km)'})
        timeline_fig.update_traces(marker=dict(size=8))
        st.plotly_chart(timeline_fig, use_container_width=True)
        if st.button("🧠 Generate Explanation for Distance per Trip Chart", key="explain_chart_1"):
            st.markdown(generate_explanation(
                AI_AVAILABLE, chart_desc_1, summary_data_1 + f"\n\nData: {data_sorted[['datetime', 'distance']].to_dict(orient='records')}", "the distance per trip chart"))

        chart_desc_2 = "Line chart showing how the total distance traveled (Y-axis, in km) accumulates over time (X-axis)."
        summary_data_2 = f"Total distance covered in period: {data_sorted['distance'].sum()} km. Period start date: {data_sorted['datetime'].min().date()}. Period end date: {data_sorted['datetime'].max().date()}."
        data_sorted['cumulative_distance'] = data_sorted['distance'].cumsum()
        cumulative_fig = px.line(data_sorted, x='datetime', y='cumulative_distance', markers=True, title="Cumulative Distance Over Time", labels={
                                 'datetime': 'Date', 'cumulative_distance': 'Cumulative Distance (km)'})
        cumulative_fig.update_traces(
            line=dict(color='green', width=2), marker=dict(size=8))
        st.plotly_chart(cumulative_fig, use_container_width=True)
        if st.button("🧠 Generate Explanation for Cumulative Distance Chart", key="explain_chart_2"):
            st.markdown(generate_explanation(
                chart_desc_2, summary_data_2 + f"\n\nData: {data_sorted[['datetime', 'cumulative_distance']].to_dict(orient='records')}", "the cumulative distance chart"))

    else:
        logger.info("Not enough datetime data to display temporal charts.")
        st.info("Not enough datetime data to display temporal charts.")

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats_df = data[numeric_cols].describe().T.round(2)
        st.dataframe(stats_df, use_container_width=True)

        chart_desc_stats = "Table showing descriptive statistics (mean, standard deviation, min, max, etc.) for all numeric columns in the dataset."
        summary_data_stats = f"Columns analyzed: {', '.join(numeric_cols)}. Key statistics include mean, standard deviation, and range for each column."
        if st.button("🧠 Generate Explanation for Descriptive Statistics Table", key="explain_chart_stats"):
            st.markdown(generate_explanation(
                chart_desc_stats, summary_data_stats + f"\n\nData: {stats_df.to_dict(orient='index')}", "the descriptive statistics table"))
    else:
        logger.info("No numeric columns available for descriptive statistics.")
        st.info("No numeric columns available for descriptive statistics.")


def show_fuel_consumption(data):
    """Display fuel consumption analysis."""
    logger.info("Displaying fuel consumption analysis.")
    st.header("Consumption Analysis")

    if data.empty:
        logger.warning("No data available to display for the selected period.")
        st.warning(
            "No data available to display for the selected period.")
        return

    required_cols = ['electricity_consumption', 'fuel_consumption',
                     'electricity_used_kwh', 'fuel_used_liters', 'distance', 'datetime']
    if not all(col in data.columns for col in required_cols):
        logger.error("Missing columns required for consumption analysis.")
        st.error("Missing columns required for consumption analysis.")
        return
    if not all(data[col].notna().any() for col in required_cols):
        logger.warning(
            "Some columns required for consumption analysis have no data.")

    col1, col2 = st.columns(2)

    with col1:
        plot_data = data[(data['electricity_consumption'] > 0)
                         | (data['fuel_consumption'] > 0)]
        if plot_data.empty:
            plot_data = data

        if not plot_data.empty:
            chart_desc_3 = "Grouped bar chart comparing average electric consumption (kWh/100km) and fuel consumption (L/100km) for each individual trip."
            avg_elec = plot_data['electricity_consumption'][plot_data['electricity_consumption'] > 0].mean(
            ) or 0
            avg_fuel = plot_data['fuel_consumption'][plot_data['fuel_consumption'] > 0].mean(
            ) or 0
            summary_data_3 = f"Average electric consumption (on trips with >0): {avg_elec:.2f} kWh/100km. Average Combustion Engine Consumption (on trips with >0): {avg_fuel:.2f} L/100km."

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
            if st.button("🧠 Generate Explanation for Consumption Bar Chart", key="explain_chart_3"):
                st.markdown(generate_explanation(
                    chart_desc_3, summary_data_3 + f"\n\nPlot Data: {plot_data[['electricity_consumption', 'fuel_consumption']].to_dict(orient='records')}", "the consumption bar chart"))
        else:
            logger.info("No consumption data available for bar chart.")
            st.info("No consumption data available for bar chart.")

    with col2:
        total_elec_kwh = data['electricity_used_kwh'].sum()
        total_fuel_L = data['fuel_used_liters'].sum()
        total_fuel_kwh_eq = total_fuel_L * 9.5

        if total_elec_kwh > 0 or total_fuel_kwh_eq > 0:
            chart_desc_4 = "Pie chart showing the proportion of total consumed energy (in equivalent kWh) coming from electricity versus fuel."
            total_energy_kwh_eq = total_elec_kwh + total_fuel_kwh_eq
            perc_elec = (total_elec_kwh / total_energy_kwh_eq *
                         100) if total_energy_kwh_eq > 0 else 0
            perc_fuel = (total_fuel_kwh_eq / total_energy_kwh_eq *
                         100) if total_energy_kwh_eq > 0 else 0
            summary_data_4 = f"Total equivalent energy: {total_energy_kwh_eq:.1f} kWh. Electricity Share: {perc_elec:.1f}%. Fuel Share: {perc_fuel:.1f}%."

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
            if st.button("🧠 Generate Explanation for Energy Distribution Pie Chart", key="explain_chart_4"):
                st.markdown(generate_explanation(
                    chart_desc_4, summary_data_4 + f"\n\nPlot Data: {{'Electricity (kWh)': {total_elec_kwh}, 'Fuel (kWh eq.)': {total_fuel_kwh_eq}}}", "the energy distribution pie chart"))
        else:
            logger.info("No total consumption data available for pie chart.")
            st.info("No total consumption data available for pie chart.")

    st.markdown("---")
    st.subheader("Consumption vs Trip Distance")
    if not data.empty:
        chart_desc_5 = "Scatter plot showing the relationship between trip distance (X-axis, km) and average consumption (Y-axis, electric in kWh/100km and fuel in L/100km)."
        summary_data_5 = f"Visualizing {len(data)} trips. Helps identify if longer trips tend to have different consumption rates (higher or lower efficiency)."

        scatter_fig = px.scatter(data, x='distance', y=['electricity_consumption', 'fuel_consumption'],
                                 title='Consumption vs Distance',
                                 labels={
            'distance': 'Trip Distance (km)', 'value': 'Consumption (/100km)', 'variable': 'Energy Type'},
            hover_data=['datetime'])
        scatter_fig.for_each_trace(lambda t: t.update(name=t.name.replace(
            "electricity_consumption", "Electric (kWh/100km)").replace("fuel_consumption", "Fuel (L/100km)")))
        st.plotly_chart(scatter_fig, use_container_width=True)
        if st.button("🧠 Generate Explanation for Consumption vs Distance Scatter Chart", key="explain_chart_5"):
            st.markdown(generate_explanation(
                chart_desc_5, summary_data_5 + f"\n\nPlot Data: {data[['distance', 'electricity_consumption', 'fuel_consumption']].to_dict(orient='records')}", "the consumption vs distance scatter plot"))
    else:
        logger.info(
            "No data available to show Consumption vs Distance scatter chart.")
        st.info(
            "No data available to show Consumption vs Distance scatter chart.")


def show_trip_details(data):
    """Display trip details."""
    logger.info("Displaying trip details.")
    st.header("Trip Details")

    if data.empty or 'datetime' not in data.columns:
        logger.warning("No trip data available or missing 'datetime' column.")
        st.warning(
            "No trip data available or missing 'datetime' column for the selected period.")
        return

    data_sorted = data.sort_values(
        by='datetime', ascending=False).reset_index(drop=True)

    def format_trip_option(index):
        trip_row = data_sorted.iloc[index]
        date_str = trip_row['datetime'].strftime(
            '%Y-%m-%d %H:%M') if pd.notna(trip_row['datetime']) else "Unknown date"
        dist_str = f"{trip_row['distance']} km" if pd.notna(
            trip_row['distance']) else "Unknown dist."
        return f"{index}: {date_str} - {dist_str}"

    trip_index = st.selectbox("Select a Trip (sorted by date descending)",
                              data_sorted.index,
                              format_func=format_trip_option)

    if trip_index is not None and trip_index in data_sorted.index:
        trip = data_sorted.iloc[trip_index]

        logger.info(f"Displaying details for Trip #{trip_index}.")
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

        st.markdown("---")
        st.subheader(
            "Comparison of this Trip with the Selected Period Average")

        avg_distance = data['distance'].mean()
        avg_speed = data['average_speed'].mean()
        avg_elec_cons = data['electricity_consumption'][data['electricity_consumption'] > 0].mean(
        ) or 0
        avg_fuel_cons = data['fuel_consumption'][data['fuel_consumption'] > 0].mean(
        ) or 0
        avg_driving_mins = data['driving_minutes'].mean() or 0

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
            y=comp_df['Metric'], x=comp_df['This Trip'], name='This Trip',
            orientation='h', marker=dict(color='rgba(58, 71, 80, 0.8)'),
            text=comp_df['This Trip'].round(2), textposition='auto'
        ))
        comp_fig.add_trace(go.Bar(
            y=comp_df['Metric'], x=comp_df['Period Average'], name='Period Average',
            orientation='h', marker=dict(color='rgba(246, 78, 139, 0.6)'),
            text=comp_df['Period Average'].round(2), textposition='auto'
        ))
        comp_fig.update_layout(barmode='group', title='Trip vs Period Average Comparison',
                               xaxis_title='Value', yaxis_title='Metric', height=400)
        st.plotly_chart(comp_fig, use_container_width=True)

    else:
        logger.info("Select a trip from the list to see its details.")
        st.info("Select a trip from the list to see its details.")


def show_performance(data):
    """Display performance metrics."""
    logger.info("Displaying performance metrics.")
    st.header("Performance Metrics")

    if data.empty:
        logger.warning("No data available to display for the selected period.")
        st.warning(
            "No data available to display for the selected period.")
        return

    required_cols = ['average_speed', 'electricity_consumption',
                     'fuel_consumption', 'distance', 'driving_minutes', 'datetime']
    if not all(col in data.columns and data[col].notna().any() for col in required_cols):
        logger.warning(
            "Missing columns or data required to show performance metrics.")
        st.warning(
            "Missing columns or data required to show performance metrics.")
        return

    col1, col2 = st.columns(2)

    with col1:
        data_fig1 = data.copy()
        for col in ['average_speed', 'electricity_consumption', 'distance', 'fuel_consumption', 'datetime']:
            if col in data_fig1.columns:
                data_fig1[col] = pd.to_numeric(data_fig1[col], errors='coerce')

        essential_plot1_cols = [
            'average_speed', 'electricity_consumption', 'distance', 'fuel_consumption', 'datetime']
        data_fig1 = data_fig1.dropna(subset=essential_plot1_cols)

        if not data_fig1.empty:
            data_fig1['average_speed'] = data_fig1['average_speed'].astype(
                float)
            data_fig1['electricity_consumption'] = data_fig1['electricity_consumption'].astype(
                float)
            data_fig1['distance'] = data_fig1['distance'].astype(float)
            data_fig1['fuel_consumption'] = data_fig1['fuel_consumption'].astype(
                float)

            chart_desc_6 = "Scatter plot showing the relationship between average trip speed (X-axis) and average electric consumption (Y-axis). Point size indicates trip distance, and color indicates fuel consumption."
            summary_data_6 = f"Analyzing {len(data_fig1)} trips. Helps understand how speed affects electric efficiency, and how distance/fuel use interact."

            fig1 = px.scatter(data_fig1, x='average_speed', y='electricity_consumption',
                              size='distance', color='fuel_consumption',
                              color_continuous_scale=px.colors.sequential.Viridis_r,
                              hover_data=['datetime', 'driving_time'],
                              title='Speed vs Electric Consumption',
                              labels={'average_speed': 'Average Speed (km/h)',
                                      'electricity_consumption': 'Electric Consumption (kWh/100km)',
                                      'fuel_consumption': 'Fuel Consumption (L/100km)',
                                      'distance': 'Distance (km)'})
            st.plotly_chart(fig1, use_container_width=True)
            if st.button("🧠 Generate Explanation for Speed vs Electric Consumption Chart", key="explain_chart_6"):
                st.markdown(generate_explanation(
                    chart_desc_6, summary_data_6 + f"\n\nPlot Data: {data_fig1[['average_speed', 'electricity_consumption', 'distance', 'fuel_consumption', 'datetime', 'driving_time']].to_dict(orient='records')}", "the speed vs electric consumption chart"))

        else:
            logger.info(
                "No valid data to display the Speed vs Electric Consumption chart after cleaning.")
            st.info(
                "No valid data to display the Speed vs Electric Consumption chart after cleaning.")

    with col2:
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
            chart_desc_7 = "Scatter plot showing the relationship between trip distance (X-axis) and driving time in minutes (Y-axis). Includes an OLS trendline indicating the general linear relationship. Point color indicates average speed."
            summary_data_7 = f"Analyzing {len(data_numeric)} trips. The trendline helps visualize the consistency between distance and time (speed)."

            fig2 = px.scatter(data_numeric, x='distance', y='driving_minutes',
                              trendline='ols', color='average_speed',
                              color_continuous_scale=px.colors.sequential.Plasma,
                              hover_data=['datetime'],
                              title='Distance vs Driving Time',
                              labels={'distance': 'Distance (km)',
                                      'driving_minutes': 'Driving Time (minutes)',
                                      'average_speed': 'Average Speed (km/h)'})
            st.plotly_chart(fig2, use_container_width=True)
            if st.button("🧠 Generate Explanation for Distance vs Driving Time Chart", key="explain_chart_7"):
                st.markdown(generate_explanation(
                    chart_desc_7, summary_data_7 + f"\n\nPlot Data: {data_numeric[['distance', 'driving_minutes', 'average_speed', 'datetime']].to_dict(orient='records')}", "the distance vs time chart"))

        elif not data_numeric.empty:
            logger.info(
                "Not enough data (need > 1) to calculate trendline. Showing points only.")
            st.info(
                "Not enough data (need > 1) to calculate trendline. Showing points only.")
            fig2 = px.scatter(data_numeric, x='distance', y='driving_minutes',
                              color='average_speed', color_continuous_scale=px.colors.sequential.Plasma,
                              hover_data=['datetime'], title='Distance vs Driving Time (no trendline)',
                              labels={'distance': 'Distance (km)', 'driving_minutes': 'Driving Time (minutes)', 'average_speed': 'Average Speed (km/h)'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            logger.info(
                "No valid numeric data to display the Distance vs Time chart.")
            st.info("No valid numeric data to display the Distance vs Time chart.")

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
        chart_desc_8 = "Line chart showing the evolution of electric consumption (kWh/100km, left Y-axis), fuel consumption (L/100km, left Y-axis), and average speed (km/h, right Y-axis) over time (X-axis)."
        summary_data_8 = f"Period: {data_sorted['datetime'].min().date()} to {data_sorted['datetime'].max().date()}. Helps identify changes in efficiency or driving style over time."

        performance_fig = go.Figure()
        if 'electricity_consumption' in data_sorted.columns and data_sorted['electricity_consumption'].notna().any():
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'], y=data_sorted['electricity_consumption'], mode='lines+markers', name='Electric (kWh/100km)', yaxis='y1'))
        if 'fuel_consumption' in data_sorted.columns and data_sorted['fuel_consumption'].notna().any():
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'], y=data_sorted['fuel_consumption'], mode='lines+markers', name='Fuel (L/100km)', yaxis='y1'))
        if 'average_speed' in data_sorted.columns and data_sorted['average_speed'].notna().any():
            performance_fig.add_trace(go.Scatter(
                x=data_sorted['datetime'], y=data_sorted['average_speed'], mode='lines+markers', name='Average Speed (km/h)', yaxis='y2'))

        if performance_fig.data:
            performance_fig.update_layout(
                title='Performance Metrics Over Time', xaxis_title='Date',
                yaxis=dict(title='Consumption (kWh or L / 100km)',
                           tickfont=dict(color="#1f77b4")),
                yaxis2=dict(title='Average Speed (km/h)', tickfont=dict(color="#ff7f0e"),
                            overlaying='y', side='right', showgrid=False),
                legend=dict(x=0.01, y=1.1, orientation="h")
            )
            st.plotly_chart(performance_fig, use_container_width=True)
            if st.button("🧠 Generate Explanation for Performance Trends Chart", key="explain_chart_8"):
                st.markdown(generate_explanation(
                    chart_desc_8, summary_data_8 + f"\n\nPlot Data: {data_sorted[['datetime', 'electricity_consumption', 'fuel_consumption', 'average_speed']].to_dict(orient='records')}", "the performance trends chart"))
        else:
            logger.info(
                "No valid data series found to plot performance trends over time.")
            st.info(
                "No valid data series found to plot performance trends over time.")

    else:
        logger.info("Not enough data to display the performance trends chart.")
        st.info("Not enough data to display the performance trends chart.")


def show_data_explorer(data):
    """Display data explorer with filtering options."""
    logger.info("Displaying data explorer.")
    st.header("Data Explorer")

    if data.empty:
        logger.warning("No data available to explore for the selected period.")
        st.warning(
            "No data available to explore for the selected period.")
        return

    st.write("Filter and explore the raw trip data.")

    col1, col2, col3 = st.columns(3)

    min_dist = int(data['distance'].min(
    )) if not data.empty and data['distance'].notna().any() else 0
    max_dist = int(data['distance'].max(
    )) if not data.empty and data['distance'].notna().any() else 1
    max_dist = max(min_dist, max_dist)
    with col1:
        distance_range = st.slider(
            "Distance Range (km)", min_dist, max_dist, (min_dist, max_dist), key="dist_slider")

    min_elec = float(data['electricity_consumption'].min(
    )) if not data.empty and data['electricity_consumption'].notna().any() else 0.0
    max_elec = float(data['electricity_consumption'].max(
    )) if not data.empty and data['electricity_consumption'].notna().any() else 1.0
    max_elec = max(min_elec, max_elec)
    with col2:
        max_elec_adj = max_elec if max_elec > min_elec else min_elec + 0.1
        elec_range = st.slider("Electric Consumption Range (kWh/100km)",
                               min_elec, max_elec_adj, (min_elec, max_elec), step=0.1, key="elec_slider")

    min_fuel = float(data['fuel_consumption'].min(
    )) if not data.empty and data['fuel_consumption'].notna().any() else 0.0
    max_fuel = float(data['fuel_consumption'].max(
    )) if not data.empty and data['fuel_consumption'].notna().any() else 1.0
    max_fuel = max(min_fuel, max_fuel)
    with col3:
        max_fuel_adj = max_fuel if max_fuel > min_fuel else min_fuel + 0.1
        fuel_range = st.slider("Fuel Consumption Range (L/100km)",
                               min_fuel, max_fuel_adj, (min_fuel, max_fuel), step=0.1, key="fuel_slider")

    filtered_df = data.copy()
    if 'distance' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['distance'].between(
            distance_range[0], distance_range[1])]
    if 'electricity_consumption' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['electricity_consumption'].between(
            elec_range[0], elec_range[1])]
    if 'fuel_consumption' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['fuel_consumption'].between(
            fuel_range[0], fuel_range[1])]

    logger.info(f"Filtered data contains {len(filtered_df)} records.")

    st.subheader(f"Filtered Data ({len(filtered_df)} records)")

    default_cols_options = ['datetime', 'distance', 'electricity_consumption', 'fuel_consumption',
                            'average_speed', 'driving_time', 'electricity_used_kwh', 'fuel_used_liters']
    default_cols = [
        col for col in default_cols_options if col in filtered_df.columns]
    all_cols = filtered_df.columns.tolist()
    display_cols = st.multiselect(
        "Select columns to display:", all_cols, default=default_cols, key="explorer_cols")

    display_df_to_show = filtered_df[display_cols].copy(
    ) if display_cols else filtered_df.copy()

    if 'datetime' in display_df_to_show.columns:
        display_df_to_show['datetime'] = pd.to_datetime(
            display_df_to_show['datetime'], errors='coerce')
        valid_dates_mask = display_df_to_show['datetime'].notna()
        display_df_to_show.loc[valid_dates_mask, 'datetime'] = display_df_to_show.loc[valid_dates_mask,
                                                                                      'datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(display_df_to_show, use_container_width=True)

    if not filtered_df.empty:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv_data,
            file_name="car_stats_filtered.csv",
            mime="text/csv",
            key="download_csv"
        )


def get_historical_monthly_aggregates(df: pd.DataFrame, gas_price: float, elec_price: float) -> pd.DataFrame:
    """Calculates historical monthly aggregates for plotting."""
    if df.empty or 'datetime' not in df.columns:
        return pd.DataFrame()  # Return empty DataFrame

    df_hist = df.copy()
    df_hist['datetime'] = pd.to_datetime(df_hist['datetime'], errors='coerce')
    df_hist = df_hist.dropna(subset=['datetime'])

    if df_hist.empty:
        return pd.DataFrame()

    df_hist['year_month'] = df_hist['datetime'].dt.to_period('M')

    monthly_aggregates = df_hist.groupby('year_month').agg(
        monthly_distance=pd.NamedAgg(column='distance', aggfunc='sum'),
        monthly_elec_kwh=pd.NamedAgg(
            column='electricity_used_kwh', aggfunc='sum'),
        monthly_fuel_l=pd.NamedAgg(column='fuel_used_liters', aggfunc='sum')
    ).reset_index()  # Reset index to access year_month easily

    # Calculate monthly cost based on aggregates and current prices
    monthly_aggregates['monthly_cost'] = (
        (monthly_aggregates['monthly_elec_kwh'] * elec_price) +
        (monthly_aggregates['monthly_fuel_l'] * gas_price)
    )

    # Convert Period to timestamp for plotting (use start of the month)
    monthly_aggregates['month_start'] = monthly_aggregates['year_month'].dt.to_timestamp()

    return monthly_aggregates


def show_predictions(all_data, gas_price, elec_price):
    """Displays simple predictions based on historical data with improved visuals."""
    logger.info("Displaying future projections.")
    st.header("Future Projections (Simple Experimental)")
    st.info("""
        **Disclaimer:** These estimates are based on your historical averages.
        Actual results may vary due to changes in habits, routes, or energy prices.
        At least 2 full months of data are required for calculations.
    """)

    if all_data.empty:
        logger.warning("No historical data available to generate predictions.")
        st.warning("No historical data available to generate predictions.")
        return

    try:
        logger.info("Generating predictions based on historical data.")
        # Pass a copy to avoid modifying the original dataframe outside this scope
        predictions = predict_next_month_stats(
            all_data.copy(), gas_price, elec_price)

        if predictions:
            logger.info("Predictions generated successfully.")
            st.subheader("Predicted Statistics for Next Month")
            st.caption(
                f"Based on average of the last {predictions.get('num_months_used', 'N/A')} months."
            )

            # Display key predicted metrics
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            with col_pred1:
                st.metric("Predicted Total Cost",
                          f"{predictions['predicted_total_cost']:.2f} €")
            with col_pred2:
                st.metric("Predicted Distance",
                          f"{predictions['predicted_distance']:.0f} km")
            with col_pred3:
                # Placeholder or another key metric like avg cost/km
                pass  # Keep layout clean or add another relevant metric

            # Show cost breakdown in an expander with improved layout and formatting
            with st.expander("See Predicted Cost Breakdown", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    predicted_elec_cost = predictions['predicted_electricity_kwh'] * elec_price
                    st.metric("Electricity Cost",
                              f"{predicted_elec_cost:.2f} €")
                    st.metric(
                        "Electricity Usage", f"{predictions['predicted_electricity_kwh']:.1f} kWh")
                with col2:
                    predicted_fuel_cost = predictions['predicted_fuel_l'] * gas_price
                    st.metric("Fuel Cost", f"{predicted_fuel_cost:.2f} €")
                    st.metric("Fuel Usage",
                              f"{predictions['predicted_fuel_l']:.1f} L")

            # --- Enhanced Historical Plotting ---
            try:
                logger.info(
                    "Preparing historical monthly aggregates for plots.")
                monthly_history = get_historical_monthly_aggregates(
                    all_data.copy(), gas_price, elec_price)

                if not monthly_history.empty and len(monthly_history) >= 1:
                    st.markdown("---")
                    st.subheader("Historical Monthly Trends & Prediction")

                    # --- Cost Plot with Prediction ---
                    fig_cost = px.line(
                        monthly_history, x='month_start', y='monthly_cost', markers=True,
                        # title="Estimated Monthly Cost Over Time with Prediction", # Title moved above
                        labels={'month_start': 'Month',
                                'monthly_cost': 'Estimated Cost (€)'}
                    )
                    fig_cost.update_traces(
                        line=dict(color='royalblue'), name='Historical Cost')

                    # Calculate next month's date for plotting the prediction
                    last_month_date = monthly_history['month_start'].max()
                    next_month_date = last_month_date + pd.DateOffset(months=1)

                    # Add prediction point as a separate trace
                    fig_cost.add_trace(go.Scatter(
                        x=[next_month_date],
                        y=[predictions['predicted_total_cost']],
                        mode='markers+text',  # Show marker and text
                        marker=dict(color='red', size=10, symbol='star'),
                        text=[
                            f"Pred: {predictions['predicted_total_cost']:.2f}€"],
                        textposition="top center",
                        name='Prediction'
                    ))

                    # Optional: Add average line
                    avg_hist_cost = monthly_history['monthly_cost'].mean()
                    fig_cost.add_hline(y=avg_hist_cost, line_dash="dash", line_color="grey",
                                       annotation_text=f"Avg: {avg_hist_cost:.2f}€",
                                       annotation_position="bottom right")

                    fig_cost.update_layout(
                        title="Estimated Monthly Cost: History & Prediction", showlegend=True)
                    st.plotly_chart(fig_cost, use_container_width=True)
                    logger.info(
                        "Historical cost trend plot with prediction generated successfully.")

                    # --- Other Historical Plots ---
                    st.markdown("---")
                    st.subheader("Historical Monthly Usage Trends")
                    plot_col1, plot_col2, plot_col3 = st.columns(3)

                    with plot_col1:
                        fig_dist = px.line(monthly_history, x='month_start', y='monthly_distance', markers=True,
                                           labels={'month_start': 'Month', 'monthly_distance': 'Distance (km)'})
                        fig_dist.update_layout(
                            title="Monthly Distance", margin=dict(t=30, b=0))  # Compact title
                        fig_dist.update_traces(line=dict(color='green'))

                        # Add prediction for next month
                        fig_dist.add_trace(go.Scatter(
                            x=[next_month_date],
                            y=[predictions['predicted_distance']],
                            mode='markers+text',
                            marker=dict(color='red', size=10, symbol='star'),
                            text=[
                                f"Pred: {predictions['predicted_distance']:.0f} km"],
                            textposition="top center",
                            name='Prediction'
                        ))

                        st.plotly_chart(fig_dist, use_container_width=True)

                    with plot_col2:
                        fig_elec = px.line(monthly_history, x='month_start', y='monthly_elec_kwh', markers=True,
                                           labels={'month_start': 'Month', 'monthly_elec_kwh': 'Electricity (kWh)'})
                        fig_elec.update_layout(
                            title="Monthly Electricity", margin=dict(t=30, b=0))
                        fig_elec.update_traces(line=dict(color='orange'))

                        # Add prediction for next month
                        fig_elec.add_trace(go.Scatter(
                            x=[next_month_date],
                            y=[predictions['predicted_electricity_kwh']],
                            mode='markers+text',
                            marker=dict(color='red', size=10, symbol='star'),
                            text=[
                                f"Pred: {predictions['predicted_electricity_kwh']:.1f} kWh"],
                            textposition="top center",
                            name='Prediction'
                        ))

                        st.plotly_chart(fig_elec, use_container_width=True)

                    with plot_col3:
                        fig_fuel = px.line(monthly_history, x='month_start', y='monthly_fuel_l', markers=True,
                                           labels={'month_start': 'Month', 'monthly_fuel_l': 'Fuel (L)'})
                        fig_fuel.update_layout(
                            title="Monthly Fuel", margin=dict(t=30, b=0))
                        fig_fuel.update_traces(line=dict(color='firebrick'))

                        # Add prediction for next month
                        fig_fuel.add_trace(go.Scatter(
                            x=[next_month_date],
                            y=[predictions['predicted_fuel_l']],
                            mode='markers+text',
                            marker=dict(color='red', size=10, symbol='star'),
                            text=[
                                f"Pred: {predictions['predicted_fuel_l']:.1f} L"],
                            textposition="top center",
                            name='Prediction'
                        ))

                        st.plotly_chart(fig_fuel, use_container_width=True)

                else:
                    logger.warning(
                        "Not enough historical monthly data to generate plots.")
                    st.info(
                        "Not enough historical monthly data to display trend plots.")

            except Exception as e:
                logger.exception("Error while generating historical plots.")
                st.warning(f"Could not generate historical plots: {e}")

        else:
            logger.warning(
                "Insufficient historical data to generate predictions.")
            st.warning(
                "Could not generate predictions. Insufficient historical data (requires at least 2 full months).")

    except Exception as e:
        logger.exception("Error while generating predictions section.")
        st.error(
            f"An error occurred while generating the predictions section: {e}")


def main():
    st.set_page_config(page_title="Hybrid Car Statistics",
                       page_icon="🚗", layout="wide")
    logger.info("Starting the application.")

    global AI_AVAILABLE
    global ask_ia_model

    AI_AVAILABLE, ask_ia_model = initialize_ai_model()

    try:
        file_path = "src/data/car_stats.csv"
        logger.info(f"Loading data from {file_path}.")
        raw_data = load_data(file_path)
        cleaned_data = clean_data(raw_data.copy())
        data = transform_data(cleaned_data)
        logger.info("Data loaded and processed successfully.")
    except Exception as e:
        logger.exception("Error during data loading or processing.")
        st.error(f"Error during data loading or processing: {e}")
        st.warning("Please upload the CSV file manually through the menu.")
        data = pd.DataFrame()

    selection, filtered_data, gas_price, elec_price, gas_comp, elec_comp = create_menu(
        data)

    if selection == "Overview":
        logger.info("Selected 'Overview' section.")
        show_overview(filtered_data, gas_price,
                      elec_price, gas_comp, elec_comp)
    elif selection == "Consumption Analysis":
        logger.info("Selected 'Consumption Analysis' section.")
        show_fuel_consumption(filtered_data)
    elif selection == "Trip Details":
        logger.info("Selected 'Trip Details' section.")
        show_trip_details(filtered_data)
    elif selection == "Performance Metrics":
        logger.info("Selected 'Performance Metrics' section.")
        show_performance(filtered_data)
    elif selection == "Predictions":
        logger.info("Selected 'Predictions' section.")
        show_predictions(data, gas_price, elec_price)
    elif selection == "Data Explorer":
        logger.info("Selected 'Data Explorer' section.")
        show_data_explorer(filtered_data)


if __name__ == "__main__":
    main()
