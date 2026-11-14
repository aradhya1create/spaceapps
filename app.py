%%writefile app.py
import streamlit as st
import lightkurve as lk
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="NASA Exoplanet Transit Detection Dashboard",
    page_icon="🪐",
    layout="wide"
)

st.title("🪐 NASA Exoplanet Transit Detection Dashboard")
st.write("Search NASA's archive for target stars, run Box Least Squares (BLS) period detection, and explore interactive transit light curves.")

# 2. Sidebar Controls
st.sidebar.header("Observation Settings")
star_name = st.sidebar.text_input("Enter Star Name", value="Kepler-10")
mission_name = st.sidebar.selectbox("Select Mission", ["Kepler", "K2", "TESS"])
min_period = st.sidebar.number_input("Min Period (Days)", value=0.5)
max_period = st.sidebar.number_input("Max Period (Days)", value=10.0)

if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Fetching data for {star_name} from NASA archive..."):
        try:
            # Data Retrieval & Cleaning
            search_result = lk.search_lightcurve(star_name, mission=mission_name)
            if len(search_result) == 0:
                st.error(f"No light curves found for {star_name} under mission {mission_name}.")
            else:
                lc = search_result[0].download()
                clean_lc = lc.remove_nans().flatten(window_length=101)
                st.success(f"Successfully loaded and cleaned light curve for {star_name}!")

                # BLS Period Detection
                periods = np.linspace(min_period, max_period, 5000)
                periodogram = clean_lc.to_periodogram(method='bls', period=periods)
                best_period = periodogram.period_at_max_power
                best_t0 = periodogram.transit_time_at_max_power

                # Display Key Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Detected Orbital Period", value=f"{best_period.value:.4f} days")
                with col2:
                    st.metric(label="Max BLS Power", value=f"{periodogram.max_power:.2f}")

                # Interactive Light Curve Plot
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=clean_lc.time.value,
                    y=clean_lc.flux.value,
                    mode='markers',
                    marker=dict(size=3, color='#4F46E5', opacity=0.6),
                    name='Normalized Flux'
                ))
                fig_lc.update_layout(
                    title=f"Flattened Light Curve for {star_name}",
                    xaxis_title="Time (Days)",
                    yaxis_title="Normalized Flux",
                    template="plotly_dark",
                    height=400
                )

                # Phase-Folded Transit Plot
                folded_lc = clean_lc.fold(period=best_period, epoch_time=best_t0)
                binned_lc = folded_lc.bin(time_bin_size=0.002)

                fig_folded = go.Figure()
                fig_folded.add_trace(go.Scatter(
                    x=folded_lc.phase.value,
                    y=folded_lc.flux.value,
                    mode='markers',
                    marker=dict(size=2, color='gray', opacity=0.3),
                    name='Raw Phase Data'
                ))
                fig_folded.add_trace(go.Scatter(
                    x=binned_lc.phase.value,
                    y=binned_lc.flux.value,
                    mode='lines+markers',
                    line=dict(color='#EF4444', width=2),
                    name='Binned Transit'
                ))
                fig_folded.update_layout(
                    title=f"Phase-Folded Transit Dip (Period = {best_period.value:.4f} days)",
                    xaxis_title="Phase",
                    yaxis_title="Normalized Flux",
                    template="plotly_dark",
                    height=450
                )

                # Display Visualizations
                st.plotly_chart(fig_lc, use_container_width=True)
                st.plotly_chart(fig_folded, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
