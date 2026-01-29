import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly
from datetime import date, timedelta

from data_loader import generate_data
from model_engine import train_model, get_accuracy_score


st.set_page_config(page_title="Hospital AI Manager", page_icon="🏥", layout="wide")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto:wght@300;400;700&display=swap');
    
    /* Force Fonts */
    h1, h2, h3, [data-testid="stMetricLabel"] {
        font-family: 'Orbitron', sans-serif !important;
        color: #2c3e50 !important;
    }
    
    p, div, button, [data-testid="stMetricValue"] {
        font-family: 'Roboto', sans-serif !important;
    }

    .main { background-color: #f0f2f6; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_wrapper():
    return generate_data()

@st.cache_resource
def train_model_wrapper(df):
    return train_model(df)

with st.spinner('Initializing AI Model...'):
    df = load_data_wrapper()
    model = train_model_wrapper(df)
    mae_score = get_accuracy_score(df)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.header("Admin Controls")
    
    tomorrow = date.today() + timedelta(days=1)
    selected_date = st.date_input("Select Forecast Date", value=tomorrow)
    patients_per_doc = st.slider("Patients per Doctor", 10, 30, 15)
    
    st.markdown("---")
    
    
    st.subheader("📍 Hospital Location")
    map_data = pd.DataFrame({'lat': [9.9252], 'lon': [78.1198]})
    st.map(map_data, zoom=14)
    st.caption("Powered by Google Maps Data")


st.title("🏥 Hospital Command Center")
st.markdown("### ⚡ Real-time Patient Forecasting Engine")


future = pd.DataFrame({'ds': [pd.to_datetime(selected_date)]})
forecast = model.predict(future)
pred = int(forecast['yhat'].values[0])
low = int(forecast['yhat_lower'].values[0])
high = int(forecast['yhat_upper'].values[0])
doctors_needed = (pred // patients_per_doc) + 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Forecast Date", str(selected_date))
col2.metric("👥 Expected Patients", f"{pred}", delta=f"{pred - 60} vs Avg")
col3.metric("📉 Confidence Interval", f"{low} - {high}")
col4.metric("✅ AI Accuracy (MAE)", f"{mae_score:.1f}", help="Mean Absolute Error")

st.markdown("---")


tab1, tab2 = st.tabs(["📈 Interactive Forecast", "📊 Weekly Patterns"])

with tab1:
    st.subheader(f"Projected Traffic for {selected_date}")
    future_range = model.make_future_dataframe(periods=60)
    full_forecast = model.predict(future_range)
    fig = plot_plotly(model, full_forecast)
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Patient Volume")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Weekly Resource Allocation")
    st.bar_chart({"Monday": 85, "Tuesday": 80, "Wednesday": 75, "Thursday": 70, "Friday": 65, "Saturday": 40, "Sunday": 35})

st.success(f"**AI Recommendation:** Allocate resources for **{doctors_needed} doctors** on {selected_date}.")