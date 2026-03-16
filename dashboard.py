
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import load_model
import joblib

scaler_y = joblib.load("scaler_y.pkl")

lstm_model = load_model("water_lstm_model.h5", compile=False)

st.set_page_config(page_title="Smart Water Distribution AI", layout="wide")

# ---------------- TITLE ----------------

st.markdown("""
<style>
.main-title {
font-size:40px;
font-weight:bold;
color:#1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Smart Water Distribution AI Dashboard</p>', unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("water_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Sensor Inputs")

city = st.sidebar.selectbox(
"City",
["Delhi","Mumbai","Bangalore","Chennai","Hyderabad"]
)

pressure = st.sidebar.slider("Pressure (psi)",40,100,60)
flow = st.sidebar.slider("Flow Rate (gpm)",700,1000,850)
activity = st.sidebar.selectbox("Activity Level",[0,1,2])

# ---------------- CITY WEATHER ----------------

city_weather = {
"Delhi":{"temp":32,"humidity":45,"rain":0,"wind":8},
"Mumbai":{"temp":29,"humidity":80,"rain":6,"wind":12},
"Bangalore":{"temp":26,"humidity":65,"rain":3,"wind":9},
"Chennai":{"temp":31,"humidity":70,"rain":4,"wind":10},
"Hyderabad":{"temp":33,"humidity":50,"rain":0,"wind":7}
}

temperature = city_weather[city]["temp"]
humidity = city_weather[city]["humidity"]
precipitation = city_weather[city]["rain"]
wind_speed = city_weather[city]["wind"]

# ---------------- CONSUMPTION PARAMETERS ----------------

st.sidebar.header("Consumption Parameters")

household_size = st.sidebar.slider("Household Size",1,10,4)
households = st.sidebar.slider("Number of Households",100,100000,1000)

# ---------------- PIPELINE PARAMETERS ----------------

pipe_age = st.sidebar.slider("Pipeline Age (years)",1,50,20)
head_loss = st.sidebar.slider("Head Loss",0,100,30)

# ---------------- WATER QUALITY ----------------

ph = st.sidebar.slider("pH Level",0.0,14.0,7.0)
turbidity = st.sidebar.slider("Turbidity",0.0,10.0,1.0)
chlorine = st.sidebar.slider("Chlorine",0.0,5.0,1.0)

# ---------------- MODEL INPUT ----------------

input_data = pd.DataFrame({
"pressure_psi":[pressure],
"flow_rate_gpm":[flow],
"temperature":[temperature],
"humidity":[humidity],
"precipitation":[precipitation],
"wind_speed":[wind_speed],
"activity_level":[activity]
})

ml_prediction = model.predict(input_data)[0]

# LSTM prediction

# LSTM prediction

input_data = np.array([[
    pressure,
    flow,
    temperature,
    humidity,
    activity
]])

input_data = input_data.reshape((1,1,5))

# predict
lstm_prediction = lstm_model.predict(input_data)

# convert back to original scale
predicted_demand = scaler_y.inverse_transform(lstm_prediction)[0][0]

# prevent negative demand
predicted_demand = max(predicted_demand,0) 

# ---------------- DEMAND CALCULATIONS ----------------

per_person = predicted_demand
household_demand = per_person * household_size
city_demand = household_demand * households

north_demand = city_demand * 0.35
central_demand = city_demand * 0.40
south_demand = city_demand * 0.25

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
"System Overview",
"Demand Analytics",
"Weather Impact",
"Infrastructure",
"AI Insights",
"City Monitoring"
])

# ==========================================================
# SYSTEM OVERVIEW
# ==========================================================

with tab1:

    st.subheader("System Metrics")

    col1,col2,col3 = st.columns(3)

    col1.metric("Pressure",pressure)
    col2.metric("Flow Rate",flow)
    col3.metric("Temperature",temperature)

    st.subheader("SCADA Live Monitoring")

    sensor_data = pd.DataFrame({
    "time":range(50),
    "pressure":np.random.normal(60,3,50),
    "flow":np.random.normal(850,20,50)
    })

    fig = px.line(sensor_data,x="time",y=["pressure","flow"])
    st.plotly_chart(fig,use_container_width=True)

    # Demand Trend

    st.subheader("Demand Trend")

    trend = np.random.normal(float(per_person),0.2,24)

    trend_df = pd.DataFrame({
    "Hour":range(24),
    "Demand":trend
    })

    fig = px.line(trend_df,x="Hour",y="Demand")
    st.plotly_chart(fig,use_container_width=True)

    # ===============================
    # CRITICAL ALERT CONDITION
    # ===============================

    critical_alert = False

    if pressure < 40 and flow > 900:
        critical_alert = True


    # ===============================
    # RED ALERT SYSTEM
    # ===============================

    if critical_alert:

        st.error("🚨 CRITICAL PIPELINE FAILURE DETECTED")

        st.markdown(
            """
            <style>
            .big-alert {
                font-size:40px;
                color:red;
                text-align:center;
                font-weight:bold;
                animation: blink 1s infinite;
            }

            @keyframes blink {
                0% {opacity:1;}
                50% {opacity:0;}
                100% {opacity:1;}
            }
            </style>

            <div class="big-alert">
            🚨 EMERGENCY WATER LEAK ALERT 🚨
            </div>
            """,
            unsafe_allow_html=True
        )

        # Alarm Sound
        st.markdown(
            """
            <audio autoplay>
            <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

        # ================================
        # AI LEAK PROBABILITY MODEL
        # ================================

    st.subheader("AI Leak Detection")

    # calculate leak probability score

    leak_score = (abs(pressure - 60) * 1.2) + (abs(flow - 850) * 0.5)

    leak_score = min(leak_score,100)

    st.metric("Leak Probability", f"{leak_score:.1f}%")

    # ================================
    # SMART VALVE CONTROL
    # ================================

    if leak_score > 70:

        valve_status = "CLOSED"
        st.error("🚨 Leak detected – pipeline isolated")

    elif leak_score > 40:

        valve_status = "PARTIAL"
        st.warning("⚠ Abnormal pipeline behaviour")

    else:

        valve_status = "OPEN"
        st.success("Pipeline operating normally")

    st.metric("Valve Status", valve_status)

    # ================================
    # SMART VALVE CONTROL
    # ================================

    if leak_score > 70:

        valve_status = "CLOSED"
        st.error("🚨 Leak detected – pipeline isolated")

    elif leak_score > 40:

        valve_status = "PARTIAL"
        st.warning("⚠ Abnormal pipeline behaviour")

    else:

        valve_status = "OPEN"
        st.success("Pipeline operating normally")

    st.metric("Valve Status", valve_status)

# ==========================================================
# DEMAND ANALYTICS
# ==========================================================

with tab2:

    st.subheader("AI Demand Prediction")

    st.metric("Per Person Demand",f"{per_person:.2f} L")

    st.subheader("Consumption Analysis")

    col1,col2,col3 = st.columns(3)

    col1.metric("Per Person",f"{per_person:.2f}")
    col2.metric("Household",f"{household_demand:.2f}")
    col3.metric("City Demand",f"{city_demand:.0f}")

    st.subheader("Zone Distribution")

    zone_df = pd.DataFrame({
    "Zone":["North","Central","South"],
    "Demand":[north_demand,central_demand,south_demand]
    })

    fig = px.bar(zone_df,x="Zone",y="Demand",color="Zone")
    st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# WEATHER IMPACT
# ==========================================================

with tab3:

    st.subheader("Weather Monitoring")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Temperature",temperature)
    col2.metric("Humidity",humidity)
    col3.metric("Rainfall",precipitation)
    col4.metric("Wind Speed",wind_speed)

    st.subheader("Weather Influence on Water System")

    weather_df = pd.DataFrame({
    "Feature":["Temperature","Humidity","Rainfall","Wind"],
    "Value":[temperature,humidity,precipitation,wind_speed]
    })

    fig = px.bar(weather_df,x="Feature",y="Value",color="Feature")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Weather Impact Analysis")

    if temperature > 30:
        st.warning("High temperature may increase water demand")

    elif precipitation > 10:
        st.info("Rainfall may reduce outdoor consumption")

    elif humidity > 80:
        st.info("High humidity detected")

    else:
        st.success("Weather conditions normal")

# ==========================================================
# INFRASTRUCTURE
# ==========================================================

with tab4:

    st.subheader("Infrastructure Health Score")

    health_score = 100

    if pipe_age > 30:
        health_score -= 20

    if head_loss > 50:
        health_score -= 20

    if turbidity > 5:
        health_score -= 20

    if chlorine < 0.5:
        health_score -= 10

    st.metric("System Health Score", f"{health_score}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Infrastructure Health"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "green"},
            'steps':[
                {'range':[0,40],'color':"red"},
                {'range':[40,70],'color':"yellow"},
                {'range':[70,100],'color':"lightgreen"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # ---------------- Pipeline Monitoring ----------------

    st.subheader("Pipeline Health Monitoring")

    if pipe_age > 35:
        st.error("⚠ Aging pipeline detected")

    elif head_loss > 60:
        st.warning("⚠ Flow resistance detected")

    else:
        st.success("✅ Pipeline healthy")

    # Predictive Maintenance

    st.subheader("Maintenance Prediction")

    if pipe_age > 35:
        st.error("High maintenance risk")

    elif pipe_age > 25:
        st.warning("Pipeline aging – inspection recommended")

    else:
        st.success("No maintenance required")

    st.divider()

    # ---------------- Water Quality ----------------

    st.subheader("Water Quality Monitoring")

    if turbidity > 5:
        st.error("⚠ High turbidity detected")

    elif chlorine < 0.5:
        st.warning("⚠ Low chlorine level")

    elif ph < 6.5 or ph > 8.5:
        st.warning("⚠ pH imbalance")

    else:
        st.success("✅ Water quality safe")

    # Water Quality Chart

    st.subheader("Water Quality Indicators")

    quality_df = pd.DataFrame({
        "Parameter":["pH","Turbidity","Chlorine"],
        "Value":[ph,turbidity,chlorine]
    })

    fig = px.bar(
        quality_df,
        x="Parameter",
        y="Value",
        color="Parameter",
        title="Water Quality Metrics"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # ---------------- Infrastructure Alerts ----------------

    st.subheader("Infrastructure Alerts")

    alerts = []

    if pipe_age > 35:
        alerts.append("Pipeline aging detected")

    if head_loss > 60:
        alerts.append("High flow resistance")

    if turbidity > 5:
        alerts.append("Water turbidity high")

    if chlorine < 0.5:
        alerts.append("Low chlorine level")

    if len(alerts) == 0:
        st.success("No active alerts")

    else:
        for alert in alerts:
            st.error(alert)

with tab4:

    st.subheader("Infrastructure Health Score")
    ...
    # Infrastructure alerts code
    ...

    # ---------------- PIPE FAILURE RISK ----------------

    st.subheader("Pipe Failure Risk Prediction")

    risk_score = (pipe_age * 1.2) + (head_loss * 0.8)
    risk_score = min(risk_score, 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Pipe Failure Risk"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "red"},
            'steps':[
                {'range':[0,40],'color':"lightgreen"},
                {'range':[40,70],'color':"yellow"},
                {'range':[70,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)

    if risk_score > 70:
        st.error("High risk of pipeline failure")

    elif risk_score > 40:
        st.warning("Moderate pipeline risk")

    else:
        st.success("Pipeline failure risk low")

    # ---------------- ZONE PRESSURE ----------------

    st.subheader("Zone Pressure Heatmap")

    zone_data = pd.DataFrame({
        "Zone":["North","Central","South"],
        "Pressure":[
            np.random.normal(pressure,3),
            np.random.normal(pressure,3),
            np.random.normal(pressure,3)
        ]
    })

    fig = px.bar(
        zone_data,
        x="Zone",
        y="Pressure",
        color="Pressure",
        title="Zone Pressure Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# AI INSIGHTS
# ==========================================================

with tab5:

    # ---------------- FEATURE IMPORTANCE ----------------

    st.subheader("AI Feature Importance")

    features = ["Activity","Flow","Pressure","Temperature"]
    importance = [0.97,0.01,0.01,0.002]

    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        title="Feature Contribution to Demand"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    

    # ---------------- AI PREDICTION EXPLANATION ----------------

    st.subheader("AI Prediction Explanation")

    explanation = f"""
    The AI model predicted **{per_person:.2f} liters per person** based on:

    • Temperature: {temperature} °C
    • Flow Rate: {flow} gpm
    • Pressure: {pressure} psi
    • Activity Level: {activity}
    """

    st.markdown(explanation)

    st.divider()

    # ---------------- AI SYSTEM SCORE ----------------

    st.subheader("AI System Intelligence Score")

    ai_score = 100

    if per_person > 3.5:
        ai_score -= 20

    if pipe_age > 30:
        ai_score -= 15

    if turbidity > 5:
        ai_score -= 20

    if head_loss > 60:
        ai_score -= 15

    st.metric("AI System Score", f"{ai_score}%")

    st.divider()

        # ---------------- PIPELINE LEAK DETECTION ----------------

    st.subheader("Pipeline Leak Detection")

    col1, col2 = st.columns(2)

    col1.metric("Pressure", f"{pressure} psi")
    col2.metric("Flow Rate", f"{flow} gpm")

    if pressure < 45 and flow > 900:
        st.error("🚨 Possible pipeline leak detected")

    elif pressure < 50:
        st.warning("⚠ Pressure drop detected in pipeline")

    elif flow > 950:
        st.warning("⚠ Unusual high flow detected")

    else:
        st.success("✅ Pipeline operating normally")

    # ---------------- LEAK PROBABILITY MODEL ----------------

    st.subheader("AI Leak Probability")
    
    pressure_diff = abs(pressure - 60)
    flow_diff = abs(flow - 850)
    
    leak_probability = (pressure_diff * 1.5) + (flow_diff * 0.05)
    
    leak_probability = min(leak_probability,100)
    
    st.metric("Leak Probability", f"{leak_probability:.1f}%")

    # ---------------- CRITICAL ALERT ----------------

    if leak_probability > 80:

        st.error("🚨 CRITICAL PIPELINE LEAK DETECTED")

        st.markdown(
        """
        <audio autoplay>
        <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
        </audio>
        """,
        unsafe_allow_html=True
        )

    # ---------------- AI DEMAND FORECAST ----------------

    st.subheader("AI Demand Forecast (Next 24 Hours)")

    # Training data for regression model
    X_train = np.arange(24).reshape(-1,1)
    y_train = np.random.normal(per_person,0.2,24)

    # Train Linear Regression model
    forecast_model = LinearRegression()
    forecast_model.fit(X_train,y_train)

    # Predict demand for next 24 hours
    hours = np.arange(24).reshape(-1,1)
    forecast = forecast_model.predict(hours)

    # Create dataframe
    forecast_df = pd.DataFrame({
        "Hour": range(24),
        "Demand": forecast
    })

    fig = px.line(
        forecast_df,
        x="Hour",
        y="Demand",
        title="AI Forecasted Water Demand"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    #...............

    st.subheader("AI Pump Optimization")

    if per_person > 3.5:
        st.warning("AI recommends increasing pump output")

    elif per_person < 2:
        st.info("AI recommends reducing pump pressure")

    else:
        st.success("Pump operation optimal")

    # ---------------- DYNAMIC AI INSIGHTS ----------------

    st.subheader("Dynamic AI Insights")

    insights = []

    if per_person > 3.5:
        insights.append("High water demand predicted by AI")

    if temperature > 30:
        insights.append("High temperature may increase consumption")

    if humidity > 80:
        insights.append("High humidity detected")

    if pipe_age > 30:
        insights.append("Aging infrastructure may impact efficiency")

    if len(insights) == 0:
        st.success("System operating in optimal conditions")

    else:
        for insight in insights:
            st.info(insight)

    # ==========================================================
    # AI CHAT ASSISTANT
    # ==========================================================

    st.subheader("AI Water System Assistant")

    user_input = st.chat_input("Ask about system status...")

    if user_input:

        if "leak" in user_input.lower():
            st.write("AI: Leak monitoring active across all pipeline zones.")

        elif "demand" in user_input.lower():
            st.write(f"AI: Current predicted demand is {per_person:.2f} liters per person.")

        elif "pressure" in user_input.lower():
            st.write(f"AI: Current system pressure is {pressure} psi.")

        elif "weather" in user_input.lower():
            st.write(f"AI: Temperature {temperature}°C with humidity {humidity}%.")

        else:
            st.write("AI: System operating normally. No critical alerts.")
# ==========================================================
# CITY MONITORING
# ==========================================================

with tab6:

    st.subheader("City Monitoring Map")

    city_location = {
    "Delhi":[28.6139,77.2090],
    "Mumbai":[19.0760,72.8777],
    "Bangalore":[12.9716,77.5946],
    "Chennai":[13.0827,80.2707],
    "Hyderabad":[17.3850,78.4867]
    }

    map_data = pd.DataFrame({
    "lat":[city_location[city][0]],
    "lon":[city_location[city][1]]
    })

    st.map(map_data)

    st.caption(f"Monitoring City: {city}")

