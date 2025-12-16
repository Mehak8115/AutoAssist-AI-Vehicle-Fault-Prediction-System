#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Vehicle Fault Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_css():
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .prediction-card {
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }
        .fault-detected {
            background: linear-gradient(135deg, #9E050D 0%, #C60610 100%);
            color: white;
        }
        .no-fault {
            background: linear-gradient(135deg, #016547 0%, #018D63 100%);
            color: white;
        }
        .input-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .section-header {
            color: #667eea;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .hero-section {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .hero-title {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .hero-subtitle {
            font-size: 20px;
            opacity: 0.9;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            margin-top: 50px;
            border-top: 2px solid #ddd;
        }
        .fault-type-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.08);
            margin: 15px 0;
            border-left: 5px solid #f5576c;
        }
        .fault-type-title {
            color: #f5576c;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .rca-section {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        .capa-section {
            background: #d1ecf1;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }
        .dataframe {
            color: #333 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Function to determine parameter status
def get_parameter_status(param_name, value):
    """Determine if parameter is normal, warning, or critical"""
    thresholds = {
        'engine_temp': {'warning': 100, 'critical': 120, 'unit': '°C'},
        'battery_voltage': {'low_critical': 11.8, 'low_warning': 12.2, 'high_warning': 13.0, 'high_critical': 13.5, 'unit': 'V'},
        'tire_pressure': {'low_critical': 26, 'low_warning': 28, 'high_warning': 36, 'high_critical': 40, 'unit': 'PSI'},
        'brake_wear': {'warning': 60, 'critical': 75, 'unit': '%'},
        'rpm': {'warning': 5000, 'critical': 6500, 'unit': ''},
        'vehicle_speed': {'warning': 120, 'critical': 160, 'unit': 'km/h'},
        'fuel_efficiency': {'low_warning': 8, 'low_critical': 5, 'unit': 'km/L'},
        'ambient_temp': {'low_critical': -10, 'low_warning': 0, 'high_warning': 40, 'high_critical': 50, 'unit': '°C'},
        'humidity': {'high_warning': 80, 'high_critical': 90, 'unit': '%'}
    }

    if param_name not in thresholds:
        return '✅ Normal'

    t = thresholds[param_name]

    # Check different threshold types
    if 'critical' in t:
        if value >= t['critical']:
            return '🔴 Critical'
        elif value >= t['warning']:
            return '⚠️ Warning'

    if 'low_critical' in t and 'high_critical' in t:
        if value <= t['low_critical'] or value >= t['high_critical']:
            return '🔴 Critical'
        elif value <= t['low_warning'] or value >= t['high_warning']:
            return '⚠️ Warning'
    elif 'low_critical' in t:
        if value <= t['low_critical']:
            return '🔴 Critical'
        elif value <= t['low_warning']:
            return '⚠️ Warning'
    elif 'high_critical' in t:
        if value >= t['high_critical']:
            return '🔴 Critical'
        elif value >= t['high_warning']:
            return '⚠️ Warning'

    return '✅ Normal'

# Function to analyze fault types
def analyze_fault_types(engine_temp, battery_voltage, tire_pressure, brake_wear, 
                        vehicle_speed, ambient_temp, weather, fuel_efficiency, humidity):
    """Analyze and return detected fault types with RCA and CAPA"""
    faults_detected = []

    # 1. Engine overheating
    if engine_temp > 100 or (engine_temp > 95 and ambient_temp > 35):
        faults_detected.append({
            'type': '🔥 Engine Overheating',
            'severity': 'Critical' if engine_temp > 110 else 'Warning',
            'rca': f'Excessive engine temperature ({engine_temp}°C) due to cooling system inefficiency or high ambient temperature ({ambient_temp}°C)',
            'capa': 'Inspect radiator, coolant levels, fan operation, and thermal sensors. Check for blockages in cooling system.'
        })

    # 2. Tire Pressure Loss
    if tire_pressure < 28 or tire_pressure > 38:
        faults_detected.append({
            'type': '🛞 Tire Pressure Abnormality',
            'severity': 'Critical' if tire_pressure < 26 or tire_pressure > 40 else 'Warning',
            'rca': f'Tire pressure ({tire_pressure} PSI) outside optimal range due to air leakage, temperature change, or puncture',
            'capa': 'Inspect tire integrity, check for punctures, and recalibrate TPMS (Tire Pressure Monitoring System).'
        })

    # 3. Battery Failure
    if battery_voltage < 12.2 or battery_voltage > 13.2:
        faults_detected.append({
            'type': '🔋 Battery Voltage Issue',
            'severity': 'Critical' if battery_voltage < 11.8 or battery_voltage > 13.5 else 'Warning',
            'rca': f'Battery voltage ({battery_voltage}V) abnormal due to aging, cold weather ({ambient_temp}°C), or charging system fault',
            'capa': 'Test battery health, check alternator output, inspect charging system, and replace battery if needed.'
        })

    # 4. Brake Wear
    if brake_wear > 60:
        faults_detected.append({
            'type': '🛑 Excessive Brake Wear',
            'severity': 'Critical' if brake_wear > 75 else 'Warning',
            'rca': f'Brake wear at {brake_wear}% indicates significant pad degradation requiring immediate attention',
            'capa': 'Replace brake pads immediately, inspect rotors for damage, and check brake fluid levels.'
        })

    # 5. Weather-Induced Risk
    if weather in ['Storm', 'Foggy', 'Rainy'] and vehicle_speed > 80:
        faults_detected.append({
            'type': '🌧️ Weather-Induced Risk',
            'severity': 'Warning',
            'rca': f'Adverse weather conditions ({weather}) combined with high vehicle speed ({vehicle_speed} km/h)',
            'capa': 'Reduce speed immediately, enable traction control, activate hazard lights, and issue driver alert.'
        })

    # 6. Poor Fuel Efficiency
    if fuel_efficiency < 8:
        faults_detected.append({
            'type': '⛽ Poor Fuel Efficiency',
            'severity': 'Warning',
            'rca': f'Low fuel efficiency ({fuel_efficiency} km/L) indicates engine performance issues or driving pattern problems',
            'capa': 'Check air filter, spark plugs, fuel injectors, and tire pressure. Analyze driving patterns.'
        })

    # 7. High Humidity Risk
    if humidity > 85 and weather == 'Rainy':
        faults_detected.append({
            'type': '💧 High Humidity Risk',
            'severity': 'Warning',
            'rca': f'Excessive humidity ({humidity}%) in rainy conditions may affect electrical systems',
            'capa': 'Monitor electrical components, check for water ingress, and ensure proper sealing.'
        })

    # If no specific faults but prediction says fault
    if len(faults_detected) == 0:
        faults_detected.append({
            'type': '⚙️ General System Fault',
            'severity': 'Warning',
            'rca': 'Multiple parameters showing marginal values requiring preventive inspection',
            'capa': 'Perform comprehensive vehicle diagnostics and preventive maintenance check.'
        })

    return faults_detected

# Load models function
@st.cache_resource
def load_models():
    try:
        # Check if files exist
        if not os.path.exists("vehicle_fault_random.pkl"):
            return None, None, "Model file 'vehicle_fault_random.pkl' not found"
        if not os.path.exists("weather_encoder_model.pkl"):
            return None, None, "Encoder file 'weather_encoder_model.pkl' not found"

        model = joblib.load("vehicle_fault_random.pkl")
        le = joblib.load("weather_encoder_model.pkl")
        return model, le, None
    except Exception as e:
        return None, None, str(e)

# Main app function
def main():
    # Load CSS
    load_css()

    # Load models
    model, le, error = load_models()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🚗 About This App")
        st.info("""
        This **Vehicle Fault Prediction System** uses machine learning to predict potential vehicle faults 
        based on real-time sensor data and environmental conditions.

        **Benefits:**
        - 🔍 Early fault detection
        - 💰 Reduced maintenance costs
        - 🛡️ Enhanced vehicle safety
        - ⏰ Preventive maintenance scheduling
        """)

        st.markdown("### 🤖 Model Information")
        if model is not None:
            st.success("""
            - **Algorithm:** Random Forest Classifier
            - **Input Features:** 10 parameters
            - **Output:** Binary classification
            - **Status:** ✅ Models loaded successfully
            """)
        else:
            st.error("""
            - **Status:** ❌ Models not loaded
            - Please check model files
            """)

        st.markdown("### 📊 Feature Categories")
        st.markdown("""
        - **Engine & Battery:** Temperature, Voltage
        - **Driving Metrics:** RPM, Speed, Efficiency
        - **Safety:** Tire Pressure, Brake Wear
        - **Environment:** Weather, Temperature, Humidity
        """)

    # Hero Section
    st.markdown("""
        <div class="hero-section">
            <div class="hero-title">🚗 AutoAssist AI: Vehicle Fault Prediction System</div>
            <div class="hero-subtitle">AI-Powered Predictive Maintenance for Your Vehicle</div>
        </div>
    """, unsafe_allow_html=True)

    # Check if models loaded
    if error:
        st.error(f"❌ Error loading models: {error}")
        st.info("""
        **Please ensure:**
        1. `vehicle_fault_random.pkl` exists in the same directory
        2. `weather_encoder_model.pkl` exists in the same directory
        3. Files are valid joblib/pickle files

        **Current directory:** """ + os.getcwd())
        return

    # Introduction
    st.markdown("""
    ### 🎯 How It Works
    This intelligent system analyzes multiple vehicle parameters including engine performance, battery health, 
    driving conditions, and environmental factors to predict potential faults before they occur. Early detection 
    helps prevent breakdowns, reduces repair costs, and ensures optimal vehicle performance.
    """)

    st.markdown("---")

    # Input Form
    with st.form("prediction_form"):
        st.markdown("### 📝 Enter Vehicle Parameters")

        # Engine & Battery Section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔧 Engine & Battery Metrics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            engine_temp = st.number_input(
                "🌡️ Engine Temperature (°C)",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=1.0,
                help="Normal range: 80-100°C"
            )
            battery_voltage = st.number_input(
                "🔋 Battery Voltage (V)",
                min_value=0.0,
                max_value=20.0,
                value=12.6,
                step=0.1,
                help="Normal range: 12.4-12.8V"
            )
        with col2:
            tire_pressure = st.number_input(
                "⚙️ Tire Pressure (PSI)",
                min_value=0.0,
                max_value=60.0,
                value=32.0,
                step=0.5,
                help="Recommended: 30-35 PSI"
            )
            brake_wear = st.number_input(
                "🛑 Brake Wear (%)",
                min_value=0.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Replace at >70%"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Driving Conditions Section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏎️ Driving Conditions</div>', unsafe_allow_html=True)
        col3, col4, col5 = st.columns(3)
        with col3:
            rpm = st.number_input(
                "⚡ RPM",
                min_value=0.0,
                max_value=8000.0,
                value=2500.0,
                step=100.0,
                help="Engine revolutions per minute"
            )
        with col4:
            vehicle_speed = st.number_input(
                "🚗 Vehicle Speed (km/h)",
                min_value=0.0,
                max_value=250.0,
                value=60.0,
                step=5.0,
                help="Current driving speed"
            )
        with col5:
            fuel_efficiency = st.number_input(
                "⛽ Fuel Efficiency (km/L)",
                min_value=0.0,
                max_value=50.0,
                value=15.0,
                step=0.5,
                help="Fuel consumption rate"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Environment & Weather Section
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🌤️ Environment & Weather</div>', unsafe_allow_html=True)
        col6, col7, col8 = st.columns(3)
        with col6:
            ambient_temp = st.number_input(
                "🌡️ Ambient Temperature (°C)",
                min_value=-40.0,
                max_value=60.0,
                value=25.0,
                step=1.0,
                help="External temperature"
            )
        with col7:
            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
                help="Relative humidity"
            )
        with col8:
            weather_condition = st.selectbox(
                "☁️ Weather Condition",
                options=["Clear", "Rainy", "Foggy", "Hot", "Cold", "Storm"],
                help="Current weather conditions"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🔮 Predict Vehicle Fault")

    # Prediction Logic
    if submit_button:
        if model is not None and le is not None:
            try:
                # Encode weather condition
                weather_encoded = le.transform([weather_condition])[0]

                # Prepare input features in correct order
                features = np.array([[
                    engine_temp,
                    battery_voltage,
                    tire_pressure,
                    rpm,
                    brake_wear,
                    vehicle_speed,
                    fuel_efficiency,
                    ambient_temp,
                    humidity,
                    weather_encoded
                ]])

                # Make prediction
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]

                # Get confidence (probability of predicted class)
                confidence = prediction_proba[prediction] * 100

                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")

                if prediction == 1:  # Fault detected
                    st.markdown(f"""
                        <div class="prediction-card fault-detected">
                            <h1>⚠️ FAULT DETECTED</h1>
                            <h2>Immediate attention required!</h2>
                            <p style="font-size: 20px; margin-top: 20px;">
                                The system has detected a potential fault in your vehicle.
                                Please schedule a maintenance check as soon as possible.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_b:
                        st.markdown("#### 🎯 Prediction Confidence")
                        st.progress(confidence / 100)
                        st.markdown(f"""
                            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #f5576c; margin-top: 10px;">
                                {confidence:.2f}%
                            </div>
                        """, unsafe_allow_html=True)

                    # Analyze fault types
                    st.markdown("---")
                    st.markdown("### 🔍 Detected Fault Types")

                    fault_types = analyze_fault_types(
                        engine_temp, battery_voltage, tire_pressure, brake_wear,
                        vehicle_speed, ambient_temp, weather_condition, fuel_efficiency, humidity
                    )

                    if len(fault_types) == 0 or (len(fault_types) == 1 and 'General System Fault' in fault_types[0]['type']):
                        st.info("ℹ️ No specific fault pattern identified. The model detected general anomalies requiring inspection.")
                        st.markdown("""
                            <div class="fault-type-card">
                                <div class="fault-type-title">⚙️ General System Anomaly</div>
                                <p><strong>Severity:</strong> <span style="color: #ffc107; font-weight: bold;">Warning</span></p>
                                <div class="rca-section">
                                    <strong>🔬 Root Cause Analysis (RCA):</strong><br/>
                                    Multiple parameters showing marginal values or unusual patterns requiring comprehensive diagnostics
                                </div>
                                <div class="capa-section">
                                    <strong>🔧 Corrective & Preventive Actions (CAPA):</strong><br/>
                                    Perform comprehensive vehicle diagnostics and preventive maintenance check
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        for i, fault in enumerate(fault_types, 1):
                            severity_color = "#dc3545" if fault['severity'] == 'Critical' else "#ffc107"
                            st.markdown(f"""
                                <div class="fault-type-card" style="border-left-color: {severity_color};">
                                    <div class="fault-type-title">{i}. {fault['type']}</div>
                                    <p><strong>Severity:</strong> <span style="color: {severity_color}; font-weight: bold;">{fault['severity']}</span></p>
                                    <div class="rca-section">
                                        <strong>🔬 Root Cause Analysis (RCA):</strong><br/>
                                        {fault['rca']}
                                    </div>
                                    <div class="capa-section">
                                        <strong>🔧 Corrective & Preventive Actions (CAPA):</strong><br/>
                                        {fault['capa']}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    st.warning("""
                    **⚠️ Immediate Actions Required:**
                    - Contact a certified mechanic immediately
                    - Avoid long-distance travel until inspection
                    - Monitor all dashboard warning lights
                    - Keep vehicle at reduced speeds if driving is necessary
                    """)

                else:  # No fault
                    st.markdown(f"""
                        <div class="prediction-card no-fault">
                            <h1>✅ NO FAULT DETECTED</h1>
                            <h2>Your vehicle is in good condition!</h2>
                            <p style="font-size: 20px; margin-top: 20px;">
                                All parameters are within normal range.
                                Continue with regular maintenance schedule.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_b:
                        st.markdown("#### 🎯 Prediction Confidence")
                        st.progress(confidence / 100)
                        st.markdown(f"""
                            <div style="text-align: center; font-size: 24px; font-weight: bold; color: #00f2fe; margin-top: 10px;">
                                {confidence:.2f}%
                            </div>
                        """, unsafe_allow_html=True)

                    st.success("""
                    **✅ Maintenance Tips:**
                    - Continue regular oil changes every 5,000-7,500 km
                    - Check tire pressure monthly
                    - Inspect brake pads every 10,000 km
                    - Monitor battery health quarterly
                    - Keep your vehicle clean and well-maintained
                    """)

                # Feature Analysis with dynamic status
                st.markdown("---")
                st.markdown("### 📈 Parameter Analysis")

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("#### 🔧 Mechanical Parameters")
                    param_data = pd.DataFrame({
                        'Parameter': ['Engine Temp', 'Battery Voltage', 'Tire Pressure', 'Brake Wear'],
                        'Value': [f"{engine_temp}°C", f"{battery_voltage}V", f"{tire_pressure} PSI", f"{brake_wear}%"],
                        'Status': [
                            get_parameter_status('engine_temp', engine_temp),
                            get_parameter_status('battery_voltage', battery_voltage),
                            get_parameter_status('tire_pressure', tire_pressure),
                            get_parameter_status('brake_wear', brake_wear)
                        ]
                    })
                    # Style the dataframe
                    st.dataframe(
                        param_data, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                            "Value": st.column_config.TextColumn("Value", width="medium"),
                            "Status": st.column_config.TextColumn("Status", width="medium"),
                        }
                    )

                with col_right:
                    st.markdown("#### 🌤️ Environmental & Performance")
                    env_data = pd.DataFrame({
                        'Parameter': ['Ambient Temp', 'Humidity', 'Weather', 'Fuel Efficiency'],
                        'Value': [f"{ambient_temp}°C", f"{humidity}%", weather_condition, f"{fuel_efficiency} km/L"],
                        'Status': [
                            get_parameter_status('ambient_temp', ambient_temp),
                            get_parameter_status('humidity', humidity),
                            '✅ Normal' if weather_condition in ['Clear', 'Hot'] else '⚠️ Caution',
                            get_parameter_status('fuel_efficiency', fuel_efficiency)
                        ]
                    })
                    # Style the dataframe
                    st.dataframe(
                        env_data, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                            "Value": st.column_config.TextColumn("Value", width="medium"),
                            "Status": st.column_config.TextColumn("Status", width="medium"),
                        }
                    )

            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("Please ensure all input values are valid and the model is properly trained.")
        else:
            st.error("❌ Models not loaded. Cannot make prediction.")

    # Footer
    st.markdown("""
        <div class="footer">
            <p style="font-size: 16px; font-weight: bold;">🚗 Vehicle Fault Prediction System</p>
            <p>Powered by AutoAssist AI | Developed with ❤️ using Streamlit</p>
            <p style="font-size: 12px; color: #999;">© 2025 Predictive Maintenance Solutions</p>
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()


# In[ ]:




