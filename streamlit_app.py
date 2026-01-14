import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load model and scaler
with open('car_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data for statistics
df = pd.read_csv('carprices (1) (1).csv')

st.set_page_config(page_title="Car Price Regression", layout="wide")

st.title("🚗 Car Price Prediction Model")
st.markdown("---")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.info("""
This app predicts car selling prices based on:
- **Mileage** (in miles)
- **Age** (in years)

The model uses MinMax Scaling for data normalization.
""")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Prediction")
    st.subheader("Enter car details:")

    mileage = st.number_input(
        "Mileage (miles)", min_value=0, value=50000, step=1000)
    age = st.number_input("Age (years)", min_value=0, value=5, step=1)

    if st.button("Predict Price", use_container_width=True):
        # Scale the input
        input_data = np.array([[mileage, age]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        predicted_price = model.predict(input_scaled)[0]

        st.success(f"### Predicted Price: ${predicted_price:,.2f}")

        # Show input details
        with st.expander("Input Details"):
            st.write(f"**Mileage:** {mileage:,} miles")
            st.write(f"**Age:** {age} years")
            st.write(f"**Scaled Mileage:** {input_scaled[0][0]:.4f}")
            st.write(f"**Scaled Age:** {input_scaled[0][1]:.4f}")

with col2:
    st.header("📈 Model Statistics")

    # Calculate model metrics on the data
    X = df[['Mileage', 'Age(yrs)']].values
    y = df['Sell Price($)'].values
    X_scaled = scaler.transform(X)

    y_pred_all = model.predict(X_scaled)
    r2 = r2_score(y, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y, y_pred_all))

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("R² Score", f"{r2:.4f}")
    with metric_col2:
        st.metric("RMSE", f"${rmse:,.2f}")

    st.subheader("Model Coefficients:")
    st.write(f"- **Mileage Coefficient:** {model.coef_[0]:.4f}")
    st.write(f"- **Age Coefficient:** {model.coef_[1]:.4f}")
    st.write(f"- **Intercept:** ${model.intercept_:,.2f}")

st.markdown("---")

# Data Analysis Section
st.header("📉 Data Analysis")

tab1, tab2, tab3 = st.tabs(
    ["Dataset Overview", "Price vs Mileage", "Price vs Age"])

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Avg Price", f"${df['Sell Price($)'].mean():,.2f}")
    with col3:
        st.metric(
            "Price Range", f"${df['Sell Price($)'].min():,} - ${df['Sell Price($)'].max():,}")

with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Mileage'], df['Sell Price($)'],
               alpha=0.6, s=100, label='Actual')

    # Plot regression line
    mileage_range = np.linspace(df['Mileage'].min(), df['Mileage'].max(), 100)
    age_mean = df['Age(yrs)'].mean()
    pred_data = np.array([[m, age_mean] for m in mileage_range])
    pred_data_scaled = scaler.transform(pred_data)
    pred_prices = model.predict(pred_data_scaled)

    ax.plot(mileage_range, pred_prices, 'r-',
            linewidth=2, label='Regression Line')
    ax.set_xlabel('Mileage (miles)', fontsize=12)
    ax.set_ylabel('Sell Price ($)', fontsize=12)
    ax.set_title('Car Price vs Mileage', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['Age(yrs)'], df['Sell Price($)'],
               alpha=0.6, s=100, label='Actual')

    # Plot regression line
    age_range = np.linspace(df['Age(yrs)'].min(), df['Age(yrs)'].max(), 100)
    mileage_mean = df['Mileage'].mean()
    pred_data = np.array([[mileage_mean, a] for a in age_range])
    pred_data_scaled = scaler.transform(pred_data)
    pred_prices = model.predict(pred_data_scaled)

    ax.plot(age_range, pred_prices, 'r-', linewidth=2, label='Regression Line')
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Sell Price ($)', fontsize=12)
    ax.set_title('Car Price vs Age', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.caption("Built with Streamlit • Machine Learning Model with MinMax Scaler")
