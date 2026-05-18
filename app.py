import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)

# =====================================
# TITLE
# =====================================

st.title("Retail Sales Forecasting and Inventory Demand Planning")

st.write(
    "This application analyzes historical retail sales data "
    "and predicts future sales using Machine Learning."
)

# =====================================
# FILE UPLOAD
# =====================================

uploaded_file = st.file_uploader(
    "Upload Retail Sales CSV File",
    type=["csv"]
)

# =====================================
# RUN AFTER UPLOAD
# =====================================

if uploaded_file is not None:

    # =====================================
    # LOAD DATASET
    # =====================================

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # =====================================
    # PREPROCESSING
    # =====================================

    df = df[['Date', 'Weekly_Sales']].copy()

    # Aggregate sales by date
    df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

    # Rename columns for Prophet
    df.rename(columns={
        'Date': 'ds',
        'Weekly_Sales': 'y'
    }, inplace=True)

    # Convert date format
    df['ds'] = pd.to_datetime(df['ds'])

    # Sort values
    df = df.sort_values('ds')

    # Remove missing values
    df = df.dropna()

    # =====================================
    # KPI METRICS
    # =====================================

    st.subheader("Sales KPIs")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"{df['y'].sum():,.0f}"
    )

    col2.metric(
        "Average Sales",
        f"{df['y'].mean():,.0f}"
    )

    col3.metric(
        "Peak Sales",
        f"{df['y'].max():,.0f}"
    )

    # =====================================
    # SALES TREND ANALYSIS
    # =====================================

    st.subheader("Sales Trend Analysis")

    fig1, ax1 = plt.subplots(figsize=(12,5))

    ax1.plot(df['ds'], df['y'])

    ax1.set_title("Historical Retail Sales Trend")

    ax1.set_xlabel("Date")

    ax1.set_ylabel("Sales")

    st.pyplot(fig1)

    # =====================================
    # MONTHLY SALES ANALYSIS
    # =====================================

    st.subheader("Monthly Sales Analysis")

    df['month'] = df['ds'].dt.month

    monthly_sales = df.groupby('month')['y'].mean()

    fig2, ax2 = plt.subplots(figsize=(10,5))

    monthly_sales.plot(kind='bar', ax=ax2)

    ax2.set_title("Average Monthly Sales")

    ax2.set_xlabel("Month")

    ax2.set_ylabel("Average Sales")

    st.pyplot(fig2)

    # =====================================
    # MACHINE LEARNING FORECASTING
    # =====================================

    st.subheader("Machine Learning Forecasting")

    model = Prophet()

    model.fit(df[['ds', 'y']])

    # Create future dates
    future = model.make_future_dataframe(periods=30)

    # Predict future sales
    forecast = model.predict(future)

    # =====================================
    # FULL FORECAST GRAPH
    # =====================================

    st.subheader("Historical + Forecasted Sales")

    fig3 = model.plot(forecast)

    st.pyplot(fig3)

    # =====================================
    # FUTURE 30 DAYS FORECAST ONLY
    # =====================================

    st.subheader("Next 30 Days Future Sales Forecast")

    future_only = forecast.tail(30)

    fig_future, ax_future = plt.subplots(figsize=(12,5))

    ax_future.plot(
        future_only['ds'],
        future_only['yhat'],
        marker='o'
    )

    ax_future.set_title(
        "Next 30 Days Predicted Sales"
    )

    ax_future.set_xlabel("Future Dates")

    ax_future.set_ylabel("Predicted Sales")

    st.pyplot(fig_future)

    # =====================================
    # FORECAST COMPONENTS
    # =====================================

    st.subheader("Trend and Seasonality")

    fig4 = model.plot_components(forecast)

    st.pyplot(fig4)

    # =====================================
    # ACTUAL VS PREDICTED
    # =====================================

    st.subheader("Actual vs Predicted Sales")

    fig5, ax5 = plt.subplots(figsize=(12,5))

    ax5.plot(
        df['ds'],
        df['y'],
        label='Actual Sales'
    )

    ax5.plot(
        forecast['ds'],
        forecast['yhat'],
        label='Predicted Sales'
    )

    ax5.legend()

    ax5.set_title("Actual vs Predicted Sales")

    st.pyplot(fig5)

    # =====================================
    # MODEL EVALUATION
    # =====================================

    st.subheader("Model Evaluation")

    train = df[:-30]

    test = df[-30:]

    eval_model = Prophet()

    eval_model.fit(train)

    future_eval = eval_model.make_future_dataframe(
        periods=30
    )

    forecast_eval = eval_model.predict(future_eval)

    pred = forecast_eval.tail(30)['yhat'].values

    actual = test['y'].values

    mae = mean_absolute_error(actual, pred)

    rmse = np.sqrt(
        mean_squared_error(actual, pred)
    )

    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # =====================================
    # BUSINESS INSIGHTS
    # =====================================

    st.subheader("Business Insights")

    st.write(
        """
        - Sales show seasonal trends across different months.
        - Machine Learning predicts future retail sales demand.
        - Forecasting helps improve inventory planning.
        - Data visualization supports business analysis.
        - Sales insights help businesses make better decisions.
        """
    )

else:

    st.info(
        "Please upload a CSV dataset to continue."
    )