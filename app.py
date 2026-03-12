import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(page_title="Revenue Intelligence Dashboard", layout="wide")

st.title("📊 Customer Churn & RFM Intelligence")
st.markdown("""
This dashboard identifies high-value customer segments and predicts 
revenue at risk in the Olist e-commerce dataset.
""")

# 2. Load Data (Simplified for the app)
@st.cache_data # This keeps the app fast!
def load_data():
    df = pd.read_csv('olist_sample.csv')
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Filter Analytics")
state_filter = st.sidebar.multiselect("Select States", options=df['customer_state'].unique(), default=['SP', 'RJ', 'MG'])

# 4. Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{len(df):,}")
col2.metric("Unique Customers", f"{df['customer_unique_id'].nunique():,}")
col3.metric("Avg Order Value", f"${df['payment_value'].mean():.2f}")

# 5. Visualization
st.subheader("Orders by Day of Week")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['weekday'] = df['order_purchase_timestamp'].dt.day_name()

fig, ax = plt.subplots()
sns.countplot(data=df[df['customer_state'].isin(state_filter)], x='weekday', ax=ax, palette='viridis')
st.pyplot(fig)
import datetime as dt

st.subheader("Monthly Cohort Retention (%)")

# 1. Processing the Cohorts
@st.cache_data
def get_retention_data(df):
    # Ensure dates are correct
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    
    # Helper for first day of month
    def get_month(x): return dt.datetime(x.year, x.month, 1)
    
    df['OrderMonth'] = df['order_purchase_timestamp'].apply(get_month)
    df['CohortMonth'] = df.groupby('customer_unique_id')['OrderMonth'].transform('min')
    
    # Calculate index
    years_diff = df['OrderMonth'].dt.year - df['CohortMonth'].dt.year
    months_diff = df['OrderMonth'].dt.month - df['CohortMonth'].dt.month
    df['CohortIndex'] = years_diff * 12 + months_diff
    
    # Create Matrix
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['customer_unique_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='customer_unique_id')
    
    # Calculate Percentages
    cohort_sizes = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)
    retention.index = retention.index.strftime('%Y-%m')
    return retention

# 2. Display the Heatmap
retention_matrix = get_retention_data(df)

fig_cohort, ax_cohort = plt.subplots(figsize=(12, 8))
sns.heatmap(retention_matrix, annot=True, fmt='.1%', cmap='YlGnBu', vmin=0.0, vmax=0.05, ax=ax_cohort)
ax_cohort.set_title("Retention Rates by Cohort")
st.pyplot(fig_cohort)

st.divider()
st.header("🎯 Targeted Recovery List")
st.write("Download the list of 'At Risk' and 'Lost' customers for marketing outreach.")

# 1. Identify At-Risk Customers (180+ Days)
snapshot_date = df['order_purchase_timestamp'].max() + dt.timedelta(days=1)
rfm = df.groupby('customer_unique_id').agg({
    'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
    'order_id': 'count',
    'payment_value': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# 2. Filter for Churned (180+ days)
churn_list = rfm[rfm['Recency'] >= 180].sort_values('Monetary', ascending=False)

# 3. Streamlit Download Button
csv = churn_list.to_csv().encode('utf-8')

st.download_button(
    label="📥 Download Churn Recovery List (CSV)",
    data=csv,
    file_name='at_risk_customers.csv',
    mime='text/csv',
    help="Click to export customers who haven't purchased in over 180 days."
)

st.dataframe(churn_list.head(10)) # Show a preview of the top 10

from sklearn.linear_model import LinearRegression
import numpy as np

st.divider()
st.header("📈 Revenue Growth Forecast")
st.write("This model uses Linear Regression to predict future monthly revenue based on historical trends.")

# 1. Prepare Time-Series Data
@st.cache_data
def get_forecast_data(df):
    # Aggregating revenue by month
    df['Month'] = df['order_purchase_timestamp'].dt.to_period('M')
    monthly_rev = df.groupby('Month')['payment_value'].sum().reset_index()
    monthly_rev['Month_Index'] = np.arange(len(monthly_rev)) # 0, 1, 2... for the model
    return monthly_rev

monthly_data = get_forecast_data(df)

# 2. Train the Linear Regression Model
X = monthly_data[['Month_Index']] # Features (Time)
y = monthly_data['payment_value']  # Target (Revenue)

model = LinearRegression()
model.fit(X, y)

# 3. Predict the Next 3 Months
future_months = np.array([len(monthly_data), len(monthly_data)+1, len(monthly_data)+2]).reshape(-1, 1)
predictions = model.predict(future_months)

# 4. Visualization
fig_forecast, ax_f = plt.subplots(figsize=(10, 5))
ax_f.plot(monthly_data['Month'].astype(str), y, label='Actual Revenue', marker='o', color='#1f77b4')

ax_f.plot(['Next Month', 'In 2 Months', 'In 3 Months'], predictions, label='Forecast', linestyle='--', marker='s', color='#ff7f0e')

ax_f.set_title("Historical Revenue vs. 3-Month Forecast")
ax_f.set_ylabel("Revenue ($)")
plt.xticks(rotation=45)
ax_f.legend()
st.pyplot(fig_forecast)

# 5. Display Forecast Metrics
col_f1, col_f2 = st.columns(2)
col_f1.metric("Predicted Revenue (Next Month)", f"${predictions[0]:,.2f}")
col_f2.metric("Growth Trend", "Upward" if model.coef_[0] > 0 else "Downward")