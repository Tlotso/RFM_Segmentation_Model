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