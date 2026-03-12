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
    df = pd.read_csv(r"C:\Users\TLHOMAMO PHETO\Documents\Python Code Projects\Segmentation_Model\Brazilian E-Commerce Public Dataset by Olist.csv")
    return df

df = load_data()

# 3. Sidebar Filters
st.sidebar.header("Filter Analytics")
state_filter = st.sidebar.multiselect("Select States", options=df['customer_state'].unique(), default=['SP', 'RJ', 'MG'])

# 4. Metrics Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{len(df):,}")
col2.metric("Unique Customers", f"{df['customer_unique_id'].nunique():,}")
col3.metric("Avg Order Value", f"${df['payment_value'].mean():.2f}")b

# 5. Visualization
st.subheader("Orders by Day of Week")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['weekday'] = df['order_purchase_timestamp'].dt.day_name()

fig, ax = plt.subplots()
sns.countplot(data=df[df['customer_state'].isin(state_filter)], x='weekday', ax=ax, palette='viridis')
st.pyplot(fig)