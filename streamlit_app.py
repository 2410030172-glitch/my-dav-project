import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join('Data', 'Womens Clothing E-Commerce Reviews_Dataset.csv')

st.set_page_config(page_title="Women's Clothing E-Commerce Analytics", layout="wide")

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # basic cleaning / rename similar to backend/app.py
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Recommended IND': 'Recommended',
        'Division Name': 'Division',
        'Department Name': 'Department',
        'Class Name': 'Product name'
    }, inplace=True)
    # drop columns if present
    for col in ['Title', 'Review Text', 'Positive Feedback Count']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Load data
try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

st.title("Women's Clothing E-Commerce Analytics")

# Controls
with st.sidebar:
    st.header('Controls')
    n_top = st.selectbox('Top products (by review count)', [10, 15, 20, 30, 50], index=1)
    rating_n = st.selectbox('Products for rating chart', [10, 15, 20, 30, 50], index=2)
    notebook_style = st.checkbox('Use notebook-style (seaborn/matplotlib) plots', value=False)
    show_age = 'Age' in df.columns
    st.write(f"Rows: {len(df):,} — Columns: {len(df.columns)}")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('Most Reviewed Products')
    top = df['Product name'].value_counts().head(n_top)
    if notebook_style:
        plt.figure(figsize=(8,6))
        sns.set_style('whitegrid')
        sns.barplot(x=top.values, y=top.index, palette='Blues_r')
        plt.xlabel('Review count')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        fig1 = go.Figure(go.Bar(x=top.values[::-1], y=top.index[::-1], orientation='h', marker_color='rgb(37,150,190)'))
        fig1.update_layout(height=420, margin=dict(t=30, b=40, l=120), paper_bgcolor='white')
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader('Customer Age Distribution')
    if show_age:
        age_counts = df['Age'].value_counts().sort_index()
        if notebook_style:
            plt.figure(figsize=(6,4))
            sns.set_style('whitegrid')
            sns.barplot(x=age_counts.index, y=age_counts.values, palette='Blues')
            plt.xlabel('Age')
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            fig2 = go.Figure(go.Bar(x=age_counts.index, y=age_counts.values, marker_color='rgb(140,200,250)'))
            fig2.update_layout(height=420, margin=dict(t=30), xaxis_title='Age')
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info('No `Age` column in dataset.')

st.markdown('---')

st.subheader('Average Product Ratings (top by review count)')
if 'Rating' in df.columns:
    grouped = df.groupby('Product name')['Rating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(rating_n)
    if notebook_style:
        plt.figure(figsize=(10,4))
        sns.set_style('whitegrid')
        sns.barplot(x=grouped.index, y=grouped['mean'], palette='Oranges')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average rating')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        fig3 = go.Figure(go.Bar(x=grouped.index, y=grouped['mean'], marker_color='orange'))
        fig3.update_layout(height=420, margin=dict(t=30), xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info('No `Rating` column in dataset.')

st.markdown('---')
st.caption('This Streamlit app reads the same CSV used by the Flask backend. You can run `streamlit run streamlit_app.py` to launch it locally.')
