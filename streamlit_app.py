import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = os.path.join('Data', 'Womens Clothing E-Commerce Reviews_Dataset.csv')

st.set_page_config(page_title="Women's Clothing E-Commerce Analytics", layout="wide")

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Show raw columns for debugging
    st.sidebar.write("Raw columns:", df.columns.tolist())
    
    # basic cleaning / rename similar to backend/app.py
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    
    # Map various possible column names to our desired names
    column_mappings = {
        'Recommended IND': 'Recommended',
        'Division Name': 'Division',
        'Department Name': 'Department',
        'Class Name': 'Product name',
        'Class Name ': 'Product name'  # Notice the extra space
    }
    
    # Only rename columns that exist
    renames = {old: new for old, new in column_mappings.items() if old in df.columns}
    if renames:
        df.rename(columns=renames, inplace=True)
    
    # drop columns if present
    for col in ['Title', 'Review Text', 'Positive Feedback Count']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Show cleaned columns for debugging
    st.sidebar.write("After cleaning:", df.columns.tolist())
    
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
    # Debug: show dataframe columns so we can diagnose missing plots
    st.write('Columns present:', df.columns.tolist())

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
        # interactive controls for clarity
        bins = st.slider('Histogram bins', min_value=10, max_value=100, value=50, key='age_bins')
        show_kde = st.checkbox('Show KDE (notebook-style only)', value=False, key='age_kde')
        show_stats = st.checkbox('Show mean/median lines', value=True, key='age_stats')

        if notebook_style:
            plt.figure(figsize=(12,5))
            sns.set_style('whitegrid')
            sns.histplot(df['Age'], kde=show_kde, bins=bins, color='skyblue')
            mean_age = df['Age'].mean()
            median_age = df['Age'].median()
            if show_stats:
                plt.axvline(mean_age, color='red', linestyle='--', label=f'Mean: {mean_age:.1f}')
                plt.axvline(median_age, color='green', linestyle='-.', label=f'Median: {median_age:.1f}')
                plt.legend()
            plt.xlabel('Age')
            plt.title('Customer Age Distribution')
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            # Plotly interactive histogram with optional mean/median lines
            fig2 = px.histogram(df, x='Age', nbins=bins, title='Customer Age Distribution', color_discrete_sequence=['rgb(140,200,250)'])
            fig2.update_layout(height=480, margin=dict(t=40))
            if show_stats:
                mean_age = df['Age'].mean()
                median_age = df['Age'].median()
                fig2.add_vline(x=mean_age, line=dict(color='red', dash='dash'), annotation_text=f'Mean: {mean_age:.1f}', annotation_position='top left')
                fig2.add_vline(x=median_age, line=dict(color='green', dash='dot'), annotation_text=f'Median: {median_age:.1f}', annotation_position='top right')
            # add a small boxplot below as marginal
            fig2.update_traces(marker_line_width=0.5, marker_line_color='rgba(0,0,0,0.05)')
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

# Additional visualizations ported from the notebook
st.header('Extra visualizations from notebook')

# Correlation heatmap (numeric columns)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    st.subheader('Correlation matrix (numeric features)')
    corr = df[num_cols].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap='rocket', annot=False, ax=ax)
        ax.set_title('Correlation matrix of features', fontweight='bold')
        st.pyplot(fig)
        plt.clf()
else:
    st.info('Not enough numeric columns to show correlation matrix.')

st.markdown('---')

# Division / Department / Product counts using Plotly for better interactivity
st.subheader('Product Categories Distribution')

# Find relevant columns
division_col = next((c for c in df.columns if 'division' in c.lower()), None)
dept_col = next((c for c in df.columns if 'department' in c.lower()), None)
product_col = next((c for c in df.columns if any(x in c.lower() for x in ['product', 'class'])), None)

# Division counts
if division_col:
    st.write(f"### {division_col} Distribution")
    counts = df[division_col].value_counts()
    fig1 = go.Figure(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation='h',
        marker_color='rgb(55,126,184)'
    ))
    fig1.update_layout(
        height=400,
        margin=dict(l=200, r=20, t=30, b=50),
        xaxis_title="Count",
        yaxis_title=division_col,
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

# Department counts
if dept_col:
    st.write(f"### {dept_col} Distribution")
    counts = df[dept_col].value_counts()
    fig2 = go.Figure(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation='h',
        marker_color='rgb(77,175,74)'
    ))
    fig2.update_layout(
        height=400,
        margin=dict(l=200, r=20, t=30, b=50),
        xaxis_title="Count",
        yaxis_title=dept_col,
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# Product counts (top 20)
if product_col:
    st.write(f"### Top 20 {product_col} by Count")
    counts = df[product_col].value_counts().nlargest(20)
    fig3 = go.Figure(go.Bar(
        y=counts.index,
        x=counts.values,
        orientation='h',
        marker_color='rgb(152,78,163)'
    ))
    fig3.update_layout(
        height=600,
        margin=dict(l=200, r=20, t=30, b=50),
        xaxis_title="Count",
        yaxis_title=product_col,
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

if not any([division_col, dept_col, product_col]):
    st.error("No category columns found in the data. Expected columns containing 'division', 'department', or 'class'/'product'.")
    st.write("Available columns:", df.columns.tolist())

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

st.markdown('---')

# Rating distribution per product (average rating)
st.subheader('Average rating per product (sample)')
if 'Rating' in df.columns and 'Product name' in df.columns:
    data_rating = df.groupby('Product name', as_index=False)['Rating'].mean().sort_values('Rating', ascending=False)
    # show top 20 by average rating
    sample = data_rating.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Rating', y='Product name', data=sample, palette='Oranges', ax=ax)
    ax.set_xlabel('Average Rating')
    ax.set_ylabel('Product name')
    ax.set_title('Rating distribution of clothing products (top 20 by avg rating)')
    st.pyplot(fig)
    plt.clf()
else:
    st.info('Missing `Rating` or `Product name` column to show rating distribution.')

st.markdown('---')

# Recommended vs Not Recommended counts per product (top products)
st.subheader('Recommended vs Not recommended (top products)')
if 'Recommended' in df.columns and 'Product name' in df.columns:
    top = df['Product name'].value_counts().nlargest(20).index
    grouped = df[df['Product name'].isin(top)].groupby(['Product name', 'Recommended']).size().unstack(fill_value=0)
    # plot as grouped bar chart using plotly for clarity
    traces = []
    for col in grouped.columns:
        traces.append(go.Bar(name=str(col), x=grouped.index, y=grouped[col]))
    figp = go.Figure(data=traces)
    figp.update_layout(barmode='group', xaxis_tickangle=-45, height=500, title='Recommended (1) vs Not recommended (0) for top 20 products')
    st.plotly_chart(figp, use_container_width=True)
else:
    st.info('Missing `Recommended` or `Product name` column to show recommended counts.')

st.caption('This Streamlit app reads the same CSV used by the Flask backend. You can run `streamlit run streamlit_app.py` to launch it locally.')
