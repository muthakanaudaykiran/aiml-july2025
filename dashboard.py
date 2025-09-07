import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Age
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    # Employment years
    df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(
        lambda x: np.nan if x > 100000 else -x / 365.25
    )

    # Ratios
    df["DTI"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["LOAN_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

    # Income brackets safely
    try:
        df["INCOME_BRACKET"] = pd.qcut(
            df["AMT_INCOME_TOTAL"], q=4,
            labels=["Low","Mid-Low","Mid-High","High"],
            duplicates='drop'
        )
    except:
        df["INCOME_BRACKET"] = "Unknown"

    return df

# -------------------------------
# Page 1 — Overview & Data Quality
# -------------------------------
def page_overview(df):
    st.header("📊 Overview & Data Quality")
    if df.empty:
        st.warning("DataFrame is empty. Upload a valid CSV.")
        return

    # --- Basic KPIs ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", f"{df['SK_ID_CURR'].nunique():,}" if 'SK_ID_CURR' in df.columns else "N/A")
    default_rate = df['TARGET'].mean() if 'TARGET' in df.columns else np.nan
    col2.metric("Default Rate (%)", f"{default_rate*100:.2f}%" if not np.isnan(default_rate) else "N/A")
    col3.metric("Repaid Rate (%)", f"{(1-default_rate)*100:.2f}%" if not np.isnan(default_rate) else "N/A")

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Features", df.shape[1])
    
    # --- Missing Values ---
    missing_pct = df.isnull().mean() * 100
    avg_missing = missing_pct.mean()
    col5.metric("Avg Missing / Feature (%)", f"{avg_missing:.2f}%")
    
    col6.metric("Numerical Features", df.select_dtypes(include=['int64','float64']).shape[1])
    st.write("Missing Values (Top 20 columns)")
    top_missing = missing_pct.sort_values(ascending=False).head(20)
    if not top_missing.empty:
        fig, ax = plt.subplots(figsize=(10,5))
        top_missing.plot(kind="bar", ax=ax)
        ax.set_ylabel("% Missing")
        ax.set_xlabel("Feature")
        ax.set_title("Top 20 Missing Features")
        st.pyplot(fig)
    else:
        st.write("No missing values detected.")

    # --- Categorical/Numeric counts ---
    col7, col8 = st.columns(2)
    col7.metric("Categorical Features", df.select_dtypes(include=['object']).shape[1])
    col8.metric("Median Age (Years)", f"{df['AGE_YEARS'].median():.1f}" if 'AGE_YEARS' in df.columns else "N/A")

    # --- Optional Target distribution ---
    if 'TARGET' in df.columns and not df['TARGET'].empty:
        st.subheader("Target Distribution")
        target_counts = df['TARGET'].value_counts(normalize=True)
        if not target_counts.empty:
            fig, ax = plt.subplots()
            target_counts.plot(kind="bar", ax=ax, color=["red","blue"], rot=0)
            ax.set_xticklabels(["Repaid (0)", "Default (1)"])
            ax.set_ylabel("Proportion")
            st.pyplot(fig)

# -------------------------------
# Page 2 — Target & Risk Segmentation
# -------------------------------
def page_segmentation(df):
    st.header("🧩 Target & Risk Segmentation")
    if df.empty:
        st.warning("DataFrame is empty. Upload a valid CSV.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Total Defaults", f"{df['TARGET'].sum():,}" if 'TARGET' in df.columns else "N/A")
    col2.metric("Default Rate (%)", f"{df['TARGET'].mean()*100:.2f}%" if 'TARGET' in df.columns else "N/A")

    for col in ["CODE_GENDER","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE"]:
        if col in df.columns and 'TARGET' in df.columns:
            st.subheader(f"Default Rate by {col}")
            grouped = df.groupby(col)["TARGET"].mean()
            if not grouped.empty:
                fig, ax = plt.subplots()
                grouped.plot(kind="bar", ax=ax)
                ax.set_ylabel("Default Rate")
                st.pyplot(fig)

    # Avg values for defaulters
    if 'TARGET' in df.columns:
        def_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EMPLOYMENT_YEARS']
        for c in def_cols:
            if c in df.columns:
                st.metric(f"Avg {c} — Defaulters", f"{df.loc[df['TARGET']==1,c].mean():,.0f}")

# -------------------------------
# Page 3 — Demographics & Household Profile
# -------------------------------
def page_demographics(df):
    st.header("🏠 Demographics & Household Profile")
    if df.empty:
        st.warning("DataFrame is empty. Upload a valid CSV.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("% Male", f"{(df['CODE_GENDER']=='M').mean()*100:.2f}%" if 'CODE_GENDER' in df.columns else "N/A")
    col2.metric("Avg Age — Defaulters", f"{df.loc[df['TARGET']==1,'AGE_YEARS'].mean():.1f}" if 'TARGET' in df.columns and 'AGE_YEARS' in df.columns else "N/A")
    col3.metric("Avg Age — Non-Defaulters", f"{df.loc[df['TARGET']==0,'AGE_YEARS'].mean():.1f}" if 'TARGET' in df.columns and 'AGE_YEARS' in df.columns else "N/A")

    col4, col5, col6 = st.columns(3)
    col4.metric("% With Children", f"{(df['CNT_CHILDREN']>0).mean()*100:.2f}%" if 'CNT_CHILDREN' in df.columns else "N/A")
    col5.metric("Avg Family Size", f"{df['CNT_FAM_MEMBERS'].mean():.1f}" if 'CNT_FAM_MEMBERS' in df.columns else "N/A")
    col6.metric("% Married", f"{(df['NAME_FAMILY_STATUS'].isin(['Married','Civil marriage'])).mean()*100:.2f}%" if 'NAME_FAMILY_STATUS' in df.columns else "N/A")

    col7, col8, col9 = st.columns(3)
    col7.metric("% Higher Education", f"{df['NAME_EDUCATION_TYPE'].isin(['Higher education','Academic degree']).mean()*100:.2f}%" if 'NAME_EDUCATION_TYPE' in df.columns else "N/A")
    col8.metric("% Living With Parents", f"{(df['NAME_HOUSING_TYPE']=='With parents').mean()*100:.2f}%" if 'NAME_HOUSING_TYPE' in df.columns else "N/A")
    col9.metric("% Currently Working", f"{df['OCCUPATION_TYPE'].notna().mean()*100:.2f}%" if 'OCCUPATION_TYPE' in df.columns else "N/A")

    col10 = st.columns(1)
    col10[0].metric("Avg Employment Years", f"{df['EMPLOYMENT_YEARS'].mean():.1f}" if 'EMPLOYMENT_YEARS' in df.columns else "N/A")

# -------------------------------
# Page 4 — Financial Health & Affordability
# -------------------------------
def page_financials(df):
    st.header("💰 Financial Health & Affordability")
    if df.empty:
        st.warning("DataFrame is empty. Upload a valid CSV.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Annual Income", f"{df['AMT_INCOME_TOTAL'].mean():,.0f}" if 'AMT_INCOME_TOTAL' in df.columns else "N/A")
    col2.metric("Median Annual Income", f"{df['AMT_INCOME_TOTAL'].median():,.0f}" if 'AMT_INCOME_TOTAL' in df.columns else "N/A")
    col3.metric("Avg Credit Amount", f"{df['AMT_CREDIT'].mean():,.0f}" if 'AMT_CREDIT' in df.columns else "N/A")

    col4, col5, col6 = st.columns(3)
    col4.metric("Avg Annuity", f"{df['AMT_ANNUITY'].mean():,.0f}" if 'AMT_ANNUITY' in df.columns else "N/A")
    col5.metric("Avg Goods Price", f"{df['AMT_GOODS_PRICE'].mean():,.0f}" if 'AMT_GOODS_PRICE' in df.columns else "N/A")
    col6.metric("Avg DTI", f"{df['DTI'].mean():.2f}" if 'DTI' in df.columns else "N/A")

# -------------------------------
# Page 5 — Correlations & Drivers
# -------------------------------
def page_correlations(df):
    st.header("📈 Correlations & Drivers")
    if df.empty:
        st.warning("DataFrame is empty. Upload a valid CSV.")
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols)==0:
        st.write("No numeric columns to compute correlations.")
        return

    corr = df[numeric_cols].corr()
    if 'TARGET' in corr.columns:
        target_corr = corr['TARGET'].drop('TARGET')
        st.subheader("Top 5 +Corr with TARGET")
        st.write(target_corr.tail(5))
        st.subheader("Top 5 −Corr with TARGET")
        st.write(target_corr.head(5))

# -------------------------------
# Main App
# -------------------------------
st.set_page_config(page_title="Credit Default Dashboard", layout="wide")
st.sidebar.title("Navigation")

uploaded_file = st.sidebar.file_uploader(
    "Upload application_train.csv", type=["csv"], key="file_uploader_unique"
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocess(df)

    page = st.sidebar.radio("Go to", ["Overview", "Risk Segmentation", "Demographics", "Financial Health", "Correlations"])
    if page=="Overview":
        page_overview(df)
    elif page=="Risk Segmentation":
        page_segmentation(df)
    elif page=="Demographics":
        page_demographics(df)
    elif page=="Financial Health":
        page_financials(df)
    elif page=="Correlations":
        page_correlations(df)
else:
    st.info("Please upload the application_train.csv file to start.")
