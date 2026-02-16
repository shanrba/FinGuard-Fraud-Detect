# ============================================================
# FinGuard AI - Production Fraud Detection System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import lime
import lime.lime_tabular

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(page_title="FinGuard AI Risk Engine", layout="wide")

st.sidebar.title("🛡️ FinGuard AI")

menu = [
    "1️⃣ Business Problem",
    "2️⃣ Data & EDA",
    "3️⃣ Modeling & Tuning",
    "4️⃣ Explainability",
    "5️⃣ Live Fraud Demo"
]

choice = st.sidebar.radio("Navigation", menu)

if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None


# ============================================================
# 1️⃣ BUSINESS PROBLEM
# ============================================================
def business_module():
    st.header("🏦 Business Problem & Gap Analysis")

    st.markdown("""
    ### 🚨 Current Challenges
    - Rule-based fraud detection
    - High false positives → customer friction
    - Missed fraud → direct financial loss
    - No explainability for regulators

    ### 🎯 Target State
    - ML-powered fraud probability scoring
    - Cost-sensitive threshold tuning
    - Explainable AI (LIME)
    - Real-time risk decision support
    """)

    st.info("""
    Fraud datasets are highly imbalanced (very few fraud cases).
    Therefore, accuracy alone is misleading.
    We focus on ROC-AUC and PR-AUC instead.
    """)


# ============================================================
# 2️⃣ DATA & EDA
# ============================================================
def eda_module():
    st.header("📊 Data Preprocessing & EDA")

    uploaded_file = st.file_uploader("Upload Fraud Dataset (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

    if st.session_state.df is None:
        return

    df = st.session_state.df

    st.subheader("Dataset Overview")
    st.write(df.head())

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Fraud Distribution
    if "Is_Fraud" in df.columns:
        st.subheader("Class Distribution")
        fig = px.histogram(df, x="Is_Fraud", color="Is_Fraud",
                           title="Fraud vs Legit Transactions")
        st.plotly_chart(fig)

    # Merchant Category Pattern
    if "Merchant_Category" in df.columns:
        st.subheader("Fraud Rate by Merchant Category")
        fraud_by_cat = df.groupby("Merchant_Category")["Is_Fraud"].mean().reset_index()
        fig = px.bar(fraud_by_cat, x="Merchant_Category", y="Is_Fraud")
        st.plotly_chart(fig)

    # Correlation
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r")
    st.plotly_chart(fig)


# ============================================================
# 3️⃣ MODELING & TUNING
# ============================================================
def modeling_module():
    st.header("🧠 Modeling & Hyperparameter Tuning")

    if st.session_state.df is None:
        st.warning("Upload dataset first.")
        return

    df = st.session_state.df.copy()

    if "Is_Fraud" not in df.columns:
        st.error("Target column 'Is_Fraud' not found.")
        return

    # ------------------------------
    # Feature Engineering
    # ------------------------------
    X = df.drop("Is_Fraud", axis=1)

    # Remove ID / high-cardinality columns
    drop_cols = ["Transaction_ID", "Customer_ID", "Timestamp"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # Encode low-cardinality categorical columns
    cat_cols = X.select_dtypes(include="object").columns
    low_card_cols = [col for col in cat_cols if X[col].nunique() < 20]

    X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

    # Convert to float32 (reduce memory)
    X = X.astype(np.float32)

    y = df["Is_Fraud"]

    # Store feature columns for Live Demo alignment
    st.session_state.feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------
    # Imbalance Handling
    # ------------------------------
    imbalance_method = st.radio(
        "Imbalance Handling",
        ["SMOTE", "Class Weight"]
    )

    if imbalance_method == "SMOTE":
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # ------------------------------
    # Model Selection
    # ------------------------------
    algo = st.selectbox(
        "Select Ensemble Model",
        ["Random Forest", "AdaBoost", "XGBoost"]
    )

    if st.button("🚀 Train Model"):

        if algo == "Random Forest":
            model = RandomForestClassifier(
                class_weight="balanced" if imbalance_method == "Class Weight" else None,
                random_state=42,
                n_jobs=-1
            )
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10]
            }

        elif algo == "AdaBoost":
            model = AdaBoostClassifier(random_state=42)
            param_grid = {"n_estimators": [50, 100]}

        else:
            model = XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5]
            }

        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # Save everything to session
        st.session_state.model = best_model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_train = X_train

        # ------------------------------
        # Evaluation
        # ------------------------------
        y_probs = best_model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_probs)

        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)

        st.success("Model Trained Successfully")

        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC", f"{roc_auc:.4f}")
        col2.metric("PR-AUC", f"{pr_auc:.4f}")

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        fig_roc = px.line(x=fpr, y=tpr, title="ROC Curve")
        st.plotly_chart(fig_roc)

        # PR Curve
        fig_pr = px.line(x=recall, y=precision,
                         title="Precision-Recall Curve")
        st.plotly_chart(fig_pr)

# ============================================================
# 4️⃣ EXPLAINABILITY
# ============================================================
def explainability_module():
    st.header("🔎 Explainability & Diagnostics")

    if st.session_state.model is None:
        st.warning("Train model first.")
        return

    model = st.session_state.model
    X_test = st.session_state.X_test
    X_train = st.session_state.X_train
    y_test = st.session_state.y_test

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    st.write(cm)

    st.markdown("""
    ### Bias-Variance Discussion
    - Random Forest: Low variance, robust
    - AdaBoost: Focuses on hard samples
    - XGBoost: Handles nonlinear patterns efficiently
    """)

    st.subheader("LIME Explanation")

    # Convert to NumPy (CRITICAL FIX)
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)

    explainer = lime.lime_tabular.LimeTabularExplainer(
      training_data=X_train_np,
      feature_names=st.session_state.feature_columns,
      class_names=["Legit", "Fraud"],
      mode="classification"
    )

    idx = st.slider(
      "Select Transaction Index",
      0,
      len(X_test_np) - 1,
      0)

    exp = explainer.explain_instance(
      X_test_np[idx],
      model.predict_proba,
      num_features=8
    )

    fig = exp.as_pyplot_figure()
    st.pyplot(fig)



# ============================================================
# 5️⃣ LIVE FRAUD DEMO
# ============================================================
def live_demo_module():
    st.header("🛡️ Live Fraud Risk Scoring")

    if st.session_state.model is None:
        st.warning("Train model first.")
        return

    model = st.session_state.model
    feature_columns = st.session_state.feature_columns

    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount", 0.0)
    balance = st.number_input("Account Balance", 0.0)
    monthly_spend = st.number_input("Average Monthly Spend", 0.0)

    if st.button("Predict Risk"):

        # Create base dataframe
        input_raw = pd.DataFrame({
            "Transaction_Amount": [amount],
            "Account_Balance_Pre": [balance],
            "Average_Monthly_Spend": [monthly_spend]
        })

        # Apply same dummy encoding logic
        input_encoded = pd.get_dummies(input_raw)

        # Create empty dataframe with trained feature structure
        aligned_df = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        # Fill matching columns
        for col in input_encoded.columns:
            if col in aligned_df.columns:
                aligned_df[col] = input_encoded[col]

        aligned_df = aligned_df.astype(np.float32)

        prob = model.predict_proba(aligned_df)[0][1]

        if prob > 0.5:
            st.error(f"⚠️ High Fraud Risk ({prob:.2%})")
        else:
            st.success(f"✅ Low Risk ({prob:.2%})")

# ============================================================
# ROUTER
# ============================================================
if choice == "1️⃣ Business Problem":
    business_module()
elif choice == "2️⃣ Data & EDA":
    eda_module()
elif choice == "3️⃣ Modeling & Tuning":
    modeling_module()
elif choice == "4️⃣ Explainability":
    explainability_module()
elif choice == "5️⃣ Live Fraud Demo":
    live_demo_module()
