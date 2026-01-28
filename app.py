# ===========================
# Imports
# ===========================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Online Shoppers Classification", layout="wide")
st.title("🛒 Online Shoppers Purchasing Intention Classifier")
st.write("Upload a CSV test dataset, select a model, and see evaluation results.")

# ===========================
# Dataset Upload
# ===========================
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Separate features and target
    if "Revenue" not in data.columns:
        st.error("CSV must contain a 'Revenue' column (target).")
    else:
        X = data.drop("Revenue", axis=1)
        y = data["Revenue"].map({False:0, True:1})

        # Identify feature types
        categorical_features = X.select_dtypes(include=["object"]).columns
        numerical_features = X.select_dtypes(exclude=["object"]).columns

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )

        # ===========================
        # Model Selection
        # ===========================
        model_choice = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
        )

        # ===========================
        # Train & Evaluate Model
        # ===========================
        def evaluate(y_test, y_pred, y_proba):
            return {
                "Accuracy": accuracy_score(y_test, y_pred),
                "AUC": roc_auc_score(y_test, y_proba),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred),
                "MCC": matthews_corrcoef(y_test, y_pred)
            }

        if model_choice == "Naive Bayes":
            X_transformed = preprocessor.fit_transform(X)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            nb_model = GaussianNB()
            nb_model.fit(X_transformed, y)
            y_pred = nb_model.predict(X_transformed)
            y_proba = nb_model.predict_proba(X_transformed)[:,1]
            metrics = evaluate(y, y_pred, y_proba)

        else:
            # Map models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)
            }
            clf = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", models[model_choice])
            ])
            clf.fit(X, y)
            y_pred = clf.predict(X)
            y_proba = clf.predict_proba(X)[:,1]
            metrics = evaluate(y, y_pred, y_proba)

        # ===========================
        # Display Metrics
        # ===========================
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(metrics["Accuracy"],4))
        col2.metric("AUC", round(metrics["AUC"],4))
        col3.metric("MCC", round(metrics["MCC"],4))
        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", round(metrics["Precision"],4))
        col5.metric("Recall", round(metrics["Recall"],4))
        col6.metric("F1 Score", round(metrics["F1 Score"],4))

        # ===========================
        # Confusion Matrix
        # ===========================
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ===========================
        # Classification Report
        # ===========================
        st.subheader("📋 Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
