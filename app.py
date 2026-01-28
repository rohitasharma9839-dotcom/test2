# ===========================
# Imports
# ===========================
import streamlit as st
import pandas as pd
import numpy as np
import importlib.util
import sys
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Online Shoppers Classification", layout="wide")
st.title("🛒 Online Shoppers Purchasing Intention Classifier")
st.write("Select a model to evaluate on the test dataset and see the metrics.")

# ===========================
# Download test dataset from GitHub
# ===========================
csv_url = "https://raw.githubusercontent.com/rohitasharma9839-dotcom/test2/refs/heads/main/test.csv"  # <-- replace with your raw CSV link
test_df = pd.read_csv(csv_url)

st.subheader("Test Dataset Preview")
st.dataframe(test_df.head())

# Provide download link for test CSV
st.markdown(f"[Download test.csv]({csv_url})", unsafe_allow_html=True)

# ===========================
# Model files mapping
# ===========================
model_files = {
    "Logistic Regression": "model/logistic_regression.ipynb",
    "Decision Tree": "model/decision_tree.ipynb",
    "KNN": "model/knn.ipynb",
    "Naive Bayes": "model/naive_bayes.ipynb",
    "Random Forest": "model/random_forest.ipynb",
    "XGBoost": "model/xgboost.ipynb"
}

# ===========================
# Model selection
# ===========================
selected_model = st.selectbox("Select Model", list(model_files.keys()))

# ===========================
# Run selected model
# ===========================
if st.button("Run Model"):
    model_path = model_files[selected_model]

    # Dynamically import model file
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model_module"] = model_module
    spec.loader.exec_module(model_module)

    # Note: Each model file should print metrics to console or return a dictionary named 'metrics'
    if hasattr(model_module, "metrics"):
        metrics = model_module.metrics
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(metrics["Accuracy"],4))
        col2.metric("AUC", round(metrics["AUC"],4))
        col3.metric("MCC", round(metrics["MCC"],4))
        col4, col5, col6 = st.columns(3)
        col4.metric("Precision", round(metrics["Precision"],4))
        col5.metric("Recall", round(metrics["Recall"],4))
        col6.metric("F1 Score", round(metrics["F1 Score"],4))
    else:
        st.info("Metrics printed in console by the model file.")

    # Display confusion matrix and classification report if available
    if hasattr(model_module, "y_true") and hasattr(model_module, "y_pred"):
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(model_module.y_true, model_module.y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("📋 Classification Report")
        report = classification_report(model_module.y_true, model_module.y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())


