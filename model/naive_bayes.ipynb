import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

data = pd.read_csv("../test.csv")
X = data.drop("Revenue", axis=1)
y = data["Revenue"].map({False:0, True:1})

categorical_features = X.select_dtypes(include=["object"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_transformed = preprocessor.fit_transform(X)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

nb_model = GaussianNB()
nb_model.fit(X_transformed, y)
y_pred = nb_model.predict(X_transformed)
y_proba = nb_model.predict_proba(X_transformed)[:,1]

metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": roc_auc_score(y, y_proba),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1 Score": f1_score(y, y_pred),
    "MCC": matthews_corrcoef(y, y_pred)
}

y_true = y
