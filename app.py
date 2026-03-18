
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout="wide")
st.title("📊 Personal Loan AI Marketing Dashboard")

data = pd.read_csv("train_data.csv")

# Drop unnecessary
X = data.drop(columns=["Personal Loan","ID"])
y = data["Personal Loan"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []

st.subheader("📈 Model Performance Comparison")

fig, ax = plt.subplots()

for name, model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test,y_pred),3),
        "Precision": round(precision_score(y_test,y_pred),3),
        "Recall": round(recall_score(y_test,y_pred),3),
        "F1 Score": round(f1_score(y_test,y_pred),3)
    })

    fpr, tpr, _ = roc_curve(y_test,y_prob)
    ax.plot(fpr, tpr, label=name)

st.dataframe(pd.DataFrame(results))

ax.plot([0,1],[0,1],'--')
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

# Confusion Matrix
st.subheader("📊 Confusion Matrix (Random Forest)")
model = RandomForestClassifier().fit(X_train,y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
st.write("Values:")
st.write(cm)

st.write("Percentage:")
st.write(cm / cm.sum())

# Upload section
st.subheader("🎯 Predict New Customers")
file = st.file_uploader("Upload Test Data", type=["csv"])

if file:
    df = pd.read_csv(file)
    preds = model.predict(df.drop(columns=["ID"]))
    df["Predicted Loan"] = preds
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv")
