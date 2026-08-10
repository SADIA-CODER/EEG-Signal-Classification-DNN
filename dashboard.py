import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="EEG Classification Dashboard",
    layout="centered"
)

st.title("EEG Signal Classification Dashboard")

st.write(
    "Alcoholic vs Control EEG Classification using Deep Neural Networks (DNN)"
)

# =====================================
# LOAD DATA
# =====================================

alcoholic = pd.read_csv("alcoholic_eeg_dataset.csv")
control = pd.read_csv("control_eeg_dataset.csv")

alcoholic["Class"] = "Alcoholic"
control["Class"] = "Control"

data = pd.concat(
    [alcoholic, control],
    ignore_index=True
)

# =====================================
# DATASET OVERVIEW
# =====================================

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Samples",
    f"{len(data):,}"
)

col2.metric(
    "Alcoholic Samples",
    f"{len(alcoholic):,}"
)

col3.metric(
    "Control Samples",
    f"{len(control):,}"
)

st.subheader("First 5 Records")

st.dataframe(
    data.head(),
    use_container_width=True
)

# =====================================
# FEATURE STATISTICS
# =====================================

st.header("Feature Statistics")

st.dataframe(
    data.describe(),
    use_container_width=True
)

# =====================================
# CLASS DISTRIBUTION
# =====================================

st.header("Class Distribution")

fig1, ax1 = plt.subplots(figsize=(5,4))

sns.countplot(
    x="Class",
    data=data,
    ax=ax1
)

ax1.set_title(
    "Alcoholic vs Control Samples"
)

st.pyplot(fig1)

# =====================================
# EEG SIGNAL VISUALIZATION
# =====================================

st.header("EEG Signal Visualization")

channels = sorted(
    data["channel"].unique()
)

channel = st.selectbox(
    "Select EEG Channel",
    channels
)

signal = data[
    data["channel"] == channel
]

fig2, ax2 = plt.subplots(
    figsize=(8,4)
)

ax2.plot(
    signal["sample"][:500],
    signal["eeg_value"][:500]
)

ax2.set_title(
    f"EEG Signal - {channel}"
)

ax2.set_xlabel(
    "Sample Number"
)

ax2.set_ylabel(
    "EEG Amplitude"
)

st.pyplot(fig2)

# =====================================
# CORRELATION HEATMAP
# =====================================

st.header("Feature Correlation Matrix")

try:

    features = pd.read_csv(
        "engineered_features.csv"
    )

    numeric_features = features.select_dtypes(
        include="number"
    )

    corr = numeric_features.corr()

    fig_corr, ax_corr = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax_corr
    )

    st.pyplot(fig_corr)

except Exception as e:

    st.error(
        f"Error loading engineered_features.csv: {e}"
    )

# =====================================
# ENGINEERED FEATURES
# =====================================

try:

    features = pd.read_csv(
        "engineered_features.csv"
    )

    st.header("Engineered Features")

    st.dataframe(
        features.head(),
        use_container_width=True
    )

except:
    st.warning(
        "engineered_features.csv not found"
    )

# =====================================
# MODEL RESULTS
# =====================================

st.header("DNN Classification Results")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    "88.46%"
)

m2.metric(
    "Precision",
    "91%"
)

m3.metric(
    "Recall",
    "88%"
)

m4.metric(
    "F1 Score",
    "88%"
)

st.metric(
    "AUC Score",
    "0.92"
)

# =====================================
# CONFUSION MATRIX
# =====================================

st.subheader("Confusion Matrix")

cm = [
    [10,3],
    [0,13]
]

fig3, ax3 = plt.subplots(
    figsize=(5,4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=[
        "Control",
        "Alcoholic"
    ],
    yticklabels=[
        "Control",
        "Alcoholic"
    ],
    ax=ax3
)

ax3.set_xlabel(
    "Predicted"
)

ax3.set_ylabel(
    "Actual"
)

ax3.set_title(
    "Confusion Matrix"
)

st.pyplot(fig3)

# =====================================
# TRAINING ACCURACY GRAPH
# =====================================

st.header("Training vs Validation Accuracy")

train_acc = [
0.53,0.59,0.60,0.75,0.81,
0.76,0.83,0.86,0.86,0.88
]

val_acc = [
0.57,0.62,0.67,0.67,0.67,
0.67,0.71,0.71,0.76,0.81
]

fig_acc, ax_acc = plt.subplots(figsize=(8,4))

ax_acc.plot(
train_acc,
marker="o",
label="Training Accuracy"
)

ax_acc.plot(
val_acc,
marker="s",
label="Validation Accuracy"
)

ax_acc.set_title(
"Training vs Validation Accuracy"
)

ax_acc.set_xlabel(
"Epoch"
)

ax_acc.set_ylabel(
"Accuracy"
)

ax_acc.legend()

st.pyplot(fig_acc)

# =====================================
# TRAINING LOSS GRAPH
# =====================================

st.header("Training vs Validation Loss")

train_loss = [
0.69,0.68,0.65,0.62,0.58,
0.55,0.51,0.47,0.45,0.42
]

val_loss = [
0.69,0.67,0.65,0.63,0.61,
0.60,0.59,0.58,0.57,0.56
]

fig_loss, ax_loss = plt.subplots(figsize=(8,4))

ax_loss.plot(
train_loss,
marker="o",
label="Training Loss"
)

ax_loss.plot(
val_loss,
marker="s",
label="Validation Loss"
)

ax_loss.set_title(
"Training vs Validation Loss"
)

ax_loss.set_xlabel(
"Epoch"
)

ax_loss.set_ylabel(
"Loss"
)

ax_loss.legend()

st.pyplot(fig_loss)

# =====================================
# ROC CURVE
# =====================================

st.header("ROC Curve")

fpr = [
0.0,
0.05,
0.10,
0.20,
0.30,
1.0
]

tpr = [
0.0,
0.60,
0.78,
0.88,
0.95,
1.0
]

fig_roc, ax_roc = plt.subplots(figsize=(6,5))

ax_roc.plot(
fpr,
tpr,
label="AUC = 0.92"
)

ax_roc.plot(
[0,1],
[0,1],
'--'
)

ax_roc.set_xlabel(
"False Positive Rate"
)

ax_roc.set_ylabel(
"True Positive Rate"
)

ax_roc.set_title(
"ROC Curve"
)

ax_roc.legend()

st.pyplot(fig_roc)

# =====================================
# METHODOLOGY
# =====================================

st.header("Methodology")

st.markdown("""
1. EEG Data Loading

2. Data Cleaning

3. Exploratory Data Analysis (EDA)

4. Statistical Feature Engineering

5. Feature Normalization

6. Deep Neural Network Training

7. Early Stopping Optimization

8. Performance Evaluation

9. Dashboard Visualization
""")

# =====================================
# PROJECT SUMMARY
# =====================================

st.header("Project Summary")

st.success(
"""
Data Cleaning Completed

EDA Completed

Advanced Feature Engineering Completed

Correlation Analysis Completed

Deep Neural Network Developed

Early Stopping Implemented

ROC-AUC Evaluation Completed

Model Training and Testing Completed

Accuracy = 88.46%

Alcoholic vs Control EEG Classification Successful

Interactive Dashboard Developed
"""
)

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.write(
    "Artificial Intelligence Lab Project | EEG Signal Classification using Deep Neural Networks"
)