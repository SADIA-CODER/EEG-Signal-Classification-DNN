import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import plot_model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==================================================
# LOAD DATA
# ==================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

alcoholic_file = os.path.join(
    script_dir,
    "alcoholic_eeg_dataset.csv"
)

control_file = os.path.join(
    script_dir,
    "control_eeg_dataset.csv"
)

alcoholic = pd.read_csv(alcoholic_file)
control = pd.read_csv(control_file)

# ==================================================
# CLEAN COLUMNS
# ==================================================

alcoholic.columns = alcoholic.columns.str.lower()
control.columns = control.columns.str.lower()

# remove old label/class columns if present
for col in ["label", "class"]:

    if col in alcoholic.columns:
        alcoholic.drop(columns=[col], inplace=True)

    if col in control.columns:
        control.drop(columns=[col], inplace=True)

# ==================================================
# CREATE LABELS
# ==================================================

alcoholic["label"] = 1
control["label"] = 0

# ==================================================
# KEEP REQUIRED COLUMNS
# ==================================================

alcoholic = alcoholic[
    ["trial", "channel", "sample", "eeg_value", "label"]
]

control = control[
    ["trial", "channel", "sample", "eeg_value", "label"]
]

# ==================================================
# COMBINE DATA
# ==================================================

data = pd.concat(
    [alcoholic, control],
    ignore_index=True
)

print("Dataset Shape:", data.shape)

print("\nClass Distribution:")
print(data["label"].value_counts())

# verify both classes exist
if len(data["label"].unique()) < 2:
    print("ERROR: Only one class found")
    exit()

# ==================================================
# REMOVE MISSING VALUES
# ==================================================

data.dropna(inplace=True)

# ==================================================
# EDA
# ==================================================

plt.figure(figsize=(6,4))

sns.countplot(
    x="label",
    data=data
)

plt.title("Class Distribution")
plt.show()

# ==================================================
# EEG SIGNAL PLOT
# ==================================================

signal = data[
    data["channel"] == "FP1"
]

plt.figure(figsize=(10,4))

plt.plot(
    signal["sample"][:500],
    signal["eeg_value"][:500]
)

plt.title("FP1 EEG Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.show()

# ==================================================
# FEATURE ENGINEERING
# ==================================================

features = []

for (label, channel), group in data.groupby(
    ["label", "channel"]
):

    signal = group["eeg_value"].values

    features.append({

    "Label": label,

    "Channel": channel,

    "Mean": np.mean(signal),

    "Median": np.median(signal),

    "Std": np.std(signal),

    "Variance": np.var(signal),

    "Max": np.max(signal),

    "Min": np.min(signal),

    "RMS":
    np.sqrt(np.mean(signal**2)),

    "PeakToPeak":
    np.max(signal) - np.min(signal),

    "Energy":
    np.sum(signal**2),

    "Range":
    np.max(signal) - np.min(signal),

    "CoeffVariation":
    np.std(signal)/(abs(np.mean(signal))+1e-6),

    "Skewness":
    pd.Series(signal).skew(),

    "Kurtosis":
    pd.Series(signal).kurtosis()
})

feature_df = pd.DataFrame(features)

feature_df.to_csv(
    "engineered_features.csv",
    index=False
)

print("Engineered features saved.")

print("\nFeature Dataset Shape:")
print(feature_df.shape)

print(feature_df.head())

# ==================================================
# CORRELATION HEATMAP
# ==================================================

plt.figure(figsize=(10,8))

corr = feature_df.select_dtypes(
    include=np.number
).corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Feature Correlation Matrix"
)

plt.show()

# ==================================================
# ENCODE CHANNEL
# ==================================================

encoder = LabelEncoder()

feature_df["Channel"] = encoder.fit_transform(
    feature_df["Channel"]
)

# ==================================================
# FEATURES / TARGET
# ==================================================

X = feature_df.drop(
    "Label",
    axis=1
)

y = feature_df["Label"]

# ==================================================
# NORMALIZATION
# ==================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ==================================================
# DNN MODEL
# ==================================================

model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
        )
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])

# ==================================================
# COMPILE
# ==================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True
)

print("Model architecture saved.")

# ==================================================
# TRAIN
# ==================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)

# ==================================================
# EVALUATE
# ==================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nAccuracy =", accuracy)

# ==================================================
# PREDICT
# ==================================================

y_prob = model.predict(X_test)

y_pred = (
    y_prob > 0.5
).astype(int)

# ==================================================
# METRICS
# ==================================================

print("\nAccuracy Score:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()


plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()


fpr, tpr, _ = roc_curve(
    y_test,
    y_prob.ravel()
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.show()

print("AUC Score =", roc_auc)

# ==================================================
# CONFUSION MATRIX PLOT
# ==================================================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Control","Alcoholic"],
    yticklabels=["Control","Alcoholic"]
)

plt.title(
    "Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()

# ==================================================
# SAVE MODEL
# ==================================================

model.save(
    "EEG_DNN_Model.keras"
)

print("\nModel Saved Successfully")