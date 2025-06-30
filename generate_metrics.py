import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Load model and data
model = load_model("model/weather_forecast_model.keras")
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_val = np.load("x_val.npy")
y_val = np.load("y_val.npy")

# Predict
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

# Calculate metrics
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4)
    }

metrics = {
    "train": compute_metrics(y_train, y_train_pred),
    "validation": compute_metrics(y_val, y_val_pred)
}

# Save to file
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Metrics saved to metrics.json")
