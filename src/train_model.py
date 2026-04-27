import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)


def save_model(model, path):
    """Save model and scaler to disk."""
    joblib.dump(model, path)


def load_data(csv_path):
    """Load and prepare crime data."""
    df = pd.read_csv(csv_path)
    return df


def create_features(df):
    """
    Create features for hotspot prediction.
    Aggregates crime counts per H3 cell and creates labels.
    """
    # Crime count per H3 cell
    crime_counts = (
        df.groupby("h3_cell")
          .size()
          .reset_index(name="crime_count")
    )
    
    # Hotspot threshold: top 20% most crime-dense cells
    threshold = crime_counts["crime_count"].quantile(0.80)
    crime_counts["hotspot_label"] = (
        crime_counts["crime_count"] >= threshold
    ).astype(int)
    
    # Merge labels back to main dataframe
    df = df.merge(
        crime_counts[["h3_cell", "hotspot_label"]],
        on="h3_cell",
        how="left"
    )
    
    return df, crime_counts


def prepare_training_data(df, feature_cols=None):
    """
    Prepare X and y for model training.
    
    Args:
        df: DataFrame with hotspot_label column
        feature_cols: List of feature column names. If None, uses default set.
    
    Returns:
        X, y: Feature matrix and target vector
    """
    if feature_cols is None:
        # Default features - adjust based on your data
        feature_cols = [
            "crime_count",
            "hour",
            "day_of_week",
            "month"
        ]
    
    # Filter to cells that have features
    available_cols = [c for c in feature_cols if c in df.columns]
    
    if not available_cols:
        # Fallback: use crime_count as sole feature
        available_cols = ["crime_count"]
    
    X = df[available_cols].fillna(0)
    y = df["hotspot_label"]
    
    return X, y


def train_hotspot_model(X, y, model_type="random_forest", **kwargs):
    """
    Train a hotspot prediction model.
    
    Args:
        X: Feature matrix
        y: Target vector (0 = not hotspot, 1 = hotspot)
        model_type: "random_forest" or "logistic"
        **kwargs: Additional model parameters
    
    Returns:
        model: Trained model
        scaler: Fitted StandardScaler
        metrics: Dictionary of evaluation metrics
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 10),
            random_state=kwargs.get("random_state", 42),
            n_jobs=-1
        )
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            random_state=kwargs.get("random_state", 42)
        )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }
    
    print(f"Model: {model_type}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    
    return model, scaler, metrics


def train_from_csv(csv_path, output_model_path="models/hotspot_model.joblib"):
    """
    End-to-end training pipeline from CSV file.
    
    Args:
        csv_path: Path to cleaned crime data CSV
        output_model_path: Path to save trained model
    
    Returns:
        model, scaler, metrics
    """
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    
    print("Creating features and labels...")
    df, crime_counts = create_features(df)
    
    print("Preparing training data...")
    X, y = prepare_training_data(crime_counts)
    
    print("Training model...")
    model, scaler, metrics = train_hotspot_model(X, y)
    
    # Save model and scaler together
    save_model({"model": model, "scaler": scaler}, output_model_path)
    print(f"Model saved to {output_model_path}")
    
    return model, scaler, metrics


if __name__ == "__main__":
    # Example usage
    train_from_csv("data/processed/chicago_cleaned.csv")
