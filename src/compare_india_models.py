import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from xgboost import XGBClassifier
import joblib


def load_and_prepare_india_data(path="data/raw/crime_dataset_india.csv"):
    df = pd.read_csv(path)

    # Standard feature set used in notebook 06_india_random_forest
    features = [
        'City',
        'Time of Occurrence',
        'Victim Age',
        'Victim Gender',
        'Weapon Used',
        'Police Deployed'
    ]
    target = 'Crime Domain'

    # Clean missing values
    for col in features + [target]:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Time to hour extraction (if not numeric)
    if not np.issubdtype(df['Time of Occurrence'].dtype, np.number):
        df['Time of Occurrence'] = pd.to_datetime(
            df['Time of Occurrence'], errors='coerce'
        ).dt.hour
        df['Time of Occurrence'] = df['Time of Occurrence'].fillna(0).astype(int)

    # Label encode categorical features/target consistently
    encoders = {}
    for col in features + [target]:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, encoders


def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'report': classification_report(y_test, y_pred, zero_division=0)
    }


def main():
    X_train, X_test, y_train, y_test, encoders = load_and_prepare_india_data()

    results = []

    # 1) Existing model (baseline)
    baseline_path = os.path.join('models', 'random_forest_india.pkl')
    if os.path.exists(baseline_path):
        baseline_model = joblib.load(baseline_path)
        results.append(eval_model('existing_random_forest_india', baseline_model, X_test, y_test))
    else:
        print(f"Baseline model file not found: {baseline_path}")

    # 2) New model 1: tuned random forest (same family)
    new_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    new_rf.fit(X_train, y_train)
    results.append(eval_model('new_tuned_random_forest', new_rf, X_test, y_test))
    joblib.dump(new_rf, os.path.join('models', 'new_tuned_random_forest_india.pkl'))

    # 3) New model 2: Decision Tree
    dt_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    results.append(eval_model('decision_tree', dt_model, X_test, y_test))
    joblib.dump(dt_model, os.path.join('models', 'decision_tree_india.pkl'))

    # 4) New model 3: XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    results.append(eval_model('xgboost', xgb_model, X_test, y_test))
    joblib.dump(xgb_model, os.path.join('models', 'xgboost_india.pkl'))

    # Print comparison summary
    print('\n=== 3-Model Comparison (India Dataset) ===\n')
    metric_cols = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    summary = pd.DataFrame(results)[metric_cols]
    print(summary.to_string(index=False))
    print('\nDetailed classification report for each model:\n')
    for r in results:
        print(f"--- {r['model']} ---")
        print(r['report'])


if __name__ == '__main__':
    main()
