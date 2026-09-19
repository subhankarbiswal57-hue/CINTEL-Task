"""
Titanic Survival Prediction & Model Comparison Pipeline
Author: CINTEL Team
Objective: Predict passenger survival (Survived = 0 or 1) and evaluate/compare
multiple ML classification models with zero data leakage.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb


def extract_title(name: str) -> str:
    """Extract and categorize honorific title from passenger name."""
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if not title_search:
        return 'Rare'
    title = title_search.group(1)
    if title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title in ['Mme']:
        return 'Mrs'
    elif title in ['Mr', 'Miss', 'Mrs', 'Master']:
        return title
    else:
        return 'Rare'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain features:
    - Title: Extracted from Name (proxy for age, marital/social status)
    - FamilySize: SibSp + Parch + 1 (total travel party size)
    - IsAlone: 1 if traveling solo, 0 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Input passenger dataframe containing 'Name', 'SibSp', and 'Parch'.

    Returns
    -------
    pd.DataFrame
        Copy of input dataframe with engineered features appended.
    """
    df = df.copy()
    if 'Name' in df.columns:
        df['Title'] = df['Name'].apply(extract_title)
    else:
        df['Title'] = 'Mr'

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df


def build_preprocessor():
    """
    Construct scikit-learn ColumnTransformer to eliminate data leakage.
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most frequent imputation + OneHotEncoder
    """
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features


def parse_arguments():
    """Parse command line options for the training pipeline."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Titanic Survival Prediction & Model Evaluation Benchmark"
    )
    parser.add_argument('--data-dir', type=str, default=None, help="Directory containing train.csv and test.csv")
    parser.add_argument('--results-dir', type=str, default=None, help="Directory to output benchmark metrics and plots")
    parser.add_argument('--submission-dir', type=str, default=None, help="Directory to save generated Kaggle submission")
    parser.add_argument('--random-state', type=int, default=42, help="Seed for reproducibility across train/val split and models")
    return parser.parse_args()


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_arguments()

    # Setup paths relative to project root or use CLI arguments if supplied
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir if args.data_dir else os.path.join(base_dir, 'data')
    results_dir = args.results_dir if args.results_dir else os.path.join(base_dir, 'results')
    submission_dir = args.submission_dir if args.submission_dir else os.path.join(base_dir, 'submission')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    print("=" * 70)
    print("1. LOADING TITANIC DATASET")
    print("=" * 70)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape:  {test_df.shape}")
    print("\nTarget Variable Distribution (Survived):")
    print(train_df['Survived'].value_counts(normalize=True).round(4) * 100)

    # 2 & 3. Feature Engineering
    print("\n" + "=" * 70)
    print("2 & 3. FEATURE ENGINEERING")
    print("=" * 70)
    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = train_fe[feature_cols]
    y = train_fe['Survived']
    X_test_kaggle = test_fe[feature_cols]

    print(f"Engineered Features used: {feature_cols}")

    # 4. Train-Test Split (80/20 Stratified)
    seed = args.random_state
    print("\n" + "=" * 70)
    print(f"4. TRAIN-TEST SPLIT (80% Train, 20% Test, Stratified, random_state={seed})")
    print("=" * 70)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")

    # 5. Define Pipelines with Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor()

    models = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=seed))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=seed))
        ]),
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=seed, eval_metric='logloss'))
        ])
    }

    # 6. Cross-Validation (5-Fold Stratified on Training Set)
    print("\n" + "=" * 70)
    print("5. 5-FOLD STRATIFIED CROSS-VALIDATION (ON TRAINING SET)")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_scores[name] = scores.mean()
        print(f"{name:<22} -> Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # 7. Model Training & Evaluation on Validation Set
    print("\n" + "=" * 70)
    print("6. TEST EVALUATION (VALIDATION SET - 20%)")
    print("=" * 70)

    results_records = []
    y_preds = {}
    y_probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        y_preds[name] = y_pred
        y_probs[name] = y_prob

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_prob)

        results_records.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4),
            'ROC-AUC': round(roc, 4),
            'CV Accuracy (Mean)': round(cv_scores[name], 4)
        })

    results_df = pd.DataFrame(results_records)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))

    results_csv_path = os.path.join(results_dir, 'model_comparison.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[Saved] Model comparison table saved to: {results_csv_path}")

    # 8. Visualizations: Confusion Matrices
    print("\n" + "=" * 70)
    print("7. GENERATING CONFUSION MATRICES PLOT")
    print("=" * 70)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])

        sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
                    annot_kws={"size": 13, "weight": "bold"},
                    xticklabels=['Predicted 0 (Died)', 'Predicted 1 (Survived)'],
                    yticklabels=['Actual 0 (Died)', 'Actual 1 (Survived)'])
        axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_val, y_pred):.4f} | F1: {f1_score(y_val, y_pred):.4f}", fontsize=12)

    plt.tight_layout()
    cm_path = os.path.join(results_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Saved] Confusion matrices plot saved to: {cm_path}")

    # 9. ROC Curves
    print("\n" + "=" * 70)
    print("8. GENERATING COMBINED ROC CURVE PLOT")
    print("=" * 70)
    plt.figure(figsize=(9, 7))
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (name, y_prob), color in zip(y_probs.items(), palette):
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc = roc_auc_score(y_val, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=13, weight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] ROC curves plot saved to: {roc_path}")

    # 10. Hyperparameter Tuning for Random Forest
    print("\n" + "=" * 70, flush=True)
    print("9. HYPERPARAMETER TUNING (RANDOM FOREST via GridSearchCV)", flush=True)
    print("=" * 70, flush=True)
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [4, 6],
        'classifier__min_samples_split': [2, 5]
    }
    rf_base = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_rf_pred = best_rf.predict(X_val)
    best_rf_acc = accuracy_score(y_val, best_rf_pred)
    best_rf_f1 = f1_score(y_val, best_rf_pred)

    print(f"Best RF Parameters: {grid_search.best_params_}")
    print(f"Best RF 5-Fold CV Accuracy: {grid_search.best_score_:.4f}")
    print(f"Tuned RF Validation Accuracy: {best_rf_acc:.4f} | F1: {best_rf_f1:.4f}")

    # 11. Feature Importance (Tree-based & Logistic Regression)
    print("\n" + "=" * 70)
    print("10. EXTRACTING & VISUALIZING FEATURE IMPORTANCES")
    print("=" * 70)
    # Fit preprocessor on full train to retrieve encoded feature names
    fitted_prep = best_rf.named_steps['preprocessor']
    ohe_cols = fitted_prep.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(ohe_cols)

    rf_importances = best_rf.named_steps['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp_df.head(15), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Top 15 Most Important Features (Tuned Random Forest)', fontsize=13, weight='bold')
    plt.xlabel('Gini Importance', fontsize=11)
    plt.tight_layout()
    feat_imp_path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"[Saved] Feature importance plot saved to: {feat_imp_path}")

    # 12. Sample Passenger Inference
    print("\n" + "=" * 70)
    print("11. SAMPLE PASSENGER PREDICTION INFERENCE")
    print("=" * 70)
    sample_passengers = pd.DataFrame([
        {
            'Pclass': 1, 'Sex': 'female', 'Age': 28.0, 'SibSp': 1, 'Parch': 0,
            'Fare': 80.0, 'Embarked': 'S', 'Title': 'Mrs', 'FamilySize': 2, 'IsAlone': 0
        },
        {
            'Pclass': 3, 'Sex': 'male', 'Age': 22.0, 'SibSp': 0, 'Parch': 0,
            'Fare': 7.25, 'Embarked': 'S', 'Title': 'Mr', 'FamilySize': 1, 'IsAlone': 1
        }
    ])
    sample_preds = best_rf.predict(sample_passengers)
    sample_probs = best_rf.predict_proba(sample_passengers)[:, 1]

    for i, (pred, prob) in enumerate(zip(sample_preds, sample_probs)):
        outcome = "SURVIVED (1)" if pred == 1 else "DIED (0)"
        desc = "1st Class Female, age 28, traveling with spouse" if i == 0 else "3rd Class Male, age 22, traveling alone"
        print(f"Passenger {i+1} ({desc}):")
        print(f"  -> Prediction: {outcome} with Survival Probability = {prob * 100:.2f}%\n")

    # 13. Kaggle Test Submission Generation
    print("=" * 70)
    print("12. GENERATING KAGGLE SUBMISSION FILE")
    print("=" * 70)
    test_predictions = best_rf.predict(X_test_kaggle)
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': test_predictions
    })
    sub_path = os.path.join(submission_dir, 'submission.csv')
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission Shape: {submission_df.shape}")
    print(f"Survived value counts in submission:\n{submission_df['Survived'].value_counts()}")
    print(f"[Saved] Official Kaggle submission saved to: {sub_path}")
    print("=" * 70)
    print("PIPELINE EXECUTION COMPLETE SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
