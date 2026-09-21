import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

from services.consts.AinySchema import AinySchema


def main(args:list):
    # ==========================================
    # 1. LOAD AND SETUP TARGET
    # ==========================================

    print(AinySchema.DATA_DIR, AinySchema.DATA_DIR+'training_data.csv')
    df = pd.read_csv(AinySchema.DATA_DIR+'training_data.csv')

    # --- Define Your Target Class ---
    # Since this is a classification problem, we must create a categorical target.
    # Example: Let's classify if 'Pct_Diff' (future stock change) is positive (1) or negative (0).
    # Adjust this definition based on your specific machine learning goal.
    df['Target_Class'] = (df['Pct_Diff'] > 0).astype(int)

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    # ==========================================
    # 2. CLASS BALANCE ANALYSIS
    # ==========================================
    print("--- 📊 Target Class Balance ---")
    counts = df['Target_Class'].value_counts()
    pcts = df['Target_Class'].value_counts(normalize=True) * 100
    for idx in counts.index:
        print(f"Class {idx}: {counts[idx]} rows ({pcts[idx]:.2f}%)")

    # Save class distribution plot
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x='Target_Class', palette='viridis')
    plt.title('Distribution of Target Class')
    plt.savefig('class_distribution.png')
    plt.close()

    # ==========================================
    # 3. MISSING VALUES & DATA TYPES
    # ==========================================
    print("\n--- 🔎 Missing Data Analysis (Top 50 Columns) ---")
    missing_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df)) * 100
    }).sort_values(by='Missing Values', ascending=False)

    print(missing_info.head(50))

    # ==========================================
    # 4. FEATURE GROUPING & CORRELATION
    # ==========================================
    # Because you have 200+ features, a single heatmap will be unreadable.
    # We group a few key indicator columns to look for structural relationships.

    core_indicators = [
        'Close', 'Volume', 'bhsScore', 'GrossProfit', 'OperatingIncomeLoss',
        'NetIncomeLoss', 'Assets', 'Liabilities', 'StockholdersEquity',
        'NetCashProvidedByUsedInOperatingActivities'
    ]

    # Keep only columns that actually exist in the data to avoid errors
    valid_indicators = [col for col in core_indicators if col in df.columns]

    if valid_indicators:
        print("\n--- 📈 Computing Core Feature Correlation Matrix ---")
        corr_matrix = df[valid_indicators].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Core Financial Feature Correlation')
        plt.tight_layout()
        plt.savefig('core_correlation_matrix.png')
        plt.close()
        print("Saved 'core_correlation_matrix.png' to your directory.")

    # ==========================================
    # 5. FEATURE IMPORTANCE VIA STATISTICAL TESTING (ANOVA)
    # ==========================================
    print("\n--- 🌟 Top Financial Features Predicting Target_Class (ANOVA) ---")
    # Find all numeric columns (excluding metadata/targets)
    exclude_cols = ['Target_Class', 'Pct_Diff', 'Close_Target', 'Date', 'Ticker', 'TargetDate', 'Date_Target', 'Date_y']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    features_to_test = [col for col in numeric_cols if col not in exclude_cols]

    anova_results = []
    for col in features_to_test:
        # Drop NaNs just for the statistical check
        clean_sub = df[[col, 'Target_Class']].dropna()

        # Check if we have data points in both classes
        if len(clean_sub['Target_Class'].unique()) == 2:
            group0 = clean_sub[clean_sub['Target_Class'] == 0][col]
            group1 = clean_sub[clean_sub['Target_Class'] == 1][col]

            if len(group0) > 5 and len(group1) > 5:
                f_stat, p_val = f_one_way_score = f_oneway(group0, group1)
                anova_results.append({'Feature': col, 'F-Statistic': f_stat, 'p-value': p_val})

    anova_df = pd.DataFrame(anova_results).sort_values(by='F-Statistic', ascending=False)
    print(anova_df.head(10).to_string(index=False))

    # ==========================================
    # 6. HOW TO HIGHLIGHT KEY ANOMALIES
    # ==========================================
    print("\n--- ⚠️ Missing Value Alert for Crucial Fundamentals ---")
    essential_fundamentals = ['GrossProfit', 'Assets', 'Liabilities']
    for feat in essential_fundamentals:
        if feat in df.columns:
            missing_pct = (df[feat].isnull().sum() / len(df)) * 100
            if missing_pct > 30:
                print(f"Warning: {feat} is missing {missing_pct:.1f}% of its data. "
                    f"Consider filling with 0 or dropping rows before training.")



if __name__ == "__main__":
    main(sys.argv[1:])
