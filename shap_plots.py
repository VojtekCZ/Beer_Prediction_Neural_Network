import shap
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression

def shap_summary_topN(shap_values, top_n=100, step=50):
    """
    Plots SHAP summary plots for top N features, split into steps.
    Example: top_n=100, step=50 → 1–50 and 51–100
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    top_n : int
        Number of top features to include
    step : int
        Number of features per plot
    """
    # Calculate mean absolute SHAP values and get top indices
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx = np.argsort(-mean_abs_shap)[:top_n]

    # Create plots for each step
    for i in range(0, top_n, step):
        sub_idx = top_idx[i:i+step]
        # Create subset of SHAP values
        shap_subset = shap.Explanation(
            values=shap_values.values[:, sub_idx],
            base_values=shap_values.base_values,
            data=shap_values.data[:, sub_idx],
            feature_names=[shap_values.feature_names[j] for j in sub_idx]
        )

        # Plot and display
        shap.summary_plot(shap_subset, max_display=len(sub_idx), show=False)
        plt.title(f"SHAP Summary Plot — Features {i+1}–{i+len(sub_idx)}")
        plt.tight_layout()
        plt.show()

def shap_barh_medians(shap_values, threshold=0.001, figsize=(20, 35)):
    """
    Plots horizontal bar plot for median(|SHAP value|).
    
    Parameters
    ----------
    shap_values : shap.Explanation | list | np.ndarray
        SHAP values object or array
    threshold : float
        Minimum median(|SHAP value|) to display feature
    figsize : tuple
        Figure size
    """
    # Extract SHAP values and feature names
    if isinstance(shap_values, list):  # e.g., classification
        shap_values_array = np.abs(shap_values[0].values)
        feature_names = shap_values[0].feature_names
    elif hasattr(shap_values, "values"):  # SHAP Explanation object
        shap_values_array = np.abs(shap_values.values)
        feature_names = shap_values.feature_names
    else:
        raise ValueError("Unsupported SHAP values format")

    # Calculate medians
    medians = np.median(shap_values_array, axis=0)

    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'median_abs_shap': medians
    })

    # Filter and sort
    df_filtered = df[df['median_abs_shap'] >= threshold]
    df_top = df_filtered.sort_values(by='median_abs_shap', ascending=False)

    # Print for debugging
    print(df_top)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(df_top['feature'][::-1], df_top['median_abs_shap'][::-1])  # largest on top
    plt.xlabel("Median(|SHAP value|)", fontsize=22)
    plt.title(f"Top features (median ≥ {threshold}) by absolute SHAP value", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def prepare_shap_df(shap_values, threshold=0.001, coef_threshold=0.0025):
    """
    Prepares DataFrame with SHAP metrics and influence directions.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    threshold : float
        Minimum median absolute SHAP value to include
    coef_threshold : float
        Threshold for determining direction significance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature metrics
    """
    shap_vals = shap_values.values
    input_data = shap_values.data
    feature_names = shap_values.feature_names

    signs = []
    coefs = []

    # Linear regression to determine direction
    for i in range(shap_vals.shape[1]):
        feature_values = input_data[:, i]
        shap_column = shap_vals[:, i]

        model = LinearRegression().fit(feature_values.reshape(-1, 1), shap_column)
        coef = model.coef_[0]
        coefs.append(coef)

        # Determine direction sign
        if coef > coef_threshold:
            signs.append('+')
        elif coef < -coef_threshold:
            signs.append('-')
        elif coef < 0:
            signs.append('?-')
        else:
            signs.append('?+')

    # Calculate median absolute values
    medians_abs = np.median(np.abs(shap_vals), axis=0)

    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'median_abs_shap': medians_abs,
        'coef': coefs,
        'sign': signs
    })

    # Filter and add colors
    df = df[df['median_abs_shap'] >= threshold].copy()
    df['color'] = df['sign'].map({
        '+': 'green', '-': 'red', '?-': 'gray', '?+': 'gray'
    }).fillna('gray')

    # Create signed median values for plotting
    df['signed_median'] = df.apply(
        lambda row: row['median_abs_shap'] if row['sign'] in ['+', '?+']
        else -row['median_abs_shap'], axis=1
    )

    return df

def plot_shap_signed(shap_values, threshold=0.001, coef_threshold=0.0025, figsize=(25,10)):
    """
    Plots bar chart with signed SHAP values, sorted by importance.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    threshold : float
        Minimum median absolute SHAP value to include
    coef_threshold : float
        Threshold for determining direction significance
    figsize : tuple
        Figure size
    """
    # Prepare DataFrame using helper function
    df = prepare_shap_df(shap_values, threshold, coef_threshold)
    df_sorted = df.sort_values(by='median_abs_shap', ascending=False)

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(df_sorted['feature'], df_sorted['signed_median'], color=df_sorted['color'])
    plt.axhline(0, color='black', linewidth=0.8)

    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Direction of Influence (± Median SHAP Value)", fontsize=16)
    plt.title(f"Features Sorted by SHAP Median ≥ {threshold}", fontsize=18)

    # Create legend
    green_patch = mpatches.Patch(color='green', label='Increases model output')
    red_patch = mpatches.Patch(color='red', label='Decreases model output')
    gray_patch = mpatches.Patch(color='gray', label='Unclear direction')
    plt.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_shap_signed_ordered(shap_values, threshold=0.001, coef_threshold=0.0025, figsize=(25,10)):
    """
    Plots bar chart with signed SHAP values, preserving original feature order.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    threshold : float
        Minimum median absolute SHAP value to include
    coef_threshold : float
        Threshold for determining direction significance
    figsize : tuple
        Figure size
    """
    # Prepare DataFrame using helper function
    df = prepare_shap_df(shap_values, threshold, coef_threshold)
    
    # Preserve original feature order
    df_ordered = df.set_index('feature')
    df_ordered = df_ordered.reindex(
        [col for col in shap_values.feature_names if col in df_ordered.index]
    ).reset_index()

    # Plot
    plt.figure(figsize=figsize)
    plt.bar(df_ordered['feature'], df_ordered['signed_median'], color=df_ordered['color'])
    plt.axhline(0, color='black', linewidth=0.8)

    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Direction of Influence (± Median SHAP Value)", fontsize=16)
    plt.title(f"Features with Median SHAP ≥ {threshold}", fontsize=18)

    # Create legend
    green_patch = mpatches.Patch(color='green', label='Increases model output')
    red_patch = mpatches.Patch(color='red', label='Decreases model output')
    gray_patch = mpatches.Patch(color='gray', label='Unclear direction')
    plt.legend(handles=[green_patch, red_patch, gray_patch], loc='upper right', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_top_interactions(shap_values, out_dir, top_n=5, interactions_per_feature=1):
    """
    Finds top features by mean(|SHAP|) and plots interaction scatter plots.
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    out_dir : str
        Output directory for saving plots
    top_n : int
        Number of top features to analyze
    interactions_per_feature : int
        Number of interaction scatter plots per feature
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_feature_indices = np.argsort(-mean_abs_shap)[:top_n]
    feature_names = shap_values.feature_names

    # Process each top feature
    for idx in top_feature_indices:
        feature_name = feature_names[idx]

        # Find potential interactions
        interaction_indices = shap.utils.potential_interactions(
            shap_values[:, idx], shap_values
        )

        print(f"\n🔍 Feature: {feature_name} may interact with: "
              f"{[feature_names[i] for i in interaction_indices[:10]]}")

        # Create scatter plots for selected interactions
        for i in range(min(interactions_per_feature, len(interaction_indices))):
            interacting_idx = interaction_indices[i]
            interacting_name = feature_names[interacting_idx]

            shap.plots.scatter(
                shap_values[:, idx],
                color=shap_values[:, interacting_idx],
                show=False
            )

            # Create safe filename
            filename = f"{feature_name}_vs_{interacting_name}.png"
            filename = filename.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(out_dir, filename)

            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"💾 Saved: {out_path}")

def plot_custom_interactions(shap_values, compounds_A, compounds_B, out_dir):
    """
    Plots scatter plots for manually specified feature pairs (A × B).
    
    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values object
    compounds_A : list of str
        List of feature names for x-axis
    compounds_B : list of str
        List of feature names for color encoding
    out_dir : str
        Output directory for saving plots
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create name to index mapping
    feature_names = shap_values.feature_names
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Process all A × B combinations
    for featA in compounds_A:
        for featB in compounds_B:
            if featA not in name_to_idx or featB not in name_to_idx:
                print(f"⚠️ Feature {featA} or {featB} not found in shap_values.feature_names")
                continue

            idxA = name_to_idx[featA]
            idxB = name_to_idx[featB]

            # Create scatter plot
            shap.plots.scatter(
                shap_values[:, idxA],
                color=shap_values[:, idxB],
                show=False
            )

            # Create safe filename
            filename = f"{featA}_vs_{featB}.png"
            filename = filename.replace("/", "_").replace("\\", "_")
            out_path = os.path.join(out_dir, filename)

            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"💾 Saved: {out_path}")