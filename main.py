# Importing necessary libraries
import keras  # For high-level neural network API
import shap  # For explainability of machine learning models
from keras.models import load_model  # For loading saved neural network models
import os  # For file system operations
from pathlib import Path  # Object-oriented filesystem paths
import joblib  # For efficient serialization of Python objects
from translate import translate_shap_features  # Custom function for translating feature names
from shap_plots import *  # Import all custom SHAP plotting functions

# Apply working directory
project_root = Path.cwd() 
os.chdir(project_root)
print(f"Set working directory to: {project_root}")

# Output directory for SHAP interaction plots
out_dir = r"img/shap_interactions/all"
os.makedirs(out_dir, exist_ok=True)

# Load the pre-trained ensemble model from a saved Keras file
model = load_model('models/ensemble_model.keras')

# Load precomputed SHAP values from a pickle file
shap_values = joblib.load("shap/shap_values.pkl")

# Extract the first output dimension (for models with multiple outputs)
shap_values = shap_values[:, :, 0]

# Translate feature names to English
translate_shap_features(shap_values, lang="unknown")
print(shap_values.feature_names)

# SHAP


# Generate SHAP summary plots for top 100 features in batches of 25
shap_summary_topN(shap_values, top_n=100, step=25)

# Create horizontal bar plot of median absolute SHAP values
shap_barh_medians(shap_values, threshold=0.001, figsize=(20,35))

# Plot signed SHAP values showing direction of feature influence
plot_shap_signed(shap_values, threshold=0.001, coef_threshold=0.0025, figsize=(25,10))

# Plot signed SHAP values while preserving original feature order
plot_shap_signed_ordered(shap_values, threshold=0.001, coef_threshold=0.0025, figsize=(25,10))

# Plot top interaction effects for the most important feature
#plot_top_interactions(shap_values, out_dir=out_dir, top_n=1, interactions_per_feature=5)

# Define specific feature pairs for custom interaction analysis
#compounds_A = ["valeric acid"]
#compounds_B = ["c_10", "real extract"]

# Generate custom interaction plots for specified feature combinations
#plot_custom_interactions(shap_values, compounds_A, compounds_B, out_dir)