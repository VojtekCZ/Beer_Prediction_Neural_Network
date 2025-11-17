# 🍺 Beer_Prediction_Neural_Network

**Prediction of non-alcoholic beer ratings based on chemical compositions using neural networks**

This bachelor thesis focuses on predicting consumer ratings of non-alcoholic beers using deep learning models built on chemical composition data. The primary goal is to analyze relationships between chemical properties and consumer preferences, identify the most impactful features, and interpret model explainability using SHAP values.  
The results help improve understanding of how specific chemical attributes influence beer preference and can support production optimization in the beverage industry.

**Keywords:** *Convolutional Neural Network (CNN), Deep Neural Network (DNN), Keras, SHAP, Explainable AI, Sensory Evaluation*

---

## 📁 Repository Structure


---

## 🧠 Implemented Neural Network Models

| Model | Description | Purpose |
|--------|------------|---------|
| **CNN** | 1D Convolutional network | Detects local feature patterns across chemical variables |
| **DNN** | Multi-layer fully connected network | Baseline deep learning regression model |
| **DB-NN** | Dense network with Dropout & BatchNorm | Reduced overfitting, improved generalization |
| **Ensemble** | Combination of trained models | Best scoring and most stable final model |

---

## 📊 Visualization & Training Process

The following images illustrate training results and model performance:

| Image | Description |
|--------|-------------|
| `CNN_NN.png` | Training & validation curves for CNN model |
| `db_NN.png` | Loss and performance for Dropout-BatchNorm model |
| `DNN_NN.png` | Standard dense network training performance |
| `RA_ensemble.png` | Final comparison: ensemble vs standalone models |

---

## 🔍 Explainability (SHAP Analysis)

To interpret model decisions, SHAP analysis was conducted. Outputs include:

- **Global feature influence ranking**
- **Positive/negative direction of impact**
- **Sorted feature contributions**
- **Split summary for top 100 chemical attributes**

### Selected figures:

| Image | Explanation |
|--------|-------------|
| `SHAP_median_ABS.png` | Median absolute impact per feature |
| `SHAP_pos_neg_influence.png` | Influence direction distribution |
| `SHAP_pos_neg_influence_sorted.png` | Ranking of effects sorted by strength |
| `SHAP_summary_1_50.png` & `SHAP_summary_51_100.png` | Detailed feature summary plots |

---

## 🚀 Usage

1. Open `main.ipynb` to begin preprocessing and prediction.
2. Train or load models from the `/models` directory.
3. Run `shap_plots.py` to reproduce explainability charts.

---

## 🏁 Conclusion

The project demonstrates that **deep learning methods are capable of estimating beer preference based on measurable chemical factors**. Combining models into an ensemble provides superior stability and predictive performance while SHAP analysis offers **interpretable insights** into chemical drivers of consumer preference.

---

## 📜 License

To be added (MIT recommended)

---

If you find this work useful, please ⭐ star the repository.
