# QuantChallenge 2025 Overall Rank: #11 out of 1417
https://quantchallenge.org/
 
# Research Round

**Task:**  
Given 17 columns of training data (`time`, `A-N`, `Y1`, `Y2`) and 15 columns of test data (`time`, `A-N`):

- **Goal:** Train a model on the training data and predict values (`Y1`, `Y2`) for the test data.  
- **Metric:** Achieve the highest possible \( R^2 \), defined as:  

\[
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
\]

\[
SS_{\text{res}} = \sum (y_i - \hat{y}_i)^2, \quad SS_{\text{tot}} = \sum (y_i - \bar{y})^2
\]

---

### Models Considered

- **Tree-based methods:** XGBoost, LightGBM, CatBoost.  

---

### Data Exploration

1. **Correlation**  
   - Y1 strongly correlated with predictors (\( r \approx 0.8 \)) → possible linear dependence.  
   - Y2 weaker correlation → less linear predictability.  

2. **ADF (Augmented Dickey–Fuller Test)**  
   - Tests for stationarity (constant mean, variance, autocorrelation).  
   - Both Y1 and Y2 found to be stationary.  

3. **ACF (Autocorrelation Function)**  
   - Y1: no significant autocorrelation (noise-like).  
   - Y2: spikes above confidence bounds → some MA structure.  

4. **PACF (Partial Autocorrelation Function)**  
   - Y1: no significant partial autocorrelation.  
   - Y2: significant spikes → some AR structure.  

**Interpretation:**  
- Y1: mostly noise, no strong AR/MA structure.  
- Y2: some autocorrelation, but not enough for a low-order AR/MA model.  
- Linear models likely to underfit → nonlinear models more suitable.  

---

### Why XGBoost (Nonlinear Choice)

- **Large dataset** with many columns → XGBoost can exploit interactions.  
- **Predictive performance prioritized** over interpretability.  
- **Handles nonlinear structure:** can capture feature interactions and lag effects.  
- **Comparison of boosting frameworks:**  
  - **XGBoost:** extreme gradient boosting, level-wise growth.  
  - **LightGBM:** leaf-wise growth, faster on large datasets.  
  - **CatBoost:** categorical feature handling, symmetric level-wise trees.  

---

### Testing & Evaluation Pipeline

1. **Rolling Window Testing**  
   - Train/test splits that respect time order.  

2. **Hyperparameter Optimization**  
   - Cross-validation with time-series splits.  

3. **Evaluation Metric**  
   - Compute \( R^2 \) on validation/test data.  

4. **Final Training**  
   - Retrain best model on the **entire dataset**.  

5. **Prediction**  
   - Generate predictions for Y1 and Y2 on the test set.  

### Final Approach: Blending

- Final model was a **blend of XGBoost and CatBoost**,  
- **Ridge regression** was used as a meta-learner to determine the optimal weights for combining their predictions.  
- This approach leveraged:  
  - **XGBoost:** strong baseline nonlinear learner.  
  - **CatBoost:** robust handling of categorical features.  
  - **Ridge regression:** provided a stable, regularized way to combine both, preventing overfitting and balancing contributions.
  
# Trading round
- Implemented Avellaneda-Stoikov model, parameters need adjusting in testing environment
- Developed a late-game inefficiency capture strategy, targeting predictable pricing discrepancies near market close

# Credit to Team
- https://www.linkedin.com/in/scott-yap/
- https://www.linkedin.com/in/nathan-tan-a7368b1b8/
- https://www.linkedin.com/in/aaronslchong/




