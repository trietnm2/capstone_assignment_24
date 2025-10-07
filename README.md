# Click-Through Rate (CTR) Prediction Analysis
### Predicting Online Advertising Engagement Using Machine Learning

**Author:** Nguyen Minh Triet  
**Date:** October 2, 2025  
**Project Type:** Capstone Project

---

## Executive Summary

This project tackles the challenge of predicting click-through rates (CTR) for online advertisements using machine learning. By analyzing 50,000 advertising records with 24 features, I developed and optimized predictive models achieving **0.6706 ROC-AUC** with tuned XGBoost.

**Key Achievements:**
- Identified critical factors driving ad clicks through comprehensive exploratory data analysis
- Developed and optimized 6 machine learning models: XGBoost (0.6706 ROC-AUC), Random Forest (0.6657), Neural Network (0.6579), Logistic Regression (0.6437), LightGBM (0.6727) and Naive Bayes
- Implemented three feature importance methods (tree-based, permutation, SHAP) identifying 6 consensus features: C21, C14, C17, C20, C18, C19, device_conn_type
- Applied proper chronological train-test split and one-hot encoding for robust validation
- Delivered data-driven business insight sections with quantified ROI projections

**Business Impact:** Implementing the recommendations could improve CTR by 18-36%, device-banner optimization (25-40% CTR improvement), and tiered targeting (240% ROI for Premium tier).

---

## The Challenge

With only 16.93% of ads resulting in clicks, advertisers face intense competition for user attention. Traditional approaches struggle to account for the complex interplay of timing, device context, ad placement, and user characteristics. Machine learning offers a solution by identifying patterns invisible to human analysis and making real-time predictions at scale.

---

## Research Question

**What factors most significantly influence the likelihood of a user clicking on an online advertisement, and how accurately can we predict click-through rates using machine learning?**

### Sub-Questions Addressed:
1. How do temporal patterns (hour, day, time-of-day) affect click rates?
2. What role do device characteristics and connection types play in engagement?
3. Which ad placements and positions generate the highest click-through rates?
4. Can we build predictive models that outperform random chance by a meaningful margin?
5. What practical actions can advertisers take based on the findings?

---

## Data Sources

### Dataset Overview
- **Source:** Online advertising click-through dataset
- **Size:** 50,000 records (observations)
- **Features:** 24 variables including temporal, device, and contextual information
- **Target Variable:** Binary click indicator (0 = no click, 1 = click)
- **Time Period:** Multiple days of advertising data across various contexts
- **Data Quality:** Clean dataset with no missing values or duplicates

### Feature Categories

#### Temporal Features
- **hour:** Timestamp of ad impression
- **Derived features:** Hour of day, day of week, weekend indicator, time period

#### Device Information
- **device_type:** Type of device (mobile, tablet, desktop)
- **device_conn_type:** Connection type (wifi, cellular, etc.)
- **device_id, device_ip, device_model:** Device identifiers and specifications

#### Ad Context
- **banner_pos:** Position of the banner advertisement
- **site_id, site_domain, site_category:** Website where ad appeared
- **app_id, app_domain, app_category:** Mobile app where ad appeared

#### Anonymous Categorical Variables
- **C1, C14-C21:** Anonymized categorical features with varying cardinality
- These features often capture user segments, behavioral patterns, or proprietary targeting categories

---

## Methodology

The analysis follows a comprehensive data science workflow:

### 1. Data Loading and Initial Exploration
- Loaded 50,000 advertising records
- Examined data structure, types, and distributions
- Assessed data quality (missing values, duplicates, outliers)
- Generated statistical summaries and descriptive statistics

### 2. Data Cleaning and Preprocessing
- Verified no missing values or duplicates
- Converted timestamp data to datetime format
- Handled high-cardinality features appropriately
- Prepared dataset for analysis and modeling

### 3. Exploratory Data Analysis (EDA)
- **Target Variable Analysis:** Examined class distribution and imbalance (16.93% click rate)
- **Temporal Analysis:** Investigated hour-of-day, day-of-week, and time-period patterns
  - Identified Hour 1:00 as peak performance (19.60% CTR, 116 performance index)
  - Found low-performing hours 20:00 and 22:00 (15.17% and 15.00% CTR)
- **Categorical Analysis:** Explored device types, banner positions, site/app categories
  - Device-banner combinations: 50.00% max CTR vs 0.00% min CTR
  - Connection-device pairs: 21.50% CTR for top combination
- **Correlation Analysis:** Identified relationships between features and target
- **Outlier Detection:** Used IQR method to identify and understand outliers
- **Interactive Visualizations:** Created dynamic plots for deeper insights

### 4. Feature Engineering
- Applied **chronological train-test split** (80/20) to prevent temporal data leakage
- Extracted hour of day, day of week from timestamps
- Created weekend indicator and time period categories
- Used **one-hot encoding** for categorical variables (replacing label encoding)
- Generated 52 features after encoding from 24 original features

### 5. Feature Importance Analysis
Implemented three complementary methods for robust feature selection:

#### Tree-Based Importance
- Measures feature contribution to node purity in Random Forest
- Top features: C21, C14, C17

#### Permutation Importance
- Measures performance drop when feature values are shuffled
- Evaluates real predictive power on test set
- Top features: C21, C20, C18

#### SHAP (SHapley Additive exPlanations)
- Game-theory based approach for model interpretability
- TreeExplainer on 1,000-sample subset
- Created summary plots and beeswarm visualizations
- Top features: C21, C14, C17

#### Consensus Features
Six features appeared in top 10 of ALL three methods:
- **C21, C14, C20, C18, C19, device_conn_type**

### 6. Machine Learning Modeling

#### Models Developed
1. **Random Forest Classifier**
   - Baseline: 100 trees, default parameters
   - Optimized: RandomizedSearchCV with 50 iterations
   
2. **Logistic Regression**
   - Baseline: L2 regularization
   
3. **XGBoost Classifier**
   - Optimized: RandomizedSearchCV with 50 iterations
   - Best params: learning_rate=0.033, max_depth=5, n_estimators=185
   
4. **LightGBM Classifier**
   - Gradient boosting framework for efficiency
   
5. **Gaussian Naive Bayes**
   - Probabilistic baseline model
   
6. **Neural Network (Deep Learning)**
   - Architecture: 128 -> 64 -> 32 -> 1 with batch normalization and dropout
   - 18,049 parameters, early stopping, 20 epochs

#### Model Optimization
- **RandomizedSearchCV:** XGBoost and Random Forest with 50 iterations, 3-fold CV
- **Cross-Validation:** 3-fold CV to prevent overfitting
- **Evaluation Metric:** ROC-AUC (handles class imbalance effectively)

### 7. Model Evaluation
- **Confusion Matrices:** Analyzed true/false positives and negatives for all models
- **ROC Curves:** Visualized trade-offs between sensitivity and specificity
- **Performance Comparison:** Created comprehensive comparison table across 6 models
- **Best Model Selection:** XGBoost (0.6706 ROC-AUC after tuning)

### 8. Business Insights and Recommendations
Delivered three comprehensive analysis sections:

#### Section 1: Time-Based Bid Optimization
- Hourly CTR analysis with performance indexing
- Identified high-performing hours for bid increases
- Quantified 9.0% potential CTR improvement

#### Section 2: Device and Placement Optimization
- Device-banner combination analysis
- Connection-device performance heatmaps
- Top 5 and bottom 5 combinations identified

#### Section 3: Predictive Targeting Strategy
- ML-based tiered segmentation (Premium: 240% ROI, Avoid: -100% ROI)
- Volume vs performance trade-off visualization
- Budget reallocation recommendations

---

## Results

### Model Performance

| Model | ROC-AUC | Key Characteristics |
|-------|---------|---------------------|
| **Optimized XGBoost** | **0.6706** | Best performer, tuned with RandomizedSearchCV |
| XGBoost | 0.6690 | XGBoost with imbalance |
| Optimized Random Forest | 0.6657 | RandomizedSearchCV tuning |
| Baseline Random Forest | 0.6626 | Strong baseline with default parameters |
| Neural Network | 0.6579 | 128->64->32->1 architecture, 18,049 parameters |
| Logistic Regression | 0.6437 | GridSearchCV with L1 penalty, C=100 |
| **LightGBM** | **0.6727** | Gradient boosting alternative |
| Naive Bayes | 0.5069 | Probabilistic baseline |

**Best Model:** Optimized XGBoost with **0.6706 ROC-AUC** and LightGBM with **0.6727 ROC-AUC**

**XGBoost Best Hyperparameters:**
- learning_rate: 0.033
- max_depth: 5
- n_estimators: 185
- colsample_bytree: 0.902
- gamma: 0.114
- min_child_weight: 3
- subsample: 0.952

**Random Forest Best Hyperparameters:**
- n_estimators: 121
- max_depth: 10
- min_samples_leaf: 7
- min_samples_split: 10
- max_features: sqrt

This performance significantly outperforms random guessing (50% ROC-AUC) and provides meaningful predictive power for ad targeting optimization.


### Key Findings from Data Analysis

#### 1. Click-Through Rate Baseline
- **Overall CTR:** 16.93% (8,467 clicks / 50,000 impressions)
- **Class Imbalance:** 4.9:1 ratio (no-click to click)
- **Implication:** Predicting clicks is challenging due to class imbalance

#### 2. Temporal Patterns - Validated with Performance Index
- **Hour-of-Day Effect:** Clear variations throughout the day
- **Peak Performance:** Hour 1:00 shows 19.60% CTR (Performance Index: 116)
- **Low Performance:** Hours 20:00 and 22:00 show 15.17% and 15.00% CTR (Performance Index: 89)
- **Time Period Impact:** Average CTR baseline is 16.93%
- **Actionable Insight:** Shifting 30% of budget from low to high-performing hours could increase overall CTR by 9.0%

**Bid Optimization Recommendations:**
- Increase bids by 20-30% for Hour 1:00
- Reduce bids by 20-30% for Hours 20:00 and 22:00

#### 3. Device and Banner Position Insights - Extreme Variation
**Top 5 Device-Banner Combinations:**
1. Device 1.0 + Banner Pos 3.0: **50.00% CTR** (2 impressions)
2. Device 1.0 + Banner Pos 4.0: **37.50% CTR** (8 impressions)
3. Device 4.0 + Banner Pos 7.0: **35.29% CTR** (51 impressions - statistically meaningful)
4. Device 5.0 + Banner Pos 7.0: **25.00% CTR** (8 impressions)
5. Device 0.0 + Banner Pos 0.0: **21.50% CTR** (2,730 impressions - high volume)

**Bottom 5 Device-Banner Combinations (Avoid):**
1. Device 4.0 + Banner Pos 0.0: **0.00% CTR** (1 impression)
2. Device 4.0 + Banner Pos 1.0: **8.23% CTR** (875 impressions - significant volume)
3. Device 5.0 + Banner Pos 1.0: **9.64% CTR** (166 impressions)
4. Device 1.0 + Banner Pos 5.0: **10.00% CTR** (10 impressions)
5. Device 1.0 + Banner Pos 2.0: **11.11% CTR** (9 impressions)

**Top Connection-Device Combinations:**
- Device 0.0 + Connection 0.0: **21.50% CTR**
- Device 1.0 + Connection 0.0: **17.96% CTR**
- Device 1.0 + Connection 2.0: **13.73% CTR**

**Actionable Insight:** Increase bids by 30-50% for top combinations, reduce/pause bottom performers for 25-40% CTR improvement

#### 4. Predictive Targeting Strategy - Tiered Approach
Using ML model scores to segment audiences into 4 tiers:

**Premium (High Intent):**
- Volume: 8,321 impressions (83.2% of total)
- Actual CTR: **17.80%**
- Estimated ROI: **256%**
- Action: Increase bids by 50-100%

**Standard:**
- Volume: 964 impressions (9.6% of total)
- Actual CTR: **5.82%**
- Estimated ROI: **16%**
- Action: Maintain current bids

**Low Priority:**
- Volume: 678 impressions (6.8% of total)
- Actual CTR: **3.10%**
- Estimated ROI: **-38%**
- Action: Reduce bids by 50-70%

**Avoid:**
- Volume: 37 impressions (0.4% of total)
- Actual CTR: **0.00%**
- Estimated ROI: **-100%**
- Action: Exclude entirely

**Actionable Insight:** Reallocate budget from Avoid tier to Premium tier for maximum ROI

---

## Key Files
- **main.ipynb** - Jupyter Notebook with complete analysis workflow
- **requirements.txt** - Python dependencies for reproducibility
- **README.md** - Comprehensive project documentation

