---

### **1. Introduction**




   - **Problem Statement**  
   
   
     Breast cancer remains a significant global health challenge. Early and accurate diagnosis is essential for improving survival rates and reducing treatment costs. This project aims to classify tumors as malignant or benign using machine learning, focusing on reducing false negatives to ensure timely medical interventions.
     
     
     
     
   - **Dataset Description** 
   
   
     The dataset used in this project contains clinical features recorded during tumor assessments, such as radius, texture, perimeter, and area. Each sample is labeled as malignant (1) or benign (0), representing the target variable.  
     
     Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic 
     
     
     
   - **Clinical Significance**  
     Misclassifying a malignant tumor as benign (false negative) can delay critical treatment, potentially leading to fatal outcomes. Thus, this study prioritizes recall for malignant tumors, ensuring no cases are missed.


---

### **2. Data Understanding & Preprocessing**

- **Dataset Overview**
  - **Dimensionality:** The dataset consists of \( 569 \) rows (samples) and \( 11 \) columns (features + target variable) before preprocessing.
  - **Features:** Clinical measurements ('radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
    'concavity', 'concavepoints', 'symmetry', 'fractaldimension', and 'Diagnosis' etc.) recorded in triplicate for each sample.
- **Feature Descriptions**
  - Radius: Mean radius of the tumor.
  - Texture: Variation in gray-scale intensity.
  - Perimeter: Tumor perimeter length.
  - Area: Tumor size.
  - Smoothness: Measure of the smoothness of the tumor's surface (local variation in radius lengths).
  - Compactness: Measure of the compactness of the tumor, calculated as \( \frac{{\text{perimeter}^2}}{{\text{area}}} - 1\).
  - Concavity: Extent of concave portions of the tumor's contour (indentations).
  - Concave Points: Number of points forming the concave portions of the tumor's contour.
  - Symmetry: Measure of the symmetry of the tumor's shape.
  - Fractal Dimension: Metric quantifying the complexity of the tumor's border (calculated as a ratio of changes in detail to changes in scale).
  - Diagnosis: Binary target variable indicating tumor type (0 = benign, 1 = malignant).
- **Class Distribution**
  - Benign: \( 63.3\% \) of samples.
  - Malignant: \( 36.7\% \) of samples.  
    The dataset shows class imbalance, which necessitated balancing techniques to ensure fair model training.
- **Data Preprocessing Steps**
  1.  Handled Outliers.
  2.  Averaged triplicate features into single features for simplicity.
  3.  Applied label encoding to the target variable.
  4.  Scaled numerical features to normalize data.
  5.  Handled imbalanced data with SMOTE.
- **Train-Test Split Details**  
  The dataset was split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.

---

### **3. Methodology**

- **Model Selection Rationale**  
  Logistic Regression was chosen for its interpretability and baseline performance. Other models like Support Vector Machine, Random Forest or Gradient Boosting could also be explored.
- **Model Architecture/Parameters**
  - Learning rate: Default.
  - Hyperparameters were tuned using grid search and randomized search.
- **Training Process**
  - Balanced the training set using SMOTE oversampling.
  - Evaluated performance on the test set using various metrics.
- **Evaluation Metrics Chosen**
  - **Precision:** To assess how many predicted positives were correct.
  - **Recall:** To minimize false negatives.
  - **F1-score:** To balance precision and recall.
  - **Confusion Matrix:** For a detailed breakdown of predictions.

---

### **4. Results Analysis**

- **Classification Metrics**
  - **Precision (malignant class):** \( 100\% \).
  - **Recall (malignant class):** \( 90\% \).
  - **F1-score (malignant class):** \( 95\% \).
- **Overall Accuracy:** \( 96\% \).

- **Confusion Matrix Interpretation:**  
  | | Predicted Benign | Predicted Malignant |  
  |---------------|------------------|---------------------|  
  | **Actual Benign** | 69 | 0 |  
  | **Actual Malignant** | 4 | 36 |
  - **True Negatives (TN):** 69 benign cases correctly classified.
  - **False Negatives (FN):** 4 malignant cases missed.
  - **True Positives (TP):** 36 malignant cases correctly identified.
  - **False Positives (FP):** 0 benign cases misclassified.
- **Class-wise Performance Analysis**  
  The model performed well in identifying malignant cases, but false negatives highlight areas for improvement.

---

### **5. Discussion**

- **Model Strengths and Limitations**

  - **Strengths:** High recall for malignant cases, zero false positives.
  - **Limitations:** Presence of 4 false negatives; further balancing and feature engineering may improve results.

  After carrying out hyperparameter tuning using GridSearch and RandomSearch, it was observed that the evaluation metrics remain the same.
  This might indicates that the model is likely well-optimized with its current hyperparameters, and further tuning using GridSearch and RandomSearch did not improve its performance. The following reasons might explain this:

1. **Model Saturation**: The model might have reached its performance limit with the given data and features.
2. **Data Limitation**: The dataset might not contain additional information that could improve the model's ability to distinguish between classes.
3. **Tuning Insensitivity**: The chosen hyperparameters may not have significantly impact the model's performance.
   - **Clinical Implications of False Negatives**  
     Misdiagnosing malignant tumors can lead to delayed treatment and worsen patient outcomes. Reducing false negatives is critical.

---

### **6. Conclusion & Future Work**

- **Summary of Achievements**
  - Achieved 96% accuracy and high 90% recall for malignant cases.
  - Successfully balanced the dataset and reduced false positives to zero.
- **Future Research Directions**
  - Explore advanced models (e.g., Random Forest, XGBoost).
  - Perform feature selection to reduce redundancy.
  - Tune hyperparameters to optimize model performance further.
  - Incorporate additional datasets to improve generalizability.
  - Develop a pipeline for real-time diagnosis in clinical settings.
  - Explore deep learning methods for feature extraction and classification.
