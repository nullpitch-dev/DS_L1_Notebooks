# DS_L1_Notebooks

### [annova test]
  #02[3] (MultiComparison 포함), #04[2], #05[3]

### [apriori, association_rules]
  #02[4]

### [corr]
  #03[4], #04[3], #08[1]  

### [dummy variables]  
  #04[3], #06[2~5 processing], #15[8]  

### [KMeans]
  #01[3]  

### [LinearRegression]
  #01[4], #05[4], #08[2], #08[5], #15[8]  

### [Linear Regression - OLS]
  #04[3], #04[4]  

### [LogisticRegression]
  #03[6], #06[2], #06[3]  

### [mean_squared_error, root mean_squared_error]
  #08[3], #15[8]  

### [metrics - precision_score, ROC Curve, AUC]
  #06[4]

### [Odds Ratio]
  #06[2]  

### [PCA, StandardScaler]
  #08[4], #08[5], #11[3], #11[4]  

### [ttest_ind]
  #01[2], #03[5], #04[1], #15[6]  

### [quantile]
  #01[1], #06[1]
  
---
### [Chi2 Test 개념]
  * Tests if there is a relationship between two categorical variables  
  * The data is usually displayed in a cross-tabulation format  
    with each row representing a level (group) for one variable  
    and each column representing a level (group) for another variable  
  * Compares the observed observations to the expected observations
  * The H0: There is no relationship between variable one and variable two.

### [Odds Ratio 개념]
  * Odds Ratio = P(success) / P(failure)  
  * Logistic Regression의 coef는 log(p / (1 - p))이므로, exp(coef)를 한 값은 Odds Ratio  

### [Support, Confidence, Lift 개념]
  * 지지도  
    – Support = P(X ∩ Y)  
    – 전체 거래 중 항목 X, Y 동시 포함 거래 정도  
    – 전체 구매도 경향 파악  
  * 신뢰도  
    – Confidence = P(Y | X) = P(X ∩ Y) / P(X)  
    – 항목 X 포함 거래 중 Y 포함 확률  
    – 연관성의 정도 파악  
  * 향상도  
    – Lift = P(Y | X) / P(Y) = P(X ∩ Y) / P(X)P(Y)  
    – 항목 X 구매 시 Y 포함하는 경우와 Y가 임의 구매되는 경우의 비  

### [AUC - ROC 개념]
  * TPR(True Positive Rate) = Recall = Sensitivity = TP / (TP + FN)  
  * Specificity = TN / (FP + TN)  
  * FPR(False Positive Rate) = FP / (FP + TN) = 1 - Specificity  
---

### [aggregation - category values into one list item]  
  #02[0], #12[4]

### [array handling - unique values and counts for each value]
  #01[3]

### [array handling - get index with certain value or condition]
  #01[3], #03[6]

### [array handling - join array items into one string]
  #03[6]

### [datetime, relativedelta, timedelta]
  #01[3], #01[4], #05[1], #05[2], #05[4]

### [enumerate and list comprehension]
  #03[6], #04[3]

### [fillna]  
  #15[0], #15[8]

### [if condition 주의]  
  - 0 < 1 & 0 < 2 → 0 < (1 & 0) < 2 → 0 < 0 < 2 → False  
  - 0 < 1 and 0 < 2 → (0 < 1) and (0 < 2) → True  

### [isin]
  #13[0], #15[0]  

### [rank]
  #02[2]  

### [Series to DataFrame transform]
  #04[3]

### [string handling - zfill]
  #01[3]  

