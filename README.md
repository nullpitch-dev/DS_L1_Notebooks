# DS_L1_Notebooks

### [KNN 개념]
  * Number of Neighbors를 사전 정의하고 (n이라 하면)
  * Training시 Sample data간의 거리와 각 Sample data의 Class를 가지고 있다가
  * Test Data가 들어오면 그 점에서 가장 가까운 n개의 점을 찾고 그 점들 중 가장 다수인 Class로 Classify한다.  

### [Odds Ratio 개념]
  * Odds Ratio = P(success) / P(failure)  
  * Logistic Regression의 coef는 log(p / (1 - p))이므로, exp(coef)를 한 값은 Odds Ratio  

### [AUC - ROC 개념]
  * TPR(True Positive Rate) = Recall = Sensitivity = TP / (TP + FN)  
  * Specificity = TN / (FP + TN)  
  * FPR(False Positive Rate) = FP / (FP + TN) = 1 - Specificity  
---

### [annova test]
  #02[3] (MultiComparison 포함), #04[2], #05[3], #09[3], #11[2]  

### [apriori, association_rules]
  #02[4], #10[4], #12[4]  

### [chi2_contingency]
  #07[3]

### [coefficient of variation]
  #09[1]  

### [corr]
  #03[4], #04[3], #08[1], #12[2]  

### [dummy variables]  
  #04[3], #06[2~5 processing], #07[0], #09[4], #13[4], #15[8]  

### [KMeans]
  #01[3], #10[2]  

### [KNN]
  #11[4], #13[1], #14[1]  

### [KNN - predict_proba]
  #14[1]  

### [LinearRegression]
  #01[4], #05[4], #08[2], #08[5], #15[8]  

### [LinearRegression - Score (r-squred)]
  #08[5], #14[3]  

### [Linear Regression - OLS]
  #04[3], #04[4], #07[4]  

### [LogisticRegression]
  #03[6], #06[2], #06[3], #09[4], #10[3], #12[5], #13[4], #14[4]  

### [mean_squared_error, root mean_squared_error]
  #07[5], #08[3], #15[8]  

### [metrics - precision_score, ROC Curve, AUC]
  #06[4]

### [Odds Ratio]
  #06[2], #10[3]  

### [PCA, StandardScaler]
  #08[4], #08[5], #11[3], #11[4]  

### [ttest_ind]
  #01[2], #03[5], #04[1], #10[1], #12[3], #14[2], #15[6]  

### [ttest_rel]
  #13[3]

### [quantile]
  #01[1], #06[1]
  
---
### [aggregation - category values into one list item]  
  #02[0], #12[4]

### [array handling - unique values and counts for each value]
  #01[3]

### [array handling - get index with certain value or condition]
  #01[3], #03[6]

### [array handling - join array items into one string]
  #03[6]

### [bool - True/False 1/0 처리]
  * (ser >= 5) + 0
  * np.where(ser >= 5, 1, 0)
### [crosstab]
  pd.crosstab(df["season"], df["holiday"])

### [datetime, relativedelta, timedelta]
  #01[3], #01[4], #05[1], #05[2], #05[4]

### [drop_duplicate]
  #13[0]  

### [enumerate and list comprehension]
  #03[6], #04[3], #08[5], #10[2], #10[3]  

### [fillna]  
  #13[0], #15[0], #15[8]

### [frozenset: how to get elements]  
  #10[4]  

### [if condition 주의]  
  - 0 < 1 & 0 < 2 → 0 < (1 & 0) < 2 → 0 < 0 < 2 → False  
  - 0 < 1 and 0 < 2 → (0 < 1) and (0 < 2) → True  

### [isin]
  #13[0], #15[0]  

### [math.log10(), np.sqrt(), 10 ** x]
  #07[4], #07[5]

### [np.where]
  np.where(np.array([3, 5, 7, 9]) >= 7, "YES", "NO")

### [np.r_]
  np.r_[1:5, 7, 10, 15:20]
  
### [pivot_table]
  #07[3]

### [rank]
  #02[2], #12[5]  

### [Series to DataFrame transform]
  #04[3]

### [set_index]
  #13[1]

### [sort_index : DO NOT use]
  #13[0]

### [string handling - zfill]
  #01[3]  

### [string replace]
  hours.str.replace(pat = "X", repl = "")

### [to_datetime]
  * df["datetime"] = pd.to_datetime(df["datetime"])
  * df["month"] = df["datetime"].dt.month

### [value_counts]
  df["season"].value_counts(normalize = True)
