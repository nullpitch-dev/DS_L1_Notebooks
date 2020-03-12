# DS_L1_Notebooks

### [annova test]
  #02[3] (MultiComparison 포함), #05[3]

### [apriori, association_rules]
  #02[4]

### [datetime]
  #01[3], #01[4], #05[1], #05[2], #05[4]

### [KMeans]
  #01[3]

### [LinearRegression]
  #05[4]

### [LogisticRegression]
  #03[6]

### [ttest_ind]
  #01[2], #03[5]

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

### [if condition 주의]  
  - 0 < 1 & 0 < 2 → 0 < (1 & 0) < 2 → 0 < 0 < 2 → False  
  - 0 < 1 and 0 < 2 → (0 < 1) and (0 < 2) → True
