import pandas as pd
import numpy as np
import math
######### ######### ######### ######### ######### ######### ######### ##########
# Ex02 Data ####################################################################
url_pur = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_purchase_log.csv'
url_cus = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_customer_mst.csv'
url_pro = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_product_mst.csv'
data_pur = pd.read_csv(url_pur)
data_cus = pd.read_csv(url_cus)
data_pro = pd.read_csv(url_pro)

# Ex03 Data ####################################################################
url_antibio = "https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/Antibiotic_70K_patinets.csv"
data_antibio = pd.read_csv(url_antibio)

# Ex04 Data ####################################################################
url_corolla_1 = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/corolla_1.csv'
url_corolla_2 = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/corolla_2.csv'
data_corolla1 = pd.read_csv(url_corolla_1)
data_corolla2 = pd.read_csv(url_corolla_2)

# Ex05 Data ####################################################################
url_ecommerce = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ecommerce_transaction.csv'
data_ecommerce = pd.read_csv(url_ecommerce)

################################################################################
################################################################################
################################################################################


# dummy variables ##############################################################
df_dummy = pd.get_dummies(df_original, columns=df_original.columns[_from_:_to_],
                          drop_first=True)


# elif in lambda ###############################################################
df_elif = df_elif.assign(new_col=df_elif["Col_A"].apply(
                            lambda x: "value_1" if x <= condi_1 else (
                                      "value_2" if x >= condi_2 else "middle")))


# enumerate and list comprehension #############################################
# 결과 중 특정 조건을 만족하는 변수명 찾기
idx = [i for i, values in enumerate(model_result) if values < 0]
vars_found = ", ".join([X_cols[i] for i in idx])


# Merge ########################################################################
df_merge = pd.merge(data_left, data_right[["Col_A", "Col_B"]], how="left",
                    left_on="Key_left", right_on="Key_right")


# Pivot_table for Association_rules ############################################
pivot = df_pivot.pivot_table(index="index_col", columns="column_col",
                             aggfunc="size", fill_value=0)
pivot = (pivot >= 1) + 0  # 1 이상은 1로 0은 0으로 만드는 방법


# Rank  ########################################################################
df_rank = df_rank.assign(rank=df_rank.rank(ascending=False, method="min"))


# to_datetime  #################################################################
df_dt = pd.DataFrame(['2017-01-07', '2019-05-30', '2020-10-05'], columns=['date'])
df_dt['date'] = pd.to_datetime(df_dt['date'])
df_dt = df_dt.assign(plus7year=df_dt['date'].apply(lambda x: pd.to_datetime(
                                                    str(x.year + 7) +
                                                    str(x.month).zfill(2) +
                                                    str(x.day).zfill(2))))
df_dt['date'].dt.dayofweek  # 요일 (월요일이 0)


#####################################################################################################
#####################################################################################################
#####################################################################################################


# ANOVA #####################################################################################################
# ANOVA Test 개념
# 분산을 고려했을 때 표본들의 평균이 같다고 할 수 있는지 검증
# H0 : 표본집단들의 모집단 평균이 같다
# [COMPARISON T-TEST vs. ANOVA]
#     T-test is a hypothesis test that is used to compare the means of two populations.
#     ANOVA is a statistical technique that is used to compare the means of more than two populations.
#     Test statistic:
#       * T-test : (x ̄-µ)/(s/√n)
#       * ANOVA : Between Sample Variance/Within Sample Variance


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

result = ols(formula="dependent)var ~ C(independent_var1) + independent_var2", data=df_anova).fit()
anova_table = anova_lm(result)
print(anova_table)
###########################################################################################################


# MultiComparison ########################################################################################
# 사후검정 #######################################
from statsmodels.stats.multicomp import MultiComparison

comparison = MultiComparison(data=df_mc["dependent_var"], groups=df_mc["independent_var"])
print(comparison.tukeyhsd())

"""
    Multiple Comparison of Means - Tukey HSD, FWER=0.05     
============================================================
group1 group2  meandiff  p-adj    lower      upper    reject
------------------------------------------------------------
    G1     G2  8461.1735 0.001   4117.8207 12804.5262   True
    G1     G3   717.8831   0.9  -3454.2396  4890.0058  False
    G2     G3 -7743.2904 0.001 -12467.8902 -3018.6906   True
------------------------------------------------------------
# meandiff = group2_mean - group1_mean
# reject = True : mean is different
"""
###########################################################################################################


# Association Rule #####################################################################################
# [Support, Confidence, Lift 개념]
#   지지도
#     – Support = P(X ∩ Y)
#     – 전체 거래 중 항목 X, Y 동시 포함 거래 정도
#     – 전체 구매도 경향 파악
#   신뢰도
#     – Confidence = P(Y | X) = P(X ∩ Y) / P(X)
#     – 항목 X 포함 거래 중 Y 포함 확률
#     – 연관성의 정도 파악
#   향상도
#     – Lift = P(Y | X) / P(Y) = P(X ∩ Y) / P(X)P(Y)
#     – 항목 X 구매 시 Y 포함하는 경우와 Y가 임의 구매되는 경우의 비

from mlxtend.frequent_patterns import apriori, association_rules

freq_items = apriori(pivot, min_support=0.001, use_colnames=True) # Pivot_table 참고
asso_rules = association_rules(freq_items, metric="confidence", min_threshold=0.001)

asso_rules = asso_rules.assign(check=asso_rules["antecedents"].apply(lambda x: "XXX" in x)) # XXX가 포함된 항목 check
asso_rules = asso_rules[asso_rules["check"]].sort_values(by="lift", ascending=False)
###########################################################################################################


# Correlation ##########################################################################################
df_corr_result = df_corr[['var1', 'var2']].corr(method='pearson')
###########################################################################################################


# Liner Regression ##########################################################################################
from statsmodels.api import add_constant, OLS

train_X = add_constant(train_X)
model = OLS(train_y, train_X)
ols_result = model.fit()

ols_result.pvalues # p-value가 필요하면
ols_result.tvalues # t-value가 필요하면
ols_result.rsquared # R2가 필요하면
ols_result.rsquared_adj # Adjusted R2가 필요하면

# predict
test_X = pd.DataFrame([[1, 2, 3, 4, 5]], columns=['a', 'b', 'c', 'd', 'e'])
pred = ols_result.predict(test_X)
###########################################################################################################


# Logistic Regression ##########################################################################
from sklearn.linear_model import LogisticRegression

train_X = df_lr['dep_A', 'dep_B', 'dep_C', 'dep_D', 'dep_E']
train_y = df_lr['indep']

lr = LogisticRegression(C=100000, random_state=1234, penalty='l2', solver='newton-cg')
model = lr.fit(train_X, train_y)
###########################################################################################################


# T-Test ################################################################################################
# H0 : 두 집단의 평균이 같다 (다르다고 할 수 없다)
# [COMPARISON T-TEST vs. ANOVA]
#     T-test is a hypothesis test that is used to compare the means of two populations.
#     ANOVA is a statistical technique that is used to compare the means of more than two populations.
#     Test statistic:
#       * T-test : (x ̄-µ)/(s/√n)
#       * ANOVA : Between Sample Variance/Within Sample Variance
from scipy.stats import ttest_ind

p_val, t_val = ttest_ind(cd_y['high_p'], cd_n['high_p'])
###########################################################################################################


# //////// ################################################################################################
###########################################################################################################
