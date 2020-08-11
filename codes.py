import pandas as pd
import numpy as np
import math


### concat #####################################################################
df_concat = pd.concat([df1, df2])


### dropna #####################################################################
df_dropna = df_original.dropna(subset=['col1', 'col2'])


### drop_duplicate #############################################################
df_dup = df_dup.sort_values(by=['col1', 'col2', 'col3'],
                            ascending=['True', 'True', 'True'])
df_unique = df_dup.drop_duplicates(subset=['col1', 'col2'], keep='last')


### dummy variables ############################################################
df_dummy = pd.get_dummies(df_original, columns=df_original.columns[_from_:_to_],
                          drop_first=True)


### elif in lambda #############################################################
df_elif = df_elif.assign(new_col=df_elif["Col_A"].apply(
                            lambda x: "value_1" if x <= condi_1 else (
                                      "value_2" if x >= condi_2 else "middle")))


### enumerate and list comprehension ###########################################
# 결과 중 특정 조건을 만족하는 변수명 찾기
idx = [i for i, values in enumerate(model_result) if values < 0]
vars_found = ", ".join([X_cols[i] for i in idx])


### fillna #####################################################################
df[['col1', 'col2', 'col3']] = df[['col1', 'col2', 'col3']].fillna(0)


### groupby ####################################################################
groupby('col').agg({'col1': 'fun1', 'col2': 'fun2'})
# function 종류 : count, first, last, min, max, count, nunique 등
# 중복 제거된 항목 추출
df_groupby = df_orig.groupby('col_by').agg({'col_val': lambda x: set(list(x))})


### in #########################################################################
# list에 특정 문자열이 있는지 확인
df = df.assign(check=df.col.apply(lambda x: 'Y' if ('A' in x) & ('B' in x)
                                                else 'N'))


### log10 계산###################################################################
series_log = df_log['Col_A'].apply(lambda x: math.log10(x))
series_revert = series_log.apply(lambda x: 10 ** x)


### Merge ######################################################################
df_merge = pd.merge(data_left, data_right[["Col_A", "Col_B"]], how="left",
                    left_on="Key_left", right_on="Key_right")
# key column이 없고 index 기준으로 merge 할 때:
df_merge = pd.merge(data_left, data_right[['Col_A']], how='left',
                     left_index=True, right_index=True)


### Pivot_table for Association_rules ##########################################
pivot = df_pivot.pivot_table(index="index_col", columns="column_col",
                             aggfunc="size", fill_value=0)
pivot = (pivot >= 1) + 0  # 1 이상은 1로 0은 0으로 만드는 방법


### Rank  ######################################################################
df_rank = df_rank.assign(rank=df_rank['col'].rank(ascending=False, method="min"))
# method 의미 (1등이 1명, 2등이 2명, 3등이 1명이라면)
# dense : 1, 2, 2, 3
# min   : 1, 2, 2, 4
# max   : 1. 3, 3, 4


### rename #####################################################################
df_rename = df_rename.rename(columns={'col_a_before': 'col_a_after',
                                      'col_b_before': 'col_b_after'})


### set_index ##################################################################
df_reset_index = df_original.reset_index()
df_restored_index = df_reset_index.set_index(keys='index')


### to_datetime  ###############################################################
df_dt = pd.DataFrame(['2017-01-07', '2019-05-30', '2020-10-05'],
                     columns=['date'])
df_dt['date'] = pd.to_datetime(df_dt['date'])
df_dt = df_dt.assign(plus7year=df_dt['date'].apply(lambda x: pd.to_datetime(
                                                    str(x.year + 7) +
                                                    str(x.month).zfill(2) +
                                                    str(x.day).zfill(2))))
df_dt['date'].dt.dayofweek  # 요일 (월요일이 0)

### timedelta ##################################################################
df_dt = df_dt.assign(gap=(df_dt['time2'] - df_dt['time1']).dt.days * 24 * 3600 +
                         (df_dt['time2'] - df_dt['time1']).dt.seconds)
df_dt = df_dt.assign(pay=df_dt.apply(lambda x:
                             'type1' if x['gap'] >= 3600 else 'type2', axis=1))


### Top n value의 index 찾기 ####################################################
n = 5
array_original = [5, 3, 7, 1, 2, 9, 6]
array_sorted = array_original.copy()
array_sorted.sort() # [1, 2, 3, 5, 6, 7, 9]
top_n_value = array_sorted[-n] # top_n_value = 3

top_n_idx = [i for i, v in enumerate(array_original) if v >= top_n_value]
################################################################################


################################################################################
################################################################################
################################################################################


### ANOVA ######################################################################
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

result = ols(formula="dependent)var ~ C(independent_var1) + independent_var2",
             data=df_anova).fit()
anova_table = anova_lm(result)
print(anova_table)

# MultiComparison, 사후검정#######################################################
from statsmodels.stats.multicomp import MultiComparison

comparison = MultiComparison(data=df_mc["dependent_var"],
                             groups=df_mc["independent_var"])
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
################################################################################


### Association Rule ###########################################################
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

# Pivot_table 참고
freq_items = apriori(pivot, min_support=0.001, use_colnames=True)
asso_rules = association_rules(freq_items, metric="confidence",
                               min_threshold=0.001)

# XXX가 포함된 항목 check
asso_rules = asso_rules.assign(check=asso_rules["antecedents"].apply(
                                                          lambda x: "XXX" in x)) 
asso_rules = asso_rules[asso_rules["check"]].sort_values(by="lift",
                                                         ascending=False)

# antecedents 또는 consequents 개수로 filtering
asso_rules = asso_rules.assign(ant_len=asso_rules['antecedents'].apply(
                                                              lambda x: len(x)))
asso_rules = asso_rules[asso_rules['ant_len'] == 1]

# lift나 confidence 등 특정 field로 sorting해서 antecedents나 consequents 찾기
conseq = list(
  asso_rules[asso_rules['antecedents'] == {'ANTECEDENT'}].sort_values(
    by='lift', ascending=True).iloc[0, 1])[0]
################################################################################

### CChi2 Test #################################################################
#  * Tests if there is a relationship between two categorical variables.
#  * The data is usually displayed in a cross-tabulation format
#    with each row representing a level (group) for one variable
#    and each column representing a level (group) for another variable
#  * Compares the observed observations to the expected observations
#  * The H0: There is no relationship between variable one and variable two.

from scipy.stats import chi2_contingency

chi2, p_val, dof, expected = chi2_contingency(pivot)
################################################################################


### Correlation ################################################################
df_corr_result = df_corr[['var1', 'var2']].corr(method='pearson')
################################################################################


### K-Means ####################################################################
from sklearn.cluster import KMeans

cluster = KMeans(n_clusters=7, n_init=1234, random_state=1234).fit(df_kmeans)
cluster.cluster_centers_
cluster.labels_

# 특정 inddex의 label 찾기
label = cluster.labels_[np.where(df_kmeans.index == 1234)[0][0]]
idx_same_label = [i for i, val in enumerate(cluster.labels_) if val == label]
val_same_label = df_kmeans[idx_same_label]
################################################################################


### KNN ########################################################################
# Number of Neighbors를 사전 정의하고 (n이라 하면)
# Training시 Sample data간의 거리와 각 Sample data의 Class를 가지고 있다가
# Test Data가 들어오면 그 점에서 가장 가까운 n개의 점을 찾고
# 그 점들 중 가장 다수인 Class로 Classify한다.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=NO).fit(train_X, train_y)
pred = knn.predict(test_X)
pred = pd.DataFrame(pred)

# Accuracy 계산
result = pd.merge(test_y.reset_index(), pred, left_index=True, right_index=True)
result = result.rename(columns={'XX': 'fact', 'YY': 'esti'})
result = result.assign(accu=(result['fact'] == result['esti']) * 1)
accuracy = result['accu'].sum() / result['accu'].count()
################################################################################


### Linear Regression (OSL) ####################################################
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
################################################################################


### Linear Regression ##########################################################
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(train_X, train_y)  # train_X는 Matrix여야 함 (mXn)

model.coef_  # coefficient
model.predict([[1], [2], [10], [50], [100]])  # predict
model.predict([['a', 'b', 'c']])  # predict
model.score(train_X, train_y) # 결정계수 R^2
################################################################################


### Logistic Regression ########################################################
from sklearn.linear_model import LogisticRegression

train_X = df_lr['dep_A', 'dep_B', 'dep_C', 'dep_D', 'dep_E']
train_y = df_lr['indep']

lr = LogisticRegression(C=100000, random_state=1234, penalty='l2',
                        solver='newton-cg')
model = lr.fit(train_X, train_y)

# accuracy 계산
result = model.predict_proba(test_X)
result = pd.DataFrame(result)
criteria = 0.8 # for example
result = result.assign(estimation=result[1].apply(lambda x:
                                                 'Y' if x >= criteria else 'N'))
                                  # column 1 is probability for Y
test_y = test_y.reset_index()
result = pd.merge(result, test_y, left_index=True, right_index=True)
result = result.assign(accuracy=(result['estimation'] == result['fact']) * 1)

Accuracy = result['accuracy'].sum() / result['accuracy'].count()
################################################################################


### mean squared error #########################################################
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(test_y, prict)
rmse = np.sqrt(mse)
################################################################################


### PCA 주성분 분석 ##############################################################
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

norm_data = StandardScaler().fit_transform(df_pca)
pca = PCA(n_components=6).fit(norm_data)

pca.explained_variance_  # 변수별 주성분의 크기 (eigen value)
pca.explained_variance_ratio_  # 변수별 주성분의 비율
pca.explained_variance_ratio_.sum()  # 추출된 주성분 전체의 설명력(%)
pca.components_.shape  # 주성분 수 X 원 Column 수
pca.components_  # 각 주성분별 원 Data의 Eigen Value

transformed_data = pca.transform(norm_data) # PCA fit 결과에 따른 Data 변환
transformed_data = pd.DataFrame(transformed_data) # DataFrame으로 변환
################################################################################


### T-Test #####################################################################
# H0 : 두 집단의 평균이 같다 (다르다고 할 수 없다)
# [COMPARISON T-TEST vs. ANOVA]
#     T-test is a hypothesis test that is used to compare the means of two populations.
#     ANOVA is a statistical technique that is used to compare the means of more than two populations.
#     Test statistic:
#       * T-test : (x ̄-µ)/(s/√n)
#       * ANOVA : Between Sample Variance/Within Sample Variance
from scipy.stats import ttest_ind

p_val, t_val = ttest_ind(df_a['col'], df_b['col'])
################################################################################


### Quantile ###################################################################
quantile = df_quantile['col'].quantile([0.25, 0.5, 0.75])
################################################################################



### //////// ###################################################################
################################################################################



# Ex02 Data ####################################################################
url_pur = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_purchase_log.csv'
url_cus = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_customer_mst.csv'
url_pro = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_product_mst.csv'

# Ex03 Data ####################################################################
url_antibio = "https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/Antibiotic_70K_patinets.csv"

# Ex04 Data ####################################################################
url_corolla_1 = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/corolla_1.csv'
url_corolla_2 = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/corolla_2.csv'

# Ex05 Data ####################################################################
url_ecommerce = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ecommerce_transaction.csv'

# Ex07 Data ####################################################################
url_pop = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/R_pop_stat.csv'

# Ex08 Data ####################################################################
url_housing = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/California_housing.csv'

# Ex09 Data ####################################################################
url_clothing = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/Womens_Clothing_Reviews.csv'

# Ex10 Data ####################################################################
url_imdb = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/imdb.csv'

# Ex11 Data ####################################################################
url_gestures = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/Gestures.csv'

# Ex12 Data ####################################################################
url_baseball = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/baseball.csv'

# Ex13 Data ####################################################################
url_game = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/13.csv'
