import pandas as pd
import numpy as np
import math

url_pur = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_purchase_log.csv'
url_cus = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_customer_mst.csv'
url_pro = 'https://raw.githubusercontent.com/nullpitch-dev/hj_public/master/ds_product_mst.csv'

data_pur = pd.read_csv(url_pur)
data_cus = pd.read_csv(url_cus)
data_pro = pd.read_csv(url_pro)


# elif in lambda ##############################
df_elif = df_elif.assign(new_col=df_elif["Col_A"].apply(lambda x: "value_1" if x <= condi_1 else (
                                                                  "value_2" if x >= condi_2 else "middle")))

# Merge #######################################
df_merge = pd.merge(data_left, data_right[["Col_A", "Col_B"]], how="left", left_on="Key_left", right_on="Key_right")

# Rank  #######################################
df_rank = df_rank.assign(rank=df_rank.rank(ascending=False, method="min"))

#####################################################################################################

# ANOVA #######################################
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison

result = ols(formula="item_amt ~ C(age_cls)", data=data3).fit()
anova_table = anova_lm(result)
f_val = anova_table["F"][0]



