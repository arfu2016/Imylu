"""
@Project   : Imylu
@Module    : work5_test2.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/27/18 2:50 PM
@Desc      : 
"""
from work5 import RegressionTree

from work5.utils import load_boston_house_prices
from work5.utils import train_test_split, get_r2

X, y = load_boston_house_prices()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = RegressionTree()
reg.fit(X=X_train, y=y_train, max_depth=4)
reg.print_rules()
get_r2(reg, X_test, y_test)
