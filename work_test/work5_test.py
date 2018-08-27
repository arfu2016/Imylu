"""
@Project   : Imylu
@Module    : work5.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/27/18 2:32 PM
@Desc      : 
"""
from work5 import RegressionTree

from work5.load_bike_data import load_bike_sharing_data
from work5.utils import train_test_split, get_r2

X, y = load_bike_sharing_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = RegressionTree()
reg.fit(X=X_train, y=y_train, max_depth=5)
reg.print_rules()
get_r2(reg, X_test, y_test)
print('A prediction:', reg.predict(
    [[*'1 0 1 0 0 6 1'.split(), *[0.24, 0.81, 0.1]]]), sep=' ')
