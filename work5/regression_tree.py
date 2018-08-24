"""
@Project   : Imylu
@Module    : decision_regression.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/22/18 4:29 PM
@Desc      : 
"""
import copy
import numpy as np
from utils import (load_boston_house_prices,
                   train_test_split,
                   get_r2, run_time)
from work5.logger_setup import define_logger
logger = define_logger('work5.decision_regression')


class Node:
    def __init__(self, score: float = None):
        """Node class to build tree leaves.

        Keyword Arguments:
            score {float} -- prediction of y (default: {None})
        """
        self.score = score

        self.feature = None
        self.split = None

        self.left = None
        self.right = None


class RegressionTree:
    def __init__(self):
        """DecisionRegressionTree class.

        Decision tree is for discrete variables
        Regression tree is for continuous variables
        DesicionRegressionTree can handle both

        Attributes:
            root: the root node of DecisionTree
            height: the height of DecisionTree
        """

        self.root = Node()
        self.height = 0

    def _get_split_mse(self, X, y, idx, feature, split):
        """Calculate the mse of each set when x is splitted into two pieces.
        MSE as Loss fuction:
        y_hat = Sum(y_i) / n, i <- [1, n]
        Loss(y_hat, y) = Sum((y_hat - y_i) ^ 2), i <- [1, n]
        Loss = LossLeftNode+ LossRightNode
        --------------------------------------------------------------------

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number, that is, column number of the dataframe
            split {float} -- Split point of x

        Returns:
            tuple -- MSE, split point and average of splitted x in each intervel
        """

        split_sum = [0, 0]
        split_cnt = [0, 0]
        split_sqr_sum = [0, 0]
        # Iterate each row and compare with the split point
        for i in idx:
            # idx are the selected rows of the dataframe
            xi, yi = X[i][feature], y[i]
            if xi < split:
                split_cnt[0] += 1
                split_sum[0] += yi
                split_sqr_sum[0] += yi ** 2
            else:
                split_cnt[1] += 1
                split_sum[1] += yi
                split_sqr_sum[1] += yi ** 2
        # Calculate the mse of y, D(X) = E{[X-E(X)]^2} = E(X^2)-[E(X)]^2
        # num*E{[X-E(X)]^2} = num*E(X^2)-num*[E(X)]^2
        split_avg = [split_sum[0] / split_cnt[0], split_sum[1] / split_cnt[1]]
        split_mse = [split_sqr_sum[0] - split_sum[0] * split_avg[0],
                     split_sqr_sum[1] - split_sum[1] * split_avg[1]]
        return sum(split_mse), split, split_avg

    def _get_split_info(self, X, y, idx, feature, split):
        """Calculate the reduction of standard deviation of each set when x is
        splitted into two pieces.
        Reduction of standard deviation as Loss fuction, the maximal reduction is best
        Or weighted standard deviation as loss function, the minimal value is best
        std of the column - the weighted sum of std of the two groups
        --------------------------------------------------------------------

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number, that is, column number of the dataframe
            split {float} -- Split point of x

        Returns:
            tuple -- MSE, split point and average of splitted x in each intervel
        """
        X = np.array(X)
        select_x = X[idx, feature]
        y = np.array(y)
        select_y = y[idx]
        low_y = select_y[select_x < split].mean()
        high_y = select_y[select_x >= split].mean()
        # split_std_reduce = self.info_continuous(y) - self.condition_info(
        #     select_x, select_y, split)
        split_info = self.condition_info_continuous(select_x, select_y, split)
        return split_info, split, [low_y, high_y]

    def _get_category_info(self, X, y, idx, feature, category_idx):
        """Calculate the standard deviation of each set when x is
        splitted into two discrete parts.
        The weighted standard deviation as loss function, the minimal value is best
        std of the column - the weighted sum of std of the two groups
        --------------------------------------------------------------------

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number, that is, column number of the dataframe
            category_idx {str} -- Chosen category of x to conduct binary classification

        Returns:
            tuple -- MSE, classify point and average of splitted x in each intervel
        """
        X = np.array(X)
        select_x = X[idx, feature]
        y = np.array(y)
        select_y = y[idx]
        low_y = select_y[select_x == category_idx].mean()
        high_y = select_y[select_x != category_idx].mean()
        # split_std_reduce = self.info_continuous(y) - self.condition_info(
        #     select_x, select_y, split)
        split_info = self.condition_info_categorical(select_x, select_y, category_idx)
        return split_info, category_idx, [low_y, high_y]

    def info(self, y):
        """用标准差的大小来表征连续变量的信息量的大小
        Arguments:
            y -- 1d list or numpy.ndarray object with int or float
        """
        return np.std(y)

    def condition_info_continuous(self, X, y, split):
        """
        the weighted continuous information, X is continuous
        :param X: 1d list with int or float
        :param y: 1d list with int or float
        :param split: float
        :return: float
        """
        low_rate = (X < split).sum() / X.size
        # X中的元素低于split的比例，后来算加权平均的信息熵时要用到的权重
        high_rate = 1 - low_rate

        low_info = self.info(y[np.where(X < split)])
        # np.where will give the index of True
        # X < split所对应的Y的值
        high_info = self.info(y[np.where(X >= split)])

        res = low_rate * low_info + high_rate * high_info
        # 加权平均计算分类后的信息熵

        return res

    def condition_info_categorical(self, X, y, category_idx):
        """
        the weighted categorical information, X is categorical
        :param X: 1d list with str
        :param y: 1d list with int or float
        :param category_idx: str
        :return: float
        """
        low_rate = (X == category_idx).sum() / X.size
        # X中的元素低于split的比例，后来算加权平均的信息熵时要用到的权重
        high_rate = 1 - low_rate

        low_info = self.info(y[np.where(X == category_idx)])
        # X < split所对应的Y的值
        high_info = self.info(y[np.where(X != category_idx)])

        res = low_rate * low_info + high_rate * high_info
        # 加权平均计算分类后的信息熵

        return res

    def _choose_split_point(self, X, y, idx, feature):
        """Iterate each xi and split x, y into two pieces,
        and the best split point is the xi when we get minimum mse.

        Arguments:
            x {list} -- 1d list object with int or float
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of mse, feature, split point and average
        """
        # Feature cannot be splitted if there's only one unique element.
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None
        # In case of empty split
        unique.remove(min(unique))
        # Get split point which has min mse
        mse, split, split_avg = min(
            (self._get_split_info(X, y, idx, feature, split)
             # Here we can choose different algorithms
             # _get_split_mse _get_split_info
             for split in unique), key=lambda x: x[0])
        return mse, feature, split, split_avg

    def _choose_category_point(self, X, y, idx, feature):
        """Iterate each xi and classify x, y into two parts,
        and the best category point is the xi when we get minimum info or mse.

        Arguments:
            x {list} -- 1d list with str
            y {list} -- 1d list object with int or float
            idx {list} -- indexes, 1d list object with int
            feature {int} -- Feature number

        Returns:
            tuple -- The best choice of mse, feature, category point and average
        """
        # Feature cannot be splitted if there's only one unique element.
        unique = set([X[i][feature] for i in idx])
        if len(unique) == 1:
            return None
        # 如果该column只剩一个类别的话，就返回None
        # In case of empty split
        unique.remove(min(unique))
        # Get split point which has min mse
        mse, category_idx, split_avg = min(
            (self._get_category_info(X, y, idx, feature, category)
             for category in unique), key=lambda x: x[0])
        return mse, feature, category_idx, split_avg

    def _detect_feature_type(self, x):
        """
        To determine the type of the feature
        :param x: 1d list with int, float or str
        :return: 0 or 1, 0 represents continuous, 1 represents discrete
        """
        for item in x:
            if item is not None:
                return 1 if type(item) == str else 0

    def _get_column(self, X, i):
        return [item[i] for item in X]

    def _choose_feature(self, X, y, idx):
        """Choose the feature which has minimum mse or minimal info.

        Arguments:
            X {list} -- 2d list with int, float or str
            y {list} -- 1d list with int or float
            idx {list} -- indexes, 1d list object with int

        Returns:
            tuple -- (feature number, classify point, average, idx_classify)
            could be None
        """

        m = len(X[0])
        # x[0] selects the first row
        # Compare the mse of each feature and choose best one.
        column_types = [self._detect_feature_type(self._get_column(X, i))
                        for i in range(m)]
        # logger.debug(column_types)
        split_rets = []
        for i in range(m):
            if column_types[i]:
                item = self._choose_category_point(X, y, idx, i)
            else:
                item = self._choose_split_point(X, y, idx, i)
            if item is not None:
                split_rets.append(item)
            # If it is None, it will not be considered as the chosen feature

        # Terminate if no feature can be splitted
        if not split_rets:  # split_rets == []
            return None
        _, feature, split, split_avg = min(
            split_rets, key=lambda x: x[0])
        # Get split idx into two pieces and empty the idx.
        idx_split = [[], []]
        # it contains different groups, and produces idx for next step
        while idx:
            i = idx.pop()
            # logger.debug(i)
            xi = X[i][feature]
            if column_types[feature]:
                if xi == split:
                    idx_split[0].append(i)
                else:
                    idx_split[1].append(i)
            else:
                if xi < split:
                    idx_split[0].append(i)
                else:
                    idx_split[1].append(i)
        return feature, split, split_avg, idx_split

    def _expr2literal(self, expr):
        """Auxiliary function of print_rules.

        Arguments:
            expr {list} -- 1D list like [Feature, op, split]

        Returns:
            str
        """

        feature, op, split = expr
        op = ">=" if op == 1 else "<"
        return "Feature%d %s %.4f" % (feature, op, split)

    def _get_rules(self):
        """Get the rules of all the decision tree leaf nodes.
            Print the rules with breadth-first search for a tree
            广度有限搜索，先打印的是比较浅的叶子，然后从左往右打印
            Expr: 1D list like [Feature, op, split]
            Rule: 2D list like [[Feature, op, split], score]
            Op: -1 means less than, 1 means equal or more than
        """

        que = [[self.root, []]]
        self.rules = []
        # Breadth-First Search
        while que:
            nd, exprs = que.pop(0)
            # Generate a rule when the current node is leaf node
            if not(nd.left or nd.right):
                # Convert expression to text
                literals = list(map(self._expr2literal, exprs))
                self.rules.append([literals, nd.score])
            # Expand when the current node has left child
            if nd.left:
                rule_left = copy.copy(exprs)
                rule_left.append([nd.feature, -1, nd.split])
                que.append([nd.left, rule_left])
            # Expand when the current node has right child
            if nd.right:
                rule_right = copy.copy(exprs)
                rule_right.append([nd.feature, 1, nd.split])
                que.append([nd.right, rule_right])
        # logger.debug(self.rules)

    def fit(self, X, y, max_depth=5, min_samples_split=2):
        """Build a regression decision tree.
        Note:
            At least there's one column in X has more than 2 unique elements
            y cannot be all the same value

        Arguments:
            X {list} -- 2d list object with int or float
            y {list} -- 1d list object with int or float

        Keyword Arguments:
            max_depth {int} -- The maximum depth of the tree. (default: {2})
            min_samples_split {int} -- The minimum number of samples required
            to split an internal node (default: {2})
        """
        # max_depth反映的是树的高度，最多决策多少步
        # min_samples_split决定了在一个feature上可以反复split多少次，
        # min_samples_split越小，可以split的次数越多

        # Initialize with depth, node, indexes
        self.root = Node()
        que = [[0, self.root, list(range(len(y)))]]
        # logger.debug(que)
        # Breadth-First Search
        # 决策树是一层一层构建起来的，所以要用广度优先算法
        while que:
            depth, nd, idx = que.pop(0)
            # Terminate loop if tree depth is more than max_depth
            # que开始是只有一个元素的list，如果没有新的元素加入，就只能循环一次，下一次que就为空了
            if depth == max_depth:
                break
            # Stop split when number of node samples is less than
            # min_samples_split or Node is 100% pure.
            if len(idx) < min_samples_split or \
                    set(map(lambda i: y[i], idx)) == 1:
                continue
            # Stop split if no feature has more than 2 unique elements
            feature_rets = self._choose_feature(X, y, idx)
            # None表示这些idx的X都没法继续分类了
            if feature_rets is None:
                continue
            # 对于这些idx的X，已经没法再细分了，所以对于这些idx的X，继续分类就结束了
            # Split
            nd.feature, nd.split, split_avg, idx_split = feature_rets
            nd.left = Node(split_avg[0])
            nd.right = Node(split_avg[1])
            que.append([depth+1, nd.left, idx_split[0]])
            que.append([depth+1, nd.right, idx_split[1]])
        # Update tree depth and rules
        self.height = depth
        self._get_rules()

    def print_rules(self):
        """Print the rules of all the regression decision tree leaf nodes.
        """

        for i, rule in enumerate(self.rules):
            literals, score = rule
            print("Rule %d: " % i, ' | '.join(
                literals) + ' => split_hat %.4f' % score)

    def _predict(self, row):
        """Auxiliary function of predict.

        Arguments:
            row {list} -- 1D list with int or float

        Returns:
            int or float -- prediction of yi
        """

        nd = self.root
        while nd.left and nd.right:
            if row[nd.feature] < nd.split:
                nd = nd.left
            else:
                nd = nd.right
        return nd.score

    def predict(self, X):
        """Get the prediction of y.
        可以批量预测

        Arguments:
            X {list} -- 2d list object with int or float

        Returns:
            list -- 1d list object with int or float
        """

        return [self._predict(Xi) for Xi in X]


@run_time
def test_continuous_continuous():
    print("Tesing the accuracy of RegressionTree...")
    # Load data
    X, y = load_boston_house_prices()
    # Split data randomly, train set rate 70%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=10)
    # Train model
    reg = RegressionTree()

    reg.fit(X=X_train, y=y_train, max_depth=4)
    # Show rules
    reg.print_rules()
    # Model accuracy
    get_r2(reg, X_test, y_test)


if __name__ == "__main__":
    test_continuous_continuous()
