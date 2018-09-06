"""
@Project   : Imylu
@Module    : __init__.py.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/28/18 1:19 PM
@Desc      : natural language understanding

The core is sentence vector and paragraph vector
以技术为核心，机器人大脑的原理

实体抽取：elastic search；模板规则，正则规则，词性规则，句法规则；word window classification
不需要句向量的事件抽取和文本分类：模板规则，正则规则，句法规则

事件抽取（实体抽取），文本分类
prerequisite: sentence vector (nonparametric, parametric), paragraph vector

event extraction and text classification:
nonparametric: similarity (Nearest neighbor search, 除了brute force，还有可以优化的算法),
               k nearest neighbors, decision tree, emd（地球物理）
parametric: linear regression, logistic regression, softmax regression,
            support vector machine, neural net, cnn, rnn

deep learning: machine reading comprehension, gan, lang2sql

multi-round conversation: rasa-core
"""
