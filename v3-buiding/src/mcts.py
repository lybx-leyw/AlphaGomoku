"""
棋盘设置:15*15
黑棋的位置(-1)，白棋的位置(1)，身份标识，全局视野。

MCTS算法实现：
 / 说明：MCTS设计应该以训练为导向
 / 目标函数实现：
    1.可控制mcts树构建（选择，拓展，模拟，回溯）
    2.mcts结果可视化技术
"""

"""
mcts目标：
    能针对一个当下局面进行模拟直至value网络显示有明显胜率
"""

# 导入必要的库
import math
import torch
import numpy as np

# 管理MCTS树
class MCTSNode:
    def __init__(self, prior=0, parent=None):
        self.parent = parent # 储存mcts树根节点
        self.children = []
        
    
def MCTSsearch(state,player):
    # 这里的state是[1,1,15,15]的张量
    # 1.选择