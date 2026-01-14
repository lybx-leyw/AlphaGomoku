"""
棋盘设置:15*15
黑棋的位置(-1)，白棋的位置(1)，身份标识，全局视野。

游戏环境搭建：
 / 说明：各种函数设计应该以训练为导向
 / 目标函数实现：
    1.游戏基础功能实现
    2.mcts树构建（放在mcts库中）
    3.棋局可视化技术
"""

# 导入必要的包
import torch
import numpy as np
import math
from mcts import MCTSsearch as search

# 1.游戏基础功能实现(64局游戏并行)
class GomokuGame:
    def __init__(self, paraLines=64):
        self.paraLines = paraLines
        self.board_size = 15
        self.reset()

    def reset(self):
        # 初始化棋盘状态
        self.trainMap = torch.zeros((self.paraLines, 4, self.board_size, self.board_size))
        # 设置身份标识层，黑棋先手
        self.trainMap[:, 2, :, :] = -1 
        self.curPlayer = -1
        # 各局游戏状态初始化为未结束
        self.gameOver = [False] * self.paraLines

    def makeMove(self, place):
        # 解析坐标
        row = place // 15
        col = place % 15
        # 落子
        for line in range(self.paraLines):
            row_pos = int(row[line].item())
            col_pos = int(col[line].item())  
            # 跳过已结束的局
            if int(place[line].item()) == -1 or self.gameOver[line]:
                continue
            if self.trainMap[line, 3, row_pos, col_pos] == 0:
                self.trainMap[line, 3, row_pos, col_pos] = self.curPlayer
            else:
                print("Invalid Move!")
                return False
            self.trainMap[line, self.curPlayer+1, row_pos, col_pos] = self.curPlayer
            # 判断胜负
            result = self.judgeWin(row_pos, col_pos, line)
            if result == True:
                self.gameOver[line] = True
        # 切换玩家
        self.curPlayer = -self.curPlayer
        # 更新身份标识层
        self.trainMap[:, 2, :, :] = self.curPlayer
        return True
    
    def judgeWin(self, row, col, line):
        # 解析这一步落子能否获胜
        # 利用卷积核快速检索
        kernels = [
            torch.ones(1, 1, 1, 5, dtype=torch.float32),
            torch.ones(1, 1, 5, 1, dtype=torch.float32),
            torch.eye(5, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            torch.flip(torch.eye(5, dtype=torch.float32), dims=[1]).unsqueeze(0).unsqueeze(0),
        ]
        # 水平检索
        trainMap = self.trainMap[:, self.curPlayer+1, :, :]
        # 截取局部区域
        regionMap = trainMap[:,max(0,row-4):min(self.board_size,row+5),max(0, col-4):min(self.board_size, col+5)]
        for kernel in kernels:
            conv_result = torch.nn.functional.conv2d(
                regionMap.unsqueeze(1), kernel, padding=0
            )
            if conv_result.any() == 5:
                return True
        return False
    
    # 呈现显示其中一局的棋盘状态（3.棋局可视化技术的简单实现）
    def viewBoard(self, line=0):
        # 可视化当前棋盘状态
        print(f"Game Line {line+1}:")
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.trainMap[line, 3, i, j] == -1:
                    print("X", end=' ')
                elif self.trainMap[line, 3, i, j] == 1:
                    print("O", end=' ')
                else:
                    print(".", end=' ')
            print()
        print()
    
    def MCTSsearch(self,line):#2.mcts，在对应.py模块中实现
        return search(self.trainMap[line,3,:,:],self.curPlayer)