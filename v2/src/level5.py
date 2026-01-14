import math
import numpy as np
import torch

def check_gomoku_win_train(board):
    for i in range(15):
        for j in range(15):
            if (board[0, 0, i, j] > 1 or board[0, 0, i, j] < -1):
                if board[0, 0, i, j] > 0:
                    board[0, 0, i, j] = 1
                elif board[0, 0, i, j] < 0:
                    board[0, 0, i, j] = -1
    board_2d = board[0, 0]
    t_board  = torch.zeros(1,1,15,15)
    t_board  = board
    # 检查所有可能的五连珠位置
    for i in range(15):
        for j in range(15):
            stone = board_2d[i, j].item()
            if stone == 0:
                continue
            # 检查水平方向
            if j <= 10:
                if all(board_2d[i, j+k].item() == stone for k in range(5)):
                    return int(stone)
            # 检查垂直方向
            if i <= 10:
                if all(board_2d[i+k, j].item() == stone for k in range(5)):
                    return int(stone)
            # 检查主对角线
            if i <= 10 and j <= 10:
                if all(board_2d[i+k, j+k].item() == stone for k in range(5)):
                    return int(stone)
            # 检查副对角线
            if i <= 10 and j >= 4:
                if all(board_2d[i+k, j-k].item() == stone for k in range(5)):
                    return int(stone)
    return 0


def puton(map,show_map,row,col,num,cnn,role_ai):
    k = 2
    while row>14 or row<0 or col>14 or col<0 or show_map[0,0,row,col] != 0:
        if num == 1:
            if role_ai == -1:
                print("wrong put,please try again")
                row,col = input().split()
                row = int(row)
                col = int(col)
            else:
                output = cnn(map)
                _ , indices = output.topk(k,dim=1)
                # 取第k-1个索引（因为索引从0开始）
                pred = indices[0, k-1]
                row = pred // 15
                col = pred % 15
                k += 1
        else:
            if role_ai == 1:
                print("wrong put,please try again")
                row,col = input().split()
                row = int(row)
                col = int(col)
            else:
                output = cnn(map)
                _ , indices = output.topk(k,dim=1)
                # 取第k-1个索引（因为索引从0开始）
                pred = indices[0, k-1]
                row = pred // 15
                col = pred % 15
                k += 1
    if num == 1:# ai2
        map[0,0,row,col]  = num
    else:
        map[0,1,row,col]  = num
    map[0,3,row,col]  = num
    show_map[0,0,row,col] = num

# MCTS节点类
class MCTSNode:
    pass

# MCTS搜索函数
def mcts_search():
    pass