import torch
import numpy as np
import math

def draw_map(map):
    print("======================================")
    print("\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14")
    for t_index,t_row in enumerate(map[0][0]):
        print(t_index,end='\t')
        for t_col in t_row:
            if int(t_col.item()) == 1:
                print("O", end='\t')
            elif int(t_col.item()) == -1:
                print("X", end='\t')
            elif int(t_col.item()) == 0:
                print(".", end='\t')
            elif int(t_col.item()) == 3:
                print("\033[31mO\033[0m", end='\t')
            elif int(t_col.item()) == -3:
                print("\033[31mX\033[0m", end='\t')
            elif int(t_col.item()) == 2:
                print("\033[32mO\033[0m", end='\t')
            elif int(t_col.item()) == -2:
                print("\033[32mX\033[0m", end='\t')
            else:
                print("?", end='\t')
        print('\n')
    print("======================================")

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

def help_ai_1(show_map):
    can_win = 0
    if check_gomoku_win_train(show_map) == 0:
        for i in range(15):
            for j in range(15):
                t_board = show_map.clone()
                if t_board[0,0,i,j] == 0:
                    t_board[0,0,i,j] = -1
                    if check_gomoku_win_train(t_board) == -1:
                        can_win = 1
                        return  i,j,can_win
                    else:
                        t_board[0,0,i,j] = 0
    return -1,-1,can_win

def help_ai_2(show_map):
    can_win = 0
    if check_gomoku_win_train(show_map) == 0:
        for i in range(15):
            for j in range(15):
                t_board = show_map.clone()
                if t_board[0,0,i,j] == 0:
                    t_board[0,0,i,j] = 1
                    if check_gomoku_win(t_board) == 1:
                        can_win = 1
                        return  i,j,can_win
                    else:
                        t_board[0,0,i,j] = 0
    return -1,-1,can_win

def check_gomoku_win(board):
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
                    for k in range(5):
                        t_board[0,0,i,j+k] *= 3
                    draw_map(t_board)
                    return int(stone)
            # 检查垂直方向
            if i <= 10:
                if all(board_2d[i+k, j].item() == stone for k in range(5)):
                    for k in range(5):
                        t_board[0,0,i+k,j] *= 3
                    draw_map(t_board)
                    return int(stone)
            # 检查主对角线
            if i <= 10 and j <= 10:
                if all(board_2d[i+k, j+k].item() == stone for k in range(5)):
                    for k in range(5):
                        t_board[0,0,i+k,j+k] *= 3
                    draw_map(t_board)
                    return int(stone)
            # 检查副对角线
            if i <= 10 and j >= 4:
                if all(board_2d[i+k, j-k].item() == stone for k in range(5)):
                    for k in range(5):
                        t_board[0,0,i+k,j-k] *= 3
                    draw_map(t_board)
                    return int(stone)
    return 0

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

# MCTS节点类
class MCTSNode:
    pass

# MCTS搜索函数
def mcts_search():
    pass

def draw_prob_map(action_prob_dict, show_map):
    print("======================================")
    print("\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12\t13\t14")
    for i in range(15):
        print(i, end='\t')
        for j in range(15):
            action = i * 15 + j
            if show_map[0, 0, i, j] != 0:
                if show_map[0, 0, i, j] == 1:
                    print("O", end='\t')
                elif show_map[0, 0, i, j] == -1:
                    print("X", end='\t')
                else:
                    print("?", end='\t')
            else:
                # 若该位置为空，显示概率
                # 若概率为0，显示空格
                prob = action_prob_dict.get(action, 0.0)
                if prob == 0.0:
                    print(".", end='\t')
                else:
                    print(f"{100*prob:.0f}", end='\t')
        print('\n')
    print("======================================")

def get_legal_actions(show_map):
    legal_actions = []
    for i in range(15):
        for j in range(15):
            if show_map[0, 0, i, j] == 0:  # 空位置
                legal_actions.append(i * 15 + j)  # 将二维坐标转换为一维索引
    return legal_actions

def take_action(state_map, state_show, action, player, trick_net):
    new_map = state_map.clone()
    new_show = state_show.clone()
    
    # 将一维动作转换为二维坐标
    row = action // 15
    col = action % 15
    
    # 执行落子（使用辅助函数）
    puton(new_map, new_show, int(row), int(col), player, trick_net, player)
    
    # 更新先后手标记
    if player == -1:  # 刚下了黑棋，下一手是白棋
        new_map[0, 2, :, :] = 1  # 更高效的方式
    else:  # 刚下了白棋，下一手是黑棋
        new_map[0, 2, :, :] = -1
    
    return new_map, new_show

