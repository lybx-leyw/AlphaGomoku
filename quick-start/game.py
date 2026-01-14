import torch
from torch import nn
import numpy as np

from src.game_assitant import check_gomoku_win
from src.game_assitant import puton
from src.game_assitant import help_ai_1
from src.game_assitant import help_ai_2
from src.game_assitant import mcts_search
from src.game_assitant import draw_prob_map
from src.game_assitant import draw_map
from src.net import trick_net
from src.net import value_net


#0.载入AI模型
model = trick_net()
print("AI难度介绍：1-简单决策AI，2-激进决策AI，3-平衡决策AI")
level = int(input("请输入难度（1/2/3）"))
if level == 1 or level == 2 or level == 3 or level == 4 or level == 5:
    state_dict = torch.load(f"model//trick.pkl")
if isinstance(state_dict, nn.Module):
    model = state_dict
else:
    model.load_state_dict(state_dict)
trick_net = model

model_value = value_net()
state_dict_value = torch.load(f"model//value.pkl")
if isinstance(state_dict_value, nn.Module):
    model_value = state_dict_value
else:
    model_value.load_state_dict(state_dict_value)
value_net = model_value

n_of_simulations = 100  # MCTS模拟次数
if level == 4 or level == 5:
    n_of_simulations = int(input("请输入MCTS模拟次数（建议100-1000）"))

#1.创建棋盘
map = torch.zeros(1,4,15,15)
show_map = torch.zeros(1,1,15,15)
#2.初始化先后手
role_ai = int(input("请输入AI的先后手（1，-1，2）"))
if role_ai != 2:
    for i in range(15):
        for j in range(15):
            map[0,2,i,j] = role_ai
else:
    for i in range(15):
        for j in range(15):
            map[0,2,i,j] = -1

#3.落子更新
draw_map(show_map)
while 1:
    if role_ai == 1:
        #玩家落子
        print("玩家请落子:")
        row,col = input().split()
        puton(map,show_map,int(row),int(col),-1,trick_net,role_ai)
    elif role_ai == 2:
        output = trick_net(map)
        if level == 1:
            _,pred = output.max(1)
        elif level == 2 or level == 3:
            use_value_net = 1
            if level == 3:
                # 不恰当的走棋我才用价值网络
                # 四面八方没有棋子
                if  (i+1>14 or show_map[0,0,i+1,j] == 0) and\
                    (i-1<0  or show_map[0,0,i-1,j] == 0) and\
                    (j+1>14 or show_map[0,0,i,j+1] == 0) and\
                    (j-1<0 or show_map[0,0,i,j-1] == 0) and\
                    ((i+1>14 or j+1>14) or show_map[0,0,i+1,j+1] == 0) and\
                    ((i-1<0   or j-1<0)  or show_map[0,0,i-1,j-1] == 0) and\
                    ((i+1>14 or j-1<0)  or show_map[0,0,i+1,j-1] == 0) and\
                    ((i-1<0  or j+1>14) or show_map[0,0,i-1,j+1] == 0):
                    use_value_net = 0
                else:
                    use_value_net = 1
            _,pred_topk = output.topk(2,1)
            value_list = []
            for pred in pred_topk[0]:
                # 解析步骤
                row = pred//15
                col = pred %15
                # 非法走法直接跳过
                if int(row)>14 or int(row)<0 or int(col)>14 or int(col)<0 or \
                    show_map[0,0,int(row),int(col)] != 0:
                    continue
                # 拟落子
                show_map[0,0,row,col] = -1
                value = value_net(show_map)
                value = nn.functional.softmax(value, dim=1)
                value_list.append(float(100*value[0,0]))# 黑子胜率
                show_map[0,0,row,col] = 0
                if use_value_net == 0:
                    break
            # 选子
            value_list = torch.tensor([value_list])
            _,max_value = value_list.max(1)
            pred = pred_topk[0,max_value]
        elif level == 4:
            action_probs, legal_actions_mcts = mcts_search(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        elif level == 5:
            action_probs, legal_actions_mcts = mcts_search_level5(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        #解析步骤
        row = pred//15
        col = pred %15
        t_row,t_col,can_win = help_ai_1(show_map)
        if can_win == 1:
            row,col = t_row,t_col
        elif can_win == 0:
            t_row, t_col, can_win = help_ai_2(show_map)
            if can_win == 1:
                row,col = t_row,t_col
        #落子
        puton(map,show_map,int(row),int(col),-1,trick_net,-1)
        for i in range(15):
            for j in range(15):
                map[0,2,i,j] = 1
        value = value_net(show_map)
        value = nn.functional.softmax(value, dim=1)
        print(f"我的胜率为{100.0*float(value[0,0].item())}%"
              f"你的胜率为{100.0*float(value[0,2].item())}%")
    else:
        print("AI思考中...")
        output = trick_net(map)
        if level == 1:
            _,pred = output.max(1)
        elif level == 2 or level == 3:
            use_value_net = 1
            if level == 3:
                # 不恰当的走棋我才用价值网络
                # 四面八方没有棋子
                if  (i+1>14 or show_map[0,0,i+1,j] == 0) and\
                    (i-1<0  or show_map[0,0,i-1,j] == 0) and\
                    (j+1>14 or show_map[0,0,i,j+1] == 0) and\
                    (j-1<0 or show_map[0,0,i,j-1] == 0) and\
                    ((i+1>14 or j+1>14) or show_map[0,0,i+1,j+1] == 0) and\
                    ((i-1<0  or j-1<0)  or show_map[0,0,i-1,j-1] == 0) and\
                    ((i+1>14 or j-1<0)  or show_map[0,0,i+1,j-1] == 0) and\
                    ((i-1<0  or j+1>14) or show_map[0,0,i-1,j+1] == 0):
                    use_value_net = 0
                else:
                    use_value_net = 1
            _,pred_topk = output.topk(2,1)
            value_list = []
            for pred in pred_topk[0]:
                # 解析步骤
                row = pred//15
                col = pred %15
                # 非法走法直接跳过
                if int(row)>14 or int(row)<0 or int(col)>14 or int(col)<0 or \
                    show_map[0,0,int(row),int(col)] != 0:
                    continue
                # 拟落子
                show_map[0,0,row,col] = -1
                value = value_net(show_map)
                value = nn.functional.softmax(value, dim=1)
                value_list.append(float(100*value[0,0]))# 黑子胜率
                show_map[0,0,row,col] = 0
                if use_value_net == 0:
                    break   
            # 选子
            value_list = torch.tensor([value_list])
            _,max_value = value_list.max(1)
            pred = pred_topk[0,max_value]
        elif level == 4:
            action_probs, legal_actions_mcts = mcts_search(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                # 选择可能性最大的动作
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        elif level == 5:
            action_probs, legal_actions_mcts = mcts_search_level5(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        #解析步骤
        row = pred//15
        col = pred %15
        t_row,t_col,can_win = help_ai_1(show_map)
        if can_win == 1:
            row,col = t_row,t_col
        elif can_win == 0:
            t_row, t_col, can_win = help_ai_2(show_map)
            if can_win == 1:
                row,col = t_row,t_col
        #落子
        puton(map,show_map,int(row),int(col),-1,trick_net,role_ai)
        print(f"我下在({int(row)}，{int(col)})好啦！")
        value = value_net(show_map)
        value = nn.functional.softmax(value, dim=1)
        print(f"我的胜率为{100.0*float(value[0,0].item())}%"
              f"你的胜率为{100.0*float(value[0,2].item())}%")
    draw_map(show_map)

    if role_ai == 1:
        if check_gomoku_win(show_map) == -1:
            print("玩家胜利！")
            break
    elif role_ai == 2:
        if check_gomoku_win(show_map) == -1:
            print("黑子AI获胜")
            break
    elif check_gomoku_win(show_map) == -1:
        print("不过尔尔！")
        break

    is_draw = 1
    if check_gomoku_win(show_map) == 0:
        for i in range(15):
            for j in range(15):
                if show_map[0,0,i,j] == 0:
                    is_draw = 0
                    break
            if is_draw == 0:
                break
    if is_draw == 1:
        print("draw!平局")
        break
    ''''''

    if role_ai == -1:
        #玩家落子
        print("到你落子啦！")
        row,col = input().split()
        puton(map,show_map,int(row),int(col),1,trick_net,role_ai)
    elif role_ai == 2:
        output = trick_net(map)
        if level == 1:
            _,pred = output.max(1)
        elif level == 2 or level == 3:
            use_value_net = 1
            if level == 3:
                # 不恰当的走棋我才用价值网络
                # 四面八方没有棋子
                if  (i+1>14 or show_map[0,0,i+1,j] == 0) and\
                    (i-1<0  or show_map[0,0,i-1,j] == 0) and\
                    (j+1>14 or show_map[0,0,i,j+1] == 0) and\
                    (j-1<0 or show_map[0,0,i,j-1] == 0) and\
                    ((i+1>14 or j+1>14) or show_map[0,0,i+1,j+1] == 0) and\
                    ((i-1<0  or j-1<0)  or show_map[0,0,i-1,j-1] == 0) and\
                    ((i+1>14 or j-1<0)  or show_map[0,0,i+1,j-1] == 0) and\
                    ((i-1<0  or j+1>14) or show_map[0,0,i-1,j+1] == 0):
                    use_value_net = 0
                else:
                    use_value_net = 1
            _,pred_topk = output.topk(2,1)
            value_list = []
            for pred in pred_topk[0]:
                # 解析步骤
                row = pred//15
                col = pred %15
                # 非法走法直接跳过
                if int(row)>14 or int(row)<0 or int(col)>14 or int(col)<0 or \
                    show_map[0,0,int(row),int(col)] != 0:
                    continue
                # 拟落子
                show_map[0,0,row,col] = 1
                value = value_net(show_map)
                value = nn.functional.softmax(value, dim=1)
                value_list.append(float(100*value[0,2]))# 白子胜率
                show_map[0,0,row,col] = 0
                if use_value_net == 0:
                    break
            # 选子
            value_list = torch.tensor([value_list])
            _,max_value = value_list.max(1)
            pred = pred_topk[0,max_value]
        elif level == 4:
            action_probs, legal_actions_mcts = mcts_search(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        elif level == 5:
            action_probs, legal_actions_mcts = mcts_search_level5(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        #解析步骤
        row = pred//15
        col = pred %15
        t_row,t_col,can_win = help_ai_1(show_map)
        if can_win == 1:
            row,col = t_row,t_col
        elif can_win == 0:
            t_row, t_col, can_win = help_ai_2(show_map)
            if can_win == 1:
                row,col = t_row,t_col
        #落子
        puton(map,show_map,int(row),int(col),1,trick_net,1)
        for i in range(15):
            for j in range(15):
                map[0,2,i,j] = -1
        value = value_net(show_map)
        value = nn.functional.softmax(value, dim=1)
        print(f"我的胜率为{100.0*float(value[0,2].item())}%"
              f"你的胜率为{100.0*float(value[0,0].item())}%")
    else:
        output = trick_net(map)
        if level == 1:
            _,pred = output.max(1)
        elif level == 2 or level == 3:
            use_value_net = 1
            if level == 3:
                # 不恰当的走棋我才用价值网络
                # 四面八方没有棋子
                if  (i+1>14 or show_map[0,0,i+1,j] == 0) and\
                    (i-1<0  or show_map[0,0,i-1,j] == 0) and\
                    (j+1>14 or show_map[0,0,i,j+1] == 0) and\
                    (j-1<0 or show_map[0,0,i,j-1] == 0) and\
                    ((i+1>14 or j+1>14) or show_map[0,0,i+1,j+1] == 0) and\
                    ((i-1<0  or j-1<0)  or show_map[0,0,i-1,j-1] == 0) and\
                    ((i+1>14 or j-1<0)  or show_map[0,0,i+1,j-1] == 0) and\
                    ((i-1<0  or j+1>14) or show_map[0,0,i-1,j+1] == 0):
                    use_value_net = 0
                else:
                    use_value_net = 1
            _,pred_topk = output.topk(2,1)
            value_list = []
            for pred in pred_topk[0]:
                # 解析步骤
                row = pred//15
                col = pred %15
                # 非法走法直接跳过
                if int(row)>14 or int(row)<0 or int(col)>14 or int(col)<0 or \
                    show_map[0,0,int(row),int(col)] != 0:
                    continue
                # 拟落子
                show_map[0,0,row,col] = 1
                value = value_net(show_map)
                value = nn.functional.softmax(value, dim=1)
                value_list.append(float(100*value[0,2]))# 白子胜率
                show_map[0,0,row,col] = 0
                if use_value_net == 0:
                    break
            # 选子
            value_list = torch.tensor([value_list])
            _,max_value = value_list.max(1)
            pred = pred_topk[0,max_value]
        elif level == 4:
            action_probs, legal_actions_mcts = mcts_search(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        elif level == 5:
            action_probs, legal_actions_mcts = mcts_search_level5(
                map, show_map, trick_net, value_net, 
                -1, max_simulations=n_of_simulations)
            # 输出搜索概率分布
            draw_prob_map(action_probs, show_map)
            for action in legal_actions_mcts:
                if action not in action_probs:
                    action_probs[action] = 0.0
                # 选择动作
                actions = list(action_probs.keys())
                probs = [action_probs[a] for a in actions]
                # 归一化概率（确保和为1）
                prob_sum = sum(probs)
                if prob_sum > 0:
                    probs = [p / prob_sum for p in probs]
                else:
                    probs = [1.0/len(actions) for _ in actions]
                action_idx = np.argmax(probs)  # 返回概率最大的索引
                action = actions[action_idx]
                pred = action
        #解析步骤
        row = pred//15
        col = pred %15
        t_row,t_col,can_win = help_ai_2(show_map)
        if can_win == 1:
            row,col = t_row,t_col
        elif can_win == 0:
            t_row, t_col, can_win = help_ai_1(show_map)
            if can_win == 1:
                row,col = t_row,t_col
        #落子
        puton(map,show_map,int(row),int(col),1,trick_net,role_ai)
        print(f"我下在({int(row)}，{int(col)})好啦！")
        value = value_net(show_map)
        value = nn.functional.softmax(value, dim=1)
        print(f"我的胜率为{100.0*float(value[0,2].item())}%"
              f"你的胜率为{100.0*float(value[0,0].item())}%")
    draw_map(show_map)
    if role_ai == 1:
        if check_gomoku_win(show_map) == 1:
            print("我赢啦！")
            break
    elif role_ai == 2:
        if check_gomoku_win(show_map) == 1:
            print("白子AI获胜")
            break
    elif check_gomoku_win(show_map) == 1:
        print("ai甘拜下风。")
        break

    is_draw = 1
    if check_gomoku_win(show_map) == 0:
        for i in range(15):
            for j in range(15):
                if show_map[0,0,i,j] == 0:
                    is_draw = 0
                    break
            if is_draw == 0:
                break
    if is_draw == 1:
        print("draw!平局")
        break