"""
net1,net2,net3共同的缺点是太过关注局部块状战术，缺少全局视野和线性视野(卷积核大小太小)
net4攻防一体但边角处理能力薄弱(添加了第四通道，但没有池化)
net5攻击强悍但防御较薄弱(缺少价值评判的机制)
net6增强了部分价值评判属性，但仍然保留防御薄弱的弱点
目前net4表现最佳
net5有点棒槌，但偏偏边界处理比4好，且攻击强悍，先手优势大
net6的优化不明显，克制net5，但在真人实战中效果不觉得如何，反倒net4表现更佳
net7可能过于保守
net8也没很好改善，5，6，7，8虽在边界处理上更优，但总体不如net4
net9表现很差
"""

import torch
from torch import nn
from judge import check_gomoku_win
from Symplenet import symplenet as s1 
from net3 import symplenet as s3
from net2 import symplenet as s2
from net4 import symplenet as s4
from net5 import symplenet as s5
from net9 import symplenet as s9
from net4 import my_Qnet as s15


def help_ai_1(show_map):
    can_win = 0
    if check_gomoku_win(show_map) == 0:
        for i in range(15):
            for j in range(15):
                t_board = show_map.clone()
                if t_board[0,0,i,j] == 0:
                    t_board[0,0,i,j] = -1
                    if check_gomoku_win(t_board) == -1:
                        can_win = 1
                        return  i,j,can_win
                    else:
                        t_board[0,0,i,j] = 0
    return -1,-1,can_win

def help_ai_2(show_map):
    can_win = 0
    if check_gomoku_win(show_map) == 0:
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

class s11(nn.Module):

    def __init__(self):
        super(s11, self).__init__()
        # 直接挂上预训练模型
        model = s9()
        state_dict = torch.load("net9.pkl")
        # 若加载的是整个模型
        if isinstance(state_dict, nn.Module):
            model = state_dict
        else:  # 否则，自己来载入参数
            model.load_state_dict(state_dict)
        self.net0 = model

        # 强化网络
        self.Anet = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.Anetfc = nn.Sequential(
            nn.Linear(4 * 15 * 15, 225)
        )


    def forward(self, x):
        out1 = self.net0(x)
        out2 = self.Anet(x)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.Anetfc(out2)
        final_out = out1 + out2
        return final_out


class s12(nn.Module):

    def __init__(self):
        super(s12, self).__init__()
        # 直接挂上预训练模型
        model = s4()
        state_dict = torch.load("net4.pkl")
        # 若加载的是整个模型
        if isinstance(state_dict, nn.Module):
            model = state_dict
        else:  # 否则，自己来载入参数
            model.load_state_dict(state_dict)
        self.net0 = model

        # 强化网络
        self.Anet = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.Anetfc = nn.Sequential(
            nn.Linear(4 * 15 * 15, 225)
        )

    def forward(self, x):
        out1 = self.net0(x)
        out2 = self.Anet(x)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.Anetfc(out2)
        final_out = out1 + out2
        return final_out

class s13(nn.Module):

    def __init__(self):
        super(s13, self).__init__()
        # 直接挂上预训练模型
        model = s4()
        state_dict = torch.load("net4.pkl")
        # 若加载的是整个模型
        if isinstance(state_dict, nn.Module):
            model = state_dict
        else:  # 否则，自己来载入参数
            model.load_state_dict(state_dict)
        self.net0 = model

        # 强化网络
        self.Anet = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.Anetfc = nn.Sequential(
            nn.Linear(4 * 15 * 15, 225)
        )

    def forward(self, x):
        out1 = self.net0(x)
        out2 = self.Anet(x)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.Anetfc(out2)
        final_out = 0.0*out1 + 1.0*out2
        return final_out


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

def puton(map,map_2,show_map,row,col,num):
    k = 2
    for i in range(15):
        for j in range(15):
            if ai_1 >= 9 and (map[0, 3, i, j] > 1 or map[0, 3, i, j] < -1):
                if map[0, 3, i, j] > 0:
                    map[0, 3, i, j] = 1
                elif map[0, 3, i, j] < 0:
                    map[0, 3, i, j] = -1
            if ai_2 >= 9 and (map_2[0, 3, i, j] == 2 or map_2[0, 3, i, j] == -2):
                if map_2[0, 3, i, j] > 0:
                    map_2[0, 3, i, j] = 1
                elif map_2[0, 3, i, j] < 0:
                    map_2[0, 3, i, j] = -1
    while row>14 or row<0 or col>14 or col<0 or show_map[0,0,row,col] != 0:
        if num == 1:
            if ai_2 == 0:
                print("wrong put,please try again")
                row,col = input().split()
                row = int(row)
                col = int(col)
            else:
                output = cnn_2(map_2)
                _ , indices = output.topk(k,dim=1)
                # 取第k-1个索引（因为索引从0开始）
                pred = indices[0, k-1]
                row = pred // 15
                col = pred % 15
                k += 1
        else:
            if ai_1 == 0:
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
        map[0,0,row,col]  = k1*num
        map_2[0,0,row,col]  = k1*num
        if map.shape[1]>3:
            map[0,3,row,col]  = num
            if ai_1 >= 9 and ai_1<=13:
                map[0,3,row,col] = 2*num
        if map_2.shape[1]>3:
            map_2[0,3,row,col]  = num
            if ai_2 >= 9 and ai_2<=13:
                map_2[0,3,row,col] = 2*num
    else:
        map[0,1,row,col]  = k2*num
        map_2[0,1,row,col]  = k2*num
        if map.shape[1]>3:
            map[0,3,row,col]  = num
            if ai_1 >= 9 and ai_1<=13:
                map[0,3,row,col] = 2*num
        if map_2.shape[1]>3:
            map_2[0,3,row,col]  = num
            if ai_2 >= 9 and ai_2<=13:
                map_2[0,3,row,col] = 2*num
    for i in range(15):
        for j in range(15):
            if ai_2 >= ai_1 and ai_2>3:
                show_map[0,0,i,j] = map_2[0,3,i,j]
            elif ai_2 < ai_1 and ai_1>3:
                show_map[0,0,i,j] = map[0,3,i,j]
            else:
                show_map[0,0,i,j] = map[0,0,i,j]+map[0,1,i,j]


#0.载入AI模型
ai_1 = 4
ai_2 = 15
k1 = 1
k2 = 1
if ai_1 == 6:
    k1 = 1.1
elif ai_1 == 7:
    k1 = 1.3
elif ai_1 == 8:
    k1 = 1.2
elif ai_1 == 9:
    k1 = 1.2
if ai_2 == 6:
    k2 = 1.1
elif ai_2 == 7:
    k2 = 1.3
elif ai_2 == 8:
    k2 = 1.2
elif ai_1 == 9:
    k2 = 1.2
model_map = {0: None, 1: s1, 2: s2, 3: s3, 4: s4, 5: s5, 6: s5, 7: s5, 8: s5, 9: s9, 10:s9, 11:s11, 12:s11,13:s11,14:s12,15:s4}
if ai_1 != 0:
    model = model_map[ai_1]()
if ai_2 != 0:
    model_2 = model_map[ai_2]()
if ai_1 != 0:
    state_dict = torch.load(f"net{ai_1}.pkl")
    if isinstance(state_dict, nn.Module):
        model = state_dict
    else:
        model.load_state_dict(state_dict)
    model.eval()
    cnn = model
if ai_2 != 0:
    state_dict_2 = torch.load(f"net{ai_2}.pkl")
    if isinstance(state_dict_2, nn.Module):
        model_2 = state_dict_2
    else:
        model_2.load_state_dict(state_dict_2)
    model_2.eval()
    cnn_2 = model_2
#1.创建棋盘
size = 3
size_2 = 3
if ai_1>3: 
    size   = 4
if ai_2>3: 
    size_2 = 4   
map = torch.zeros(1,size,15,15)
map_2 = torch.zeros(1,size_2,15,15)
show_map = torch.zeros(1,1,15,15)
#2.初始化先后手
for i in range(15):
    for j in range(15):
        map[0,2,i,j] = -1 # 你是‘-1’这一手（先手）
for i in range(15):
    for j in range(15):
        map_2[0,2,i,j] = 1 # 你是‘1’这一手（后手）
#3.落子更新
draw_map(show_map)
while 1:
    if ai_1 == 0:
        #玩家落子
        print("玩家1请落子:")
        row,col = input().split()
        puton(map,map_2,show_map,int(row),int(col),-1)
    else:
        print("AI1思考中...")
        output = cnn(map)
        _,pred = output.max(1)
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
        puton(map,map_2,show_map,row,col,-1)
        print(f"我下在{row}，{col}好啦！")
    draw_map(show_map)
    if ai_1 == 0:
        if check_gomoku_win(show_map) == -1:
            print("玩家1胜利！")
            break
    elif check_gomoku_win(show_map) == -1:
        print("我是ai1,我赢啦！")
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
    if ai_2 == 0:
        #玩家落子
        print("玩家2:到你落子啦！")
        row,col = input().split()
        puton(map,map_2,show_map,int(row),int(col),1)
    else:
        output = cnn_2(map_2)
        _,pred = output.max(1)
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
        puton(map,map_2,show_map,row,col,1)
        print(f"我下在{row}，{col}好啦！")
    draw_map(show_map)
    if ai_1 == 0:
        if check_gomoku_win(show_map) == 1:
            print("玩家2胜利！")
            break
    elif check_gomoku_win(show_map) == 1:
        print("ai1甘拜下风。")
        break