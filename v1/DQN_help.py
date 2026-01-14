import torch
"""
目标：
1.单子——无奖励
2.二子棋形——两边均活型：每一个方向+0.3
3.三子棋形——两边活型：每个方向＋0.5
4.四子棋形——单边活型+0.4,两边活型+0.9
5.获胜、失败另谈
6.堵住对方二子两边均活型：+0.2
7.堵住三子棋形——两边活型+0.6
8.隔断对方四子两边活型：+0.8
9.阻止获胜：+1.0
"""
'''
def draw_map(map):
    print("======================================")
    print(" 0 1 2 3 4 5 6 7 8 9 1011121314")
    for t_index,t_row in enumerate(map[0][0]):
        if t_index < 10:
            print(t_index,end=' ')
        else:
            print(t_index,end='')
        for index,t_col in enumerate(t_row):
            if int(t_col.item()) == 1:
                print("O", end=' ')
            elif int(t_col.item()) == -1:
                print("X", end=' ')
            elif int(t_col.item()) == 0:
                print(".", end=' ')
            elif int(t_col.item()) == 3:
                print("\033[31mO\033[0m", end=' ')
            elif int(t_col.item()) == -3:
                print("\033[31mX\033[0m", end=' ')
            elif int(t_col.item()) == 2:
                print("\033[32mO\033[0m", end=' ')
            elif int(t_col.item()) == -2:
                print("\033[32mX\033[0m", end=' ')
            else:
                print("?", end=' ')
            if index == 14:
                print('\n',end='')
    print("======================================")
'''
# 判断函数
def score_attack(state,action,role):
    score = 0
    # 1.解析动作，读取棋面
    init_state = state.clone()
    i = action//15
    j = action%15
    # 2.八方延申，直到没有自己的棋
    n_col = 1
    col_live = 0
    n_row = 1
    row_live = 0
    n_main_angle = 1
    main_angle_live = 0
    n_left_angle = 1
    left_angle_live = 0
    # （1）上
    if i-1>=0 and init_state[0,0,i-1,j] == role:
        n_col += 1
        if i-2>=0 and init_state[0,0,i-2,j] == role:
            n_col += 1
            if i-3>=0 and init_state[0,0,i-3,j] == role:
                n_col += 1
                if i-4>=0 and init_state[0,0,i-4,j] == role:
                    n_col += 1
                elif i-4>=0 and init_state[0,0,i-4,j] == 0:
                    col_live += 1
            elif i-3>=0 and init_state[0,0,i-3,j] == 0:
                col_live += 1
        elif i-2>=0 and init_state[0,0,i-2,j] == 0:
                col_live += 1
    elif i-1>=0 and init_state[0,0,i-1,j] == 0:
        col_live += 1
    # （2）下
    if i+1<=14 and init_state[0,0,i+1,j] == role:
        n_col += 1
        if i+2<=14 and init_state[0,0,i+2,j] == role:
            n_col += 1
            if i+3<=14 and init_state[0,0,i+3,j] == role:
                n_col += 1
                if i+4<=14 and init_state[0,0,i+4,j] == role:
                    n_col += 1
                elif i+4<=14 and init_state[0,0,i+4,j] == 0:
                    col_live += 1
            elif i+3<=14 and init_state[0,0,i+3,j] == 0:
                col_live += 1
        elif i+2<=14 and init_state[0,0,i+2,j] == 0:
                col_live += 1
    elif i+1<=14 and init_state[0,0,i+1,j] == 0:
        col_live += 1
    # （3）左
    if j-1>=0 and init_state[0,0,i,j-1] == role:
        n_row += 1
        if j-2>=0 and init_state[0,0,i,j-2] == role:
            n_row += 1
            if j-3>=0 and init_state[0,0,i,j-3] == role:
                n_row += 1
                if j-4>=0 and init_state[0,0,i,j-4] == role:
                    n_row += 1
                elif j-4>=0 and init_state[0,0,i,j-4] == 0:
                    row_live += 1
            elif j-3>=0 and init_state[0,0,i,j-3] == 0:
                row_live += 1
        elif j-2>=0 and init_state[0,0,i,j-2] == 0:
            row_live += 1
    elif j-1>=0 and init_state[0,0,i,j-1] == 0:
        row_live += 1
    # （4）右
    if j+1<=14 and init_state[0,0,i,j+1] == role:
        n_row += 1
        if j+2<=14 and init_state[0,0,i,j+2] == role:
            n_row += 1
            if j+3<=14 and init_state[0,0,i,j+3] == role:
                n_row += 1
                if j+4<=14 and init_state[0,0,i,j+4] == role:
                    n_row += 1
                elif j+4<=14 and init_state[0,0,i,j+4] == 0:
                    row_live += 1
            elif j+3<=14 and init_state[0,0,i,j+3] == 0:
                row_live += 1
        elif j+2<=14 and init_state[0,0,i,j+2] == 0:
            row_live += 1
    elif j+1<=14 and init_state[0,0,i,j+1] == 0:
        row_live += 1
    # （5）左上
    if i-1>=0 and j-1>=0 and init_state[0,0,i-1,j-1] == role:
        n_main_angle += 1
        if i-2>=0 and j-2>=0 and init_state[0,0,i-2,j-2] == role:
            n_main_angle += 1
            if i-3>=0 and j-3>=0 and init_state[0,0,i-3,j-3] == role:
                n_main_angle += 1
                if i-4>=0 and j-4>=0 and init_state[0,0,i-4,j-4] == role:
                    n_main_angle += 1
                elif i-4>=0 and j-4>=0 and init_state[0,0,i-4,j-4] == 0:
                    main_angle_live += 1
            elif i-3>=0 and j-3>=0 and init_state[0,0,i-3,j-3] == 0:
                main_angle_live += 1
        elif i-2>=0 and j-2>=0 and init_state[0,0,i-2,j-2] == 0:
            main_angle_live += 1
    elif i-1>=0 and j-1>=0 and init_state[0,0,i-1,j-1] == 0:
        main_angle_live += 1
    # （6）右下
    if i+1<=14 and j+1<=14 and init_state[0,0,i+1,j+1] == role:
        n_main_angle += 1
        if i+2<=14 and j+2<=14 and init_state[0,0,i+2,j+2] == role:
            n_main_angle += 1
            if i+3<=14 and j+3<=14 and init_state[0,0,i+3,j+3] == role:
                n_main_angle += 1
                if i+4<=14 and j+4<=14 and init_state[0,0,i+4,j+4] == role:
                    n_main_angle += 1
                elif i+4<=14 and j+4<=14 and init_state[0,0,i+4,j+4] == 0:
                    main_angle_live += 1
            elif i+3<=14 and j+3<=14 and init_state[0,0,i+3,j+3] == 0:
                main_angle_live += 1
        elif i+2<=14 and j+2<=14 and init_state[0,0,i+2,j+2] == 0:
            main_angle_live += 1
    elif i+1<=14 and j+1<=14 and init_state[0,0,i+1,j+1] == 0:
        main_angle_live += 1
    # （7）左下
    if i+1<=14 and j-1>=0 and init_state[0,0,i+1,j-1] == role:
        n_left_angle += 1
        if i+2<=14 and j-2>=0 and init_state[0,0,i+2,j-2] == role:
            n_left_angle += 1
            if i+3<=14 and j-3>=0 and init_state[0,0,i+3,j-3] == role:
                n_left_angle += 1
                if i+4<=14 and j-4>=0 and init_state[0,0,i+4,j-4] == role:
                    n_left_angle += 1
                elif i+4<=14 and j-4>=0 and init_state[0,0,i+4,j-4] == 0:
                    left_angle_live += 1
            elif i+3<=14 and j-3>=0 and init_state[0,0,i+3,j-3] == 0:
                left_angle_live += 1
        elif i+2<=14 and j-2>=0 and init_state[0,0,i+2,j-2] == 0:
            left_angle_live += 1
    elif i+1<=14 and j-1>=0 and init_state[0,0,i+1,j-1] == 0:
        left_angle_live += 1
    # （8）右上
    if j+1<=14 and i-1>=0 and init_state[0,0,i-1,j+1] == role:
        n_left_angle += 1
        if j+2<=14 and i-2>=0 and init_state[0,0,i-2,j+2] == role:
            n_left_angle += 1
            if j+3<=14 and i-3>=0 and init_state[0,0,i-3,j+3] == role:
                n_left_angle += 1
                if j+4<=14 and i-4>=0 and init_state[0,0,i-4,j+4] == role:
                    n_left_angle += 1
                elif j+4<=14 and i-4>=0 and init_state[0,0,i-4,j+4] == 0:
                    left_angle_live += 1
            elif j+3<=14 and i-3>=0 and init_state[0,0,i-3,j+3] == 0:
                left_angle_live += 1
        elif j+2<=14 and i-2>=0 and init_state[0,0,i-2,j+2] == 0:
            left_angle_live += 1
    elif j+1<=14 and i-1>=0 and init_state[0,0,i-1,j+1] == 0:
        left_angle_live += 1
    # 3.判断死活型
    for index in range(2,5):
        if n_col == index:
            if col_live == 1:
                # print(f"竖向{index}子单边活型")
                if index == 4:
                    score = max(score, 0.4)
            elif col_live == 2:
                # print(f"竖向{index}子双边活型")
                if index == 2:
                    score = max(score, 0.3)
                if index == 3:
                    score = max(score, 0.5)
                if index == 4:
                    score = max(score, 0.9)
            else:
                0# print(f"竖向{index}子死型")
        if n_row == index:
            if row_live == 1:
                #print(f"横向{index}子单边活型")
                if index == 4:
                    score = max(score, 0.4)
            elif row_live == 2:
                #print(f"横向{index}子双边活型")
                if index == 2:
                    score = max(score, 0.3)
                if index == 3:
                    score = max(score, 0.5)
                if index == 4:
                    score = max(score, 0.9)
            else:
                0#print(f"横向{index}子死型")
        if n_main_angle == index:
            if main_angle_live == 1:
                #print(f"主对角{index}子单边活型")
                if index == 4:
                    score = max(score, 0.4)
            elif main_angle_live == 2:
                #print(f"主对角{index}子双边活型")
                if index == 2:
                    score = max(score, 0.3)
                if index == 3:
                    score = max(score, 0.5)
                if index == 4:
                    score = max(score, 0.9)
            else:
                0#print(f"主对角{index}子死型")
        if n_left_angle == index:
            if left_angle_live == 1:
                #print(f"副对角{index}子单边活型")
                if index == 4:
                    score = max(score, 0.4)
            elif left_angle_live == 2:
                #print(f"副对角{index}子双边活型")
                if index == 2:
                    score = max(score, 0.3)
                if index == 3:
                    score = max(score, 0.5)
                if index == 4:
                    score = max(score, 0.9)
            else:
                0#print(f"副对角{index}子死型")
    if n_col >= 5:
        0#print("竖向赢棋")
    if n_row >= 5:
        0#print("横向赢棋")
    if n_main_angle >= 5:
        0#print("主对角赢棋")
    if n_left_angle >= 5:
        0#print("副对角赢棋")
    return score

def score_defend(state,action,input_role):
    score = 0
    init_state = state.clone()
    i = action // 15
    j = action % 15
    role = -input_role
    n_col = 1
    col_live = 0
    n_row = 1
    row_live = 0
    n_main_angle = 1
    main_angle_live = 0
    n_left_angle = 1
    left_angle_live = 0
    # （1）上
    if i-1>=0 and init_state[0,0,i-1,j] == role:
        n_col += 1
        if i-2>=0 and init_state[0,0,i-2,j] == role:
            n_col += 1
            if i-3>=0 and init_state[0,0,i-3,j] == role:
                n_col += 1
                if i-4>=0 and init_state[0,0,i-4,j] == role:
                    n_col += 1
                elif i-4>=0 and init_state[0,0,i-4,j] == 0:
                    col_live += 1
            elif i-3>=0 and init_state[0,0,i-3,j] == 0:
                col_live += 1
        elif i-2>=0 and init_state[0,0,i-2,j] == 0:
                col_live += 1
    elif i-1>=0 and init_state[0,0,i-1,j] == 0:
        col_live += 1
    # （2）下
    if i+1<=14 and init_state[0,0,i+1,j] == role:
        n_col += 1
        if i+2<=14 and init_state[0,0,i+2,j] == role:
            n_col += 1
            if i+3<=14 and init_state[0,0,i+3,j] == role:
                n_col += 1
                if i+4<=14 and init_state[0,0,i+4,j] == role:
                    n_col += 1
                elif i+4<=14 and init_state[0,0,i+4,j] == 0:
                    col_live += 1
            elif i+3<=14 and init_state[0,0,i+3,j] == 0:
                col_live += 1
        elif i+2<=14 and init_state[0,0,i+2,j] == 0:
                col_live += 1
    elif i+1<=14 and init_state[0,0,i+1,j] == 0:
        col_live += 1
    # （3）左
    if j-1>=0 and init_state[0,0,i,j-1] == role:
        n_row += 1
        if j-2>=0 and init_state[0,0,i,j-2] == role:
            n_row += 1
            if j-3>=0 and init_state[0,0,i,j-3] == role:
                n_row += 1
                if j-4>=0 and init_state[0,0,i,j-4] == role:
                    n_row += 1
                elif j-4>=0 and init_state[0,0,i,j-4] == 0:
                    row_live += 1
            elif j-3>=0 and init_state[0,0,i,j-3] == 0:
                row_live += 1
        elif j-2>=0 and init_state[0,0,i,j-2] == 0:
            row_live += 1
    elif j-1>=0 and init_state[0,0,i,j-1] == 0:
        row_live += 1
    # （4）右
    if j+1<=14 and init_state[0,0,i,j+1] == role:
        n_row += 1
        if j+2<=14 and init_state[0,0,i,j+2] == role:
            n_row += 1
            if j+3<=14 and init_state[0,0,i,j+3] == role:
                n_row += 1
                if j+4<=14 and init_state[0,0,i,j+4] == role:
                    n_row += 1
                elif j+4<=14 and init_state[0,0,i,j+4] == 0:
                    row_live += 1
            elif j+3<=14 and init_state[0,0,i,j+3] == 0:
                row_live += 1
        elif j+2<=14 and init_state[0,0,i,j+2] == 0:
            row_live += 1
    elif j+1<=14 and init_state[0,0,i,j+1] == 0:
        row_live += 1
    # （5）左上
    if i-1>=0 and j-1>=0 and init_state[0,0,i-1,j-1] == role:
        n_main_angle += 1
        if i-2>=0 and j-2>=0 and init_state[0,0,i-2,j-2] == role:
            n_main_angle += 1
            if i-3>=0 and j-3>=0 and init_state[0,0,i-3,j-3] == role:
                n_main_angle += 1
                if i-4>=0 and j-4>=0 and init_state[0,0,i-4,j-4] == role:
                    n_main_angle += 1
                elif i-4>=0 and j-4>=0 and init_state[0,0,i-4,j-4] == 0:
                    main_angle_live += 1
            elif i-3>=0 and j-3>=0 and init_state[0,0,i-3,j-3] == 0:
                main_angle_live += 1
        elif i-2>=0 and j-2>=0 and init_state[0,0,i-2,j-2] == 0:
            main_angle_live += 1
    elif i-1>=0 and j-1>=0 and init_state[0,0,i-1,j-1] == 0:
        main_angle_live += 1
    # （6）右下
    if i+1<=14 and j+1<=14 and init_state[0,0,i+1,j+1] == role:
        n_main_angle += 1
        if i+2<=14 and j+2<=14 and init_state[0,0,i+2,j+2] == role:
            n_main_angle += 1
            if i+3<=14 and j+3<=14 and init_state[0,0,i+3,j+3] == role:
                n_main_angle += 1
                if i+4<=14 and j+4<=14 and init_state[0,0,i+4,j+4] == role:
                    n_main_angle += 1
                elif i+4<=14 and j+4<=14 and init_state[0,0,i+4,j+4] == 0:
                    main_angle_live += 1
            elif i+3<=14 and j+3<=14 and init_state[0,0,i+3,j+3] == 0:
                main_angle_live += 1
        elif i+2<=14 and j+2<=14 and init_state[0,0,i+2,j+2] == 0:
            main_angle_live += 1
    elif i+1<=14 and j+1<=14 and init_state[0,0,i+1,j+1] == 0:
        main_angle_live += 1
    # （7）左下
    if i+1<=14 and j-1>=0 and init_state[0,0,i+1,j-1] == role:
        n_left_angle += 1
        if i+2<=14 and j-2>=0 and init_state[0,0,i+2,j-2] == role:
            n_left_angle += 1
            if i+3<=14 and j-3>=0 and init_state[0,0,i+3,j-3] == role:
                n_left_angle += 1
                if i+4<=14 and j-4>=0 and init_state[0,0,i+4,j-4] == role:
                    n_left_angle += 1
                elif i+4<=14 and j-4>=0 and init_state[0,0,i+4,j-4] == 0:
                    left_angle_live += 1
            elif i+3<=14 and j-3>=0 and init_state[0,0,i+3,j-3] == 0:
                left_angle_live += 1
        elif i+2<=14 and j-2>=0 and init_state[0,0,i+2,j-2] == 0:
            left_angle_live += 1
    elif i+1<=14 and j-1>=0 and init_state[0,0,i+1,j-1] == 0:
        left_angle_live += 1
    # （8）右上
    if j+1<=14 and i-1>=0 and init_state[0,0,i-1,j+1] == role:
        n_left_angle += 1
        if j+2<=14 and i-2>=0 and init_state[0,0,i-2,j+2] == role:
            n_left_angle += 1
            if j+3<=14 and i-3>=0 and init_state[0,0,i-3,j+3] == role:
                n_left_angle += 1
                if j+4<=14 and i-4>=0 and init_state[0,0,i-4,j+4] == role:
                    n_left_angle += 1
                elif j+4<=14 and i-4>=0 and init_state[0,0,i-4,j+4] == 0:
                    left_angle_live += 1
            elif j+3<=14 and i-3>=0 and init_state[0,0,i-3,j+3] == 0:
                left_angle_live += 1
        elif j+2<=14 and i-2>=0 and init_state[0,0,i-2,j+2] == 0:
            left_angle_live += 1
    elif j+1<=14 and i-1>=0 and init_state[0,0,i-1,j+1] == 0:
        left_angle_live += 1
    # 3.判断死活型
    for index in range(2,5):
        if n_col == index:
            if col_live == 1:
                0#print(f"隔断竖向{index}子单边活型")
            elif col_live == 2:
                #print(f"隔断竖向{index}子双边活型")
                if index == 2:
                    score = max(score, 0.2)
                if index == 3:
                    score = max(score, 0.6)
                if index == 4:
                    score = max(score, 0.8)
            else:
                0#print(f"隔断竖向{index}子死型")
        if n_row == index:
            if row_live == 1:
                0#print(f"隔断横向{index}子单边活型")
            elif row_live == 2:
                #print(f"隔断横向{index}子双边活型")
                if index == 2:
                    score = max(score, 0.2)
                if index == 3:
                    score = max(score, 0.6)
                if index == 4:
                    score = max(score, 0.8)
            else:
                0#print(f"隔断横向{index}子死型")
        if n_main_angle == index:
            if main_angle_live == 1:
                0#print(f"隔断主对角{index}子单边活型")
            elif main_angle_live == 2:
                #print(f"隔断主对角{index}子双边活型")
                if index == 2:
                    score = max(score, 0.2)
                if index == 3:
                    score = max(score, 0.6)
                if index == 4:
                    score = max(score, 0.8)
            else:
                0#print(f"隔断主对角{index}子死型")
        if n_left_angle == index:
            if left_angle_live == 1:
                0#print(f"隔断副对角{index}子单边活型")
            elif left_angle_live == 2:
                #print(f"隔断副对角{index}子双边活型")
                if index == 2:
                    score = max(score, 0.2)
                if index == 3:
                    score = max(score, 0.6)
                if index == 4:
                    score = max(score, 0.8)
            else:
                0#print(f"隔断副对角{index}子死型")

    if n_col >= 5:
        #print("竖向阻止赢棋")
        score = max(score, 1.0)
    if n_row >= 5:
        #print("横向阻止赢棋")
        score = max(score, 1.0)
    if n_main_angle >= 5:
        #print("主对角阻止赢棋")
        score = max(score, 1.0)
    if n_left_angle >= 5:
        #print("副对角阻止赢棋")
        score = max(score, 1.0)
    # 4.赋分
    return score
'''
try_torsor = torch.zeros(1,1,15,15)
role = 1
for i in range(17):
    draw_map(try_torsor)
    input_i,input_j = map(int,input().split())
    print(input_i,input_j)
    try_torsor[0,0,input_i,input_j] = role
    input_action = 15*input_i+input_j
    score1 = score_attack(try_torsor,input_action,role)
    score2 = score_defend(try_torsor,input_action,role)
    print("得分：",score1+score2)
    role *= -1
'''