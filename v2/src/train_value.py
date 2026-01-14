import torch
from torch import nn
from torch import optim
import random
import matplotlib.pyplot as plt
from game_assitant import check_gomoku_win_train as check_gomoku_win
from game_assitant import puton
from net import trick_net
from net import value_net 

x_plot = []
y_plot = []
plt.figure(figsize=(18, 18), dpi=100, facecolor='white')
plt.ion()

# 损失曲线图绘制
def draw():
    plt.clf()
    plt.plot(x_plot, y_plot, 'b-', linewidth=1, label='train_loss', alpha=0.8)
    plt.title('train_loss_curve', fontsize=8, fontweight='bold')
    plt.xlabel('train_cnt', fontsize=7)
    plt.ylabel('loss', fontsize=7)
    plt.grid(True, alpha=0.3, linestyle='--', color='gray')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

# 全局变量初始化函数
def reset(map,show_map,all_games):
    map = torch.zeros(1,4,15,15)
    show_map = torch.zeros(1,1,15,15)
    #2.初始化先后手
    for i in range(15):
        for j in range(15):
            map[0,2,i,j] = -1
    all_games += 1
    return map,show_map,all_games

model_value = value_net()
state_dict_value = torch.load(f"model//value.pkl")
if isinstance(state_dict_value, nn.Module):
    model_value = state_dict_value
else:
    model_value.load_state_dict(state_dict_value)

cnn = model_value
all_cnt = 0
all_draw_cnt = 0

#0.载入AI模型并设置随机参数
model = trick_net()
state_dict = torch.load(f"model//trick.pkl")
if isinstance(state_dict, nn.Module):
    model = state_dict
else:
    model.load_state_dict(state_dict)
trick_net = model

orgin_explore_rate = 0.7

for train_epoch in range(10):
    #1.创建棋盘
    map = torch.zeros(1,4,15,15)
    show_map = torch.zeros(1,1,15,15)
    #2.初始化先后手
    for i in range(15):
        for j in range(15):
            map[0,2,i,j] = -1

    #3.落子更新
    # 数据格式安排
    current_state_list = []
    final_winner_list  = []
    # 全局变量
    all_games = 0
    game_steps = 0

    while len(final_winner_list)<10000:
        if random.random() >= orgin_explore_rate:
            output = trick_net(map)
            _,pred = output.max(1)
        else:
            pred = random.randint(0, 224)
        #解析步骤
        row = pred//15
        col = pred %15
            #落子
        puton(map,show_map,int(row),int(col),-1,trick_net,-1)
        for i in range(15):
            for j in range(15):
                map[0,2,i,j] = 1
        current_state_list.append(show_map)
        game_steps += 1
        if check_gomoku_win(show_map) == -1:
            # 制做标签
            # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
            for i in range(game_steps):
                final_winner_list.append(torch.tensor([-1+1]))
            print(len(current_state_list))
            # print(len(final_winner_list),end=' ')
            map,show_map,all_games\
            = reset(map,show_map,all_games)
            game_steps = 0
            # print(f"第{all_games}局黑子AI获胜")
            continue

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
            # 制做标签
            # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
            for i in range(game_steps):
                final_winner_list.append(torch.tensor([0+1]))
            print(len(current_state_list))
            # print(len(final_winner_list),end=' ')
            map,show_map,all_games\
            = reset(map,show_map,all_games)
            game_steps = 0
            # print(f"第{all_games}局draw!平局")
            continue

        ''''''
        if random.random() >= orgin_explore_rate:
            output = trick_net(map)
            _,pred = output.max(1)
        else:
            pred = random.randint(0, 224) 
        #解析步骤
        row = pred//15
        col = pred %15
        #落子
        puton(map,show_map,int(row),int(col),1,trick_net,1)
        for i in range(15):
            for j in range(15):
                map[0,2,i,j] = -1
        current_state_list.append(show_map)
        game_steps += 1
    
        if check_gomoku_win(show_map) == 1:
            # 制做标签
            # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
            for i in range(game_steps):
                final_winner_list.append(torch.tensor([1+1]))
            print(len(current_state_list))
            # print(len(final_winner_list),end=' ')
            map,show_map,all_games\
            = reset(map,show_map,all_games)
            game_steps = 0
            # print(f"第{all_games}局白子AI获胜")
            continue

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
            # 制做标签
            # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
            for i in range(game_steps):
                final_winner_list.append(torch.tensor([0+1]))
            # print(len(current_state_list),end=' ')
            print(len(final_winner_list))
            map,show_map,all_games\
            = reset(map,show_map,all_games)
            game_steps = 0
            # print(f"第{all_games}局draw!平局")
            continue

    #4. 正式训练
    current_state_list = torch.cat(current_state_list)
    final_winner_list  = torch.cat(final_winner_list)
    # 分割训练集和测试集
    divide_idx = int(0.8*len(current_state_list))
    max_train_datas = current_state_list [:divide_idx]
    max_train_labels= final_winner_list[:divide_idx]
    max_test_datas  = current_state_list[divide_idx+1:]
    max_test_labels = final_winner_list[divide_idx+1:]
    # 数据乱序
    indices_train = torch.randperm(max_train_datas.size(0))
    max_train_datas = max_train_datas[indices_train]
    max_train_labels = max_train_labels[indices_train]
    indices_test = torch.randperm(max_test_datas.size(0))
    max_test_datas = max_test_datas[indices_test]
    max_test_labels = max_test_labels[indices_test]

    # 分批次训练
    # 优化函数
    initial_lr = 0.0001
    optimizer = optim.Adam(cnn.parameters(), initial_lr)
    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        batch_size = 1024
        for batch_idx in range(0, len(max_train_datas), batch_size):
            end_idx = min(batch_idx + batch_size, len(max_train_datas))
            actual_batch_size = end_idx - batch_idx
            batch_data  = max_train_datas[batch_idx: end_idx]
            batch_label = max_train_labels[batch_idx: end_idx]
            # 前向传播
            output = cnn(batch_data)
            loss = loss_func(output, batch_label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 简单poly下降
            lr = initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            # 记录损失
            x_plot.append(all_cnt)
            all_cnt += actual_batch_size
            loss_value = min(loss.item(), 5)  # 限制最大值
            y_plot.append(loss_value)
            if len(y_plot) >= 100:
                avg_loss = sum(y_plot[-100:]) / 100
            else:
                avg_loss = sum(y_plot) / len(y_plot)
            print(f"Epoch: {epoch + 1}/100, "
                f"训练步数: {all_cnt%len(final_winner_list)}/{int(len(final_winner_list)*0.8)},"
                f"Loss: {loss_value:.4f}, "
                f"average loss: {avg_loss:.4f}, "
                f"learning rate: {lr:.8f}")
            all_draw_cnt += 1
            if all_draw_cnt % 30 == 0:
                draw()

        # 测试集
        all_test_cnt = 0
        loss_test = 0
        rightValue= 0
        for batch_idx in range(0, len(max_test_datas), batch_size):
            end_idx = min(batch_idx + batch_size, len(max_test_datas))
            actual_batch_size = end_idx - batch_idx
            batch_data  = max_test_datas[batch_idx: end_idx]
            batch_label = max_test_labels[batch_idx: end_idx]
            # 前向传播
            output = cnn(batch_data)
            loss_test += loss_func(output, batch_label)
            # 比对
            _,pred = output.max(1)
            rightValue += (pred==batch_label).sum().item()
            all_test_cnt += actual_batch_size
            print(f"Epoch: {epoch + 1}/100, "
                f"测试步数:"
                f"{all_test_cnt%len(final_winner_list)}/{len(final_winner_list)-int(len(final_winner_list) * 0.8)}"
                f",Loss: {loss_test:.4f},"
                f"accuracy: {rightValue/len(max_test_datas):.8f}")
        print(f"完成第{epoch + 1}个epoch")

    # 最终保存
    torch.save(cnn.state_dict(), "model/value.pkl")
    print(f"第{train_epoch+1}次训练完成！")

print("训练完成！")