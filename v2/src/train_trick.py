import torch
from fontTools.misc.symfont import GreenPen

from net import trick_net
import matplotlib.pyplot as plt
import os


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


# 解析步数
def parse_move(move_str):
    try:
        if len(move_str) < 2:
            return None
        col = ord(move_str[0].lower()) - ord('a')
        row = int(move_str[1:]) - 1
        if 0 <= row < 15 and 0 <= col < 15:
            return row, col
        return None
    except:
        return None


# 可视化棋盘
def draw_map(map):
    print("======================================")
    for row in map[0][0]:
        for col in row:
            if int(col.item()) == 1:
                print("O", end='   ')
            elif int(col.item()) == -1:
                print("X", end='   ')
            elif int(col.item()) == 0:
                print(".", end='   ')
            elif int(col.item()) == 2:
                print("\033[32mO\033[0m", end='   ')
            elif int(col.item()) == -2:
                print("\033[32mX\033[0m", end='   ')
            else:
                print("?", end='   ')
        print('\n')
    print("======================================")
    plt.pause(2)


# 初始化模型
cnn = trick_net()
loss_func = torch.nn.CrossEntropyLoss()
initial_lr = 1e-2
defend_weight = 1.05
optimizer = torch.optim.Adam(cnn.parameters(), initial_lr)
all_cnt = 0
max_epochs = 100
max_datas = []
max_labels = []
all_draw_cnt = 0
all_steps = 0
role1 = torch.zeros(15,15)
role2 = torch.zeros(15,15)
for i in range(15):
    for j in range(15):
        role1[i,j] = 1
for i in range(15):
    for j in range(15):
        role2[i,j] = -1

for epoch in range(max_epochs):
    all_draw_cnt = 0
    # 数据处理并记录
    if len(max_datas) == 0:
        for index in range(184):
            # 初始化张量
            path = f'data\\data_{index + 1}.txt'
            all_train_data = []
            all_train_labels = []
            if not os.path.exists(path):
                print(f"文件不存在: {path}")
                continue
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    for line_num, line in enumerate(file):
                        line = line.strip().split()
                        if len(line) < 2:  # 至少要有2步才构成一个训练样本
                            continue
                        # 每局棋重置棋盘
                        board    = torch.zeros(1, 1, 15, 15)
                        c3_board = torch.zeros(1, 4, 15, 15)
                        all_label = []
                        all_board = []
                        last_row = -1
                        last_col = -1
                        # 处理每一步并记录
                        for move_idx, move_str in enumerate(line):
                            move = parse_move(move_str)
                            if move is None:
                                continue
                            row, col = move
                            # 更新标签
                            label = torch.tensor([row * 15 + col], dtype=torch.long)
                            all_label.append(label)
                            # 保存当下棋盘状态
                            all_board.append(c3_board.clone())
                            # 更新棋盘状态
                            if move_idx % 2 == 0:  # 对方回合（黑棋）
                                board[0, 0, row, col] = -1
                                c3_board[0, 1, row, col] = -1
                                c3_board[0, 3 ,row, col] = -2
                                c3_board[0][2] = role1
                                all_steps += 1
                                # 弱化上一手
                                if last_row >= 0:
                                    c3_board[0,3,last_row,last_col] = 1
                                    last_row = row
                                    last_col = col
                            else:  # 己方回合（白棋）
                                board[0, 0, row, col] = 1
                                c3_board[0, 0, row, col] = 1
                                c3_board[0, 3, row, col] = 2
                                c3_board[0][2] = role2
                                all_steps += 1
                                # 弱化上一手
                                if last_row >= 0:
                                    c3_board[0,3,last_row,last_col] = -1
                                    last_row = row
                                    last_col = col
                            # draw_map(board)
                            if all_steps % 10000 == 0 or all_steps >= 994385:
                                print(f"数据加载中...{min(all_steps*100/994385,100.00):.2f}%")
                        # print("分界=============================")
                        if len(all_board) == 0 or len(all_label) == 0:
                            print("段错误")
                            continue
                        else:
                            # 将列表转换为张量
                            train_board = torch.cat(all_board, dim=0)
                            train_label = torch.cat(all_label, dim=0)
                            all_train_data.append(train_board)
                            all_train_labels.append(train_label)
                    if len(all_train_data) == 0 or len(all_train_labels) == 0:
                        print("段错误")
                        continue

                    # 将列表转换为张量
                    datas = torch.cat(all_train_data, dim=0)
                    labels = torch.cat(all_train_labels, dim=0)
                    max_datas.append(datas)
                    max_labels.append(labels)

            except Exception as e:
                print(f"处理文件 {path} 时出错: {e}")
                continue

    # 将列表转化为张量
    if len(max_datas) == 0 or len(max_labels) == 0:
        print("段错误")
        continue
    if type(max_datas) != torch.Tensor:
        max_datas  = torch.cat(max_datas, dim=0)
        max_labels = torch.cat(max_labels, dim=0)
        print(max_datas.shape)
        print(max_labels.shape)
        # check_data(max_datas,max_labels)
        # 张量乱序
        indices = torch.randperm(max_datas.size(0))
        max_datas  = max_datas[indices]
        max_labels = max_labels[indices]
        # 数据预处理
        print("张量预处理...")
        for data in max_datas:
            if data[2,0,0] == 1: #白子
                data[1] *= defend_weight  #增强敌子
            elif data[2,0,0] == -1: #黑子
                data[0] *= defend_weight  #增强敌子
    # print(f"数据形状: {datas.shape}, 标签形状: {labels.shape}")

    # 分割训练集和测试集
    divide_idx = int(0.8*len(max_datas))
    max_train_datas = max_datas [:divide_idx]
    max_train_labels= max_labels[:divide_idx]
    max_test_datas  = max_datas[divide_idx+1:]
    max_test_labels = max_labels[divide_idx+1:]
    # 再次乱序
    indices_train = torch.randperm(max_train_datas.size(0))
    max_train_datas = max_train_datas[indices_train]
    max_train_labels = max_train_labels[indices_train]
    indices_test = torch.randperm(max_test_datas.size(0))
    max_test_datas = max_test_datas[indices_test]
    max_test_labels = max_test_labels[indices_test]
    # 分批次训练
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
        lr = max(initial_lr * (1 - all_cnt / int(994385 * max_epochs * 0.8 * 0.08)),1e-8)  # 总步数*自定义参量
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
              f"训练步数: {all_cnt%795508}/{int(994385*0.8)}, Loss: {loss_value:.4f}, "
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
              f"测试步数: {all_test_cnt%(994385-795508)}/{994385-int(994385 * 0.8)}, Loss: {loss_test:.4f}, "
              f"accuracy: {rightValue/len(max_test_datas):.8f}")

    # 定期保存模型
    torch.save(cnn.state_dict(), f"model/game_next_model_ckpoint_{epoch + 1}.pkl")
    print(f"完成第{epoch + 1}个epoch")

# 最终保存
torch.save(cnn.state_dict(), "model/game_next_model_final.pkl")
plt.ioff()
plt.show()
print("训练完成！")