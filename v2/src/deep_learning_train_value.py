from torch import optim
from game_assitant import check_gomoku_win_train as check_gomoku_win
from net import value_net
import torch
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


# 全局变量初始化函数
def reset(map, show_map, all_games):
    map = torch.zeros(1, 4, 15, 15)
    show_map = torch.zeros(1, 1, 15, 15)
    # 2.初始化先后手
    for i in range(15):
        for j in range(15):
            map[0, 2, i, j] = -1
    all_games += 1
    return map, show_map, all_games


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


# 训练函数
def train_model(cnn, current_state_list, final_winner_list, all_cnt, all_draw_cnt):
    # 4. 正式训练
    if len(current_state_list) == 0:
        print("当前批次没有训练数据，跳过训练")
        return all_cnt, all_draw_cnt

    current_state_tensor = torch.cat(current_state_list, dim=0)
    final_winner_tensor = torch.cat(final_winner_list, dim=0)

    # 确保两个张量长度一致
    min_length = min(current_state_tensor.size(0), final_winner_tensor.size(0))
    current_state_tensor = current_state_tensor[:min_length]
    final_winner_tensor = final_winner_tensor[:min_length]

    print(f"训练数据: {current_state_tensor.shape}, 标签: {final_winner_tensor.shape}")

    # 分割训练集和测试集
    divide_idx = int(0.8 * min_length)
    if divide_idx == 0:
        divide_idx = 1
    if divide_idx >= min_length:
        divide_idx = min_length - 1

    max_train_datas = current_state_tensor[:divide_idx]
    max_train_labels = final_winner_tensor[:divide_idx]
    max_test_datas = current_state_tensor[divide_idx:]
    max_test_labels = final_winner_tensor[divide_idx:]

    print(f"训练集大小: {len(max_train_datas)}, 测试集大小: {len(max_test_datas)}")

    if len(max_train_datas) == 0:
        print("训练集为空，跳过训练")
        return all_cnt, all_draw_cnt

    # 数据乱序
    indices_train = torch.randperm(max_train_datas.size(0))
    max_train_datas = max_train_datas[indices_train]
    max_train_labels = max_train_labels[indices_train]

    # 只有当测试集不为空时才乱序
    if len(max_test_datas) > 0:
        indices_test = torch.randperm(max_test_datas.size(0))
        max_test_datas = max_test_datas[indices_test]
        max_test_labels = max_test_labels[indices_test]

    # 分批次训练
    # 优化函数
    initial_lr = 0.001
    optimizer = optim.Adam(cnn.parameters(), initial_lr)
    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        batch_size = min(1024, len(max_train_datas))
        if batch_size == 0:
            print("批次大小为0，跳过训练")
            break

        for batch_idx in range(0, len(max_train_datas), batch_size):
            end_idx = min(batch_idx + batch_size, len(max_train_datas))
            actual_batch_size = end_idx - batch_idx
            batch_data = max_train_datas[batch_idx: end_idx]
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

            print(f"Epoch: {epoch + 1}/10, "
                  f"训练步数: {batch_idx}/{len(max_train_datas)}, "
                  f"Loss: {loss_value:.4f}, "
                  f"average loss: {avg_loss:.4f}, "
                  f"learning rate: {lr:.8f}")
            all_draw_cnt += 1
            if all_draw_cnt % 30 == 0:
                draw()

        # 测试集（如果测试集不为空）
        if len(max_test_datas) > 0:
            all_test_cnt = 0
            loss_test = 0
            rightValue = 0
            test_batch_size = min(1024, len(max_test_datas))

            for batch_idx in range(0, len(max_test_datas), test_batch_size):
                end_idx = min(batch_idx + test_batch_size, len(max_test_datas))
                actual_batch_size = end_idx - batch_idx
                batch_data = max_test_datas[batch_idx: end_idx]
                batch_label = max_test_labels[batch_idx: end_idx]

                # 前向传播
                output = cnn(batch_data)
                loss_test += loss_func(output, batch_label).item()

                # 比对
                _, pred = output.max(1)
                rightValue += (pred == batch_label).sum().item()
                all_test_cnt += actual_batch_size

                print(f"Epoch: {epoch + 1}/10, "
                      f"测试步数: {batch_idx}/{len(max_test_datas)}, "
                      f"Loss: {loss_test / (batch_idx // test_batch_size + 1):.4f}, "
                      f"accuracy: {rightValue / all_test_cnt:.8f}")

        print(f"完成第{epoch + 1}个epoch")

    return all_cnt, all_draw_cnt


# 主函数
def main():
    # 初始化模型
    model_value = value_net()
    '''
    state_dict_value = torch.load(f"model//value.pkl")
    if isinstance(state_dict_value, nn.Module):
        model_value = state_dict_value
    else:
        model_value.load_state_dict(state_dict_value)
    '''
    cnn = model_value

    all_cnt = 0
    all_draw_cnt = 0
    train_epoch = 0

    # 获取所有谱文件
    all_files = []
    for i in range(1, 185):  # 假设有184个文件
        path = f'data\\data_{i}.txt'
        if os.path.exists(path):
            all_files.append((i, path))

    # 每x个谱训练一次
    batch_size = 184
    num_batches = len(all_files) // batch_size + (1 if len(all_files) % batch_size > 0 else 0)

    for batch_idx in range(num_batches):
        print(f"\n=== 处理第{batch_idx + 1}/{num_batches}批谱文件 ===")

        # 获取当前批次的文件
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_files))
        batch_files = all_files[start_idx:end_idx]

        # 处理当前批次的谱文件
        batch_labels = []

        for file_idx, (index, path) in enumerate(batch_files):
            print(f"加载谱文件 {index}/{len(all_files)}: {path}")

            all_train_labels = []
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip().split()
                        if len(line) < 2:  # 至少要有2步才构成一个训练样本
                            continue

                        # 每局棋重置棋盘
                        all_label = []

                        # 处理每一步并记录
                        for move_str in line:
                            move = parse_move(move_str)
                            if move is None:
                                continue
                            row, col = move

                            # 更新标签
                            label = torch.tensor([row * 15 + col], dtype=torch.long)
                            all_label.append(label)

                        if len(all_label) == 0:
                            continue
                        else:
                            # 将列表转换为张量
                            train_label = torch.cat(all_label, dim=0)
                            all_train_labels.append(train_label)

                    if len(all_train_labels) == 0:
                        print(f"文件 {path} 没有有效数据")
                        continue

                    # 将列表转换为张量
                    labels = torch.cat(all_train_labels, dim=0)
                    batch_labels.append(labels)
                    print(f"文件 {path} 加载完成，包含 {len(labels)} 步")

            except Exception as e:
                print(f"处理文件 {path} 时出错: {e}")
                continue

        if len(batch_labels) == 0:
            print(f"第{batch_idx + 1}批没有有效数据，跳过")
            continue

        # 合并当前批次的标签
        max_labels = torch.cat(batch_labels, dim=0)
        print(f"当前批次共有 {len(max_labels)} 步")

        # 使用当前批次的标签进行模拟对局
        print(f"开始模拟对局...")

        # 1.创建棋盘
        map = torch.zeros(1, 4, 15, 15)
        show_map = torch.zeros(1, 1, 15, 15)
        # 2.初始化先后手
        for i in range(15):
            for j in range(15):
                map[0, 2, i, j] = -1

        # 3.落子更新
        # 数据格式安排
        current_state_list = []
        final_winner_list = []
        # 全局变量
        all_games = 0
        game_steps = 0

        role = -1
        step_count = 0
        for pred in max_labels:
            # 解析步骤
            row = pred // 15
            col = pred % 15

            # 落子
            show_map[0, 0, row, col] = role
            current_state_list.append(show_map.clone())
            game_steps += 1
            step_count += 1

            if check_gomoku_win(show_map) == role:
                print(f"已模拟对局数+1！现在已经完成{all_games+1}局")
                # 制做标签
                # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
                for i in range(game_steps):
                    final_winner_list.append(torch.tensor([role + 1]))

                map, show_map, all_games = reset(map, show_map, all_games)
                game_steps = 0
                role = -1
                continue

            is_draw = 1
            if check_gomoku_win(show_map) == 0:
                for i in range(15):
                    for j in range(15):
                        if show_map[0, 0, i, j] == 0:
                            is_draw = 0
                            break
                    if is_draw == 0:
                        break

            if is_draw == 1:
                # 制做标签
                # 规定-1映射到标签0，1映射到标签2，0映射到标签1（均+1）
                for i in range(game_steps):
                    final_winner_list.append(torch.tensor([0 + 1]))

                map, show_map, all_games = reset(map, show_map, all_games)
                game_steps = 0
                role = -1
                continue

            role = -role

            if step_count % 10000 == 0:
                print(f"模拟进度: {step_count}/{len(max_labels)}")

        print(f"模拟对局完成，生成{len(current_state_list)}个训练样本，{len(final_winner_list)}个标签")

        # 训练模型
        print(f"开始第{train_epoch + 1}次训练...")
        all_cnt, all_draw_cnt = train_model(cnn, current_state_list, final_winner_list, all_cnt, all_draw_cnt)

        # 保存模型
        print(f"第{batch_idx + 1}批谱训练完成！")

        train_epoch += 1

    # 最终保存
    torch.save(cnn.state_dict(), "model/value.pkl")
    print("所有谱训练完成！")


if __name__ == "__main__":
    main()