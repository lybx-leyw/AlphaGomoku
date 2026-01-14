import torch
import time
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

if __name__ == "__main__":

    test_board = torch.zeros(1, 1, 15, 15)
    test_board[0, 0, 7, 5:10] = -1
    start_time = time.time()
    for _ in range(1000):
        result = check_gomoku_win(test_board)
        print(result)
    print(f"优化版本执行1000次耗时: {time.time() - start_time:.4f}秒")

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