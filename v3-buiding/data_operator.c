/**
 * 我用我最熟悉的C语言完成基础数据解析工作
 * 解析目标：
 *  所有棋谱文件，逐局解析，并将解析后所得的数据写入新的文件（统一以preData_%d命名）
 */
/**
 * 解析规则：
 *      col = 读到的字符 - 'a'
 *      row = 读到的数字 - 1
 *      place = row * 15 + col
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG 1
#define MAX_LINE 184
#define PARSE_MOVE(ch, num) ((num - 1) * 15 + (ch - 'a'))

int main(void)
{
    // 存放各个文件的指针
    FILE * files[2][MAX_LINE] = { NULL };
    FILE ** read  = &files[0][0];
    FILE ** write = &files[1][0];
    int write_cnt = 0;

    // 统计总局数和最大局长
    int totalGames = 0;
    int maxMoves = 0;
    int i = 1;
    while ( i<=MAX_LINE ) {
        // 打开文件
        char filename[256];
        sprintf(filename, "data\\raw\\train_data_2\\%d.txt", i);
        if ((files[0][i-1] = fopen(filename, "r")) == NULL) {
            printf("Cannot open file %s\n", filename);
            exit(0);
        }

        // 逐行读取
        char line[1024];
        while (fgets(line, sizeof(line), files[0][i-1]) != NULL) {
            if (strlen(line) <= 1) continue; // 跳过无效行
            for (int count = 0; count < 128; count++) {
                // 出现字母表明坐标开始
                char *index = line;
                int moves = 0;
                while (*index) {
                    if (*index >= 'a' && *index <= 'z') {
                        moves++;
                        // 跳过该坐标
                        index++;
                        if (*index >= '0' && *index <= '9') 
                            index++;
                        if (*index >= '0' && *index <= '9') 
                            index++;
                    } else {
                        index++;
                    }
                } if (moves > maxMoves) maxMoves = moves;
            }
            totalGames++;
        }
        fseek(files[0][i-1], 0, SEEK_SET);
        i++;
    }

    // 平均每个文件的局数，舍弃余数
    int avgGames = totalGames / MAX_LINE;
    printf("Total Games: %d, Average games per file: %d\n",\
         totalGames, avgGames);

    i = 1;
    // 打开写入文件
    while ( i<=MAX_LINE ) {
        char filename[256];
        sprintf(filename, "data\\processed\\proData_%d.txt", i);
        if ((files[1][i-1] = fopen(filename, "w+")) == NULL) {
            printf("Cannot create file %s\n", filename);
            exit(0);
        }  
        i++;      
    }

    // 解析与写入
    char line[1024];
    while ( 1 ) {
        if (fgets(line, sizeof(line), *read) == NULL) {
            // 读取失败，检查是否需要切换文件
            if (read < &files[0][MAX_LINE - 1]) {
                FILE * prev = *read;
                read++;
                fclose(prev);
                continue;
            } else break;
        }

        // 解析单局游戏
        if (strlen(line) <= 1) continue; // 跳过无效行
        char *index = line;
        int moves_cnt = 0;
        while (*index) {
            // 出现字母表明坐标开始
            if (*index >= 'a' && *index <= 'z') {
                char ch = *index++;
                int num = *index++ - '0';
                if (*index >= '0' && *index <= '9') 
                    num = num* 10 + (*index++ - '0');
                fprintf(*write, "%d ", PARSE_MOVE(ch, num));
                moves_cnt++;
            } else {
                index++;
            }
        } if (moves_cnt < maxMoves) {
            // 填补空位
            for (int k = moves_cnt; k < maxMoves; k++) {
                fprintf(*write, "-1 ");
            }
        }

        if (++write_cnt == avgGames) {
            // 满栈，归0
            if ( write < &files[1][MAX_LINE - 1]) {
                int number = write - &files[1][0] + 1;
                printf("Finished writing file %d.\n", number);
                FILE * prev = *write;
                write++;
                fclose(prev);
                write_cnt = 0;
            } else {
                printf("Finished writing all files.\n");
                break;
            }
        } else {
            fprintf(*write, "\n");
        }
    }

    // 打印关键数据
    printf("Total games: %d\n", totalGames);
    printf("Average games per file: %d\n", avgGames);
    printf("Max Moves in a game: %d\n", maxMoves);
    return 0;
}