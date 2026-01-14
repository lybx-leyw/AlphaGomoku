#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

int main()
{
    SetConsoleOutputCP(65001);
    char path1[200];
    char path2[200];
    
    for (int i = 1; i <= 184; i++)
    {
        sprintf(path1, "GomokuGamesData\\train_data\\%d.txt", i);
        sprintf(path2, "GomokuGamesData\\train_data\\data_%d.txt", i);
        
        FILE* fp = fopen(path1, "r");
        if (!fp)
        {
            printf("Error: cannot open file %s\n", path1);
            continue; // 继续处理下一个文件
        }
        
        FILE* wp = fopen(path2, "w+");
        if (!wp)
        {
            printf("Error: cannot open file %s\n", path2);
            fclose(fp);
            continue; // 继续处理下一个文件
        }
        
        char line[500];
        int cnt = 0;
        
        while ((fgets(line, sizeof(line), fp)) != NULL) 
        {
            char t_line[800] = {0}; // 增大缓冲区并初始化
            char* pline = line;
            char* pline2 = t_line;
            
            while (*pline != '\0' && *pline != '\n')
            {
                if (*pline >= '0' && *pline <= '9')
                {
                    *pline2 = *pline;
                    pline2++;
                    
                    // 如果下一个字符不是数字，添加空格
                    if (*(pline + 1) < '0' || *(pline + 1) > '9')
                    {
                        *pline2 = ' ';
                        pline2++;
                    }
                }
                else
                {
                    *pline2 = *pline;
                    pline2++;
                }
                pline++;
            }
            *pline2 = '\0'; // 确保字符串结束
            
            fprintf(wp, "%s\n", t_line);
            printf("第%d个文件%d行copy完毕\n", i, ++cnt);
        }
        
        // 文件处理完成后再关闭
        fclose(fp);
        fclose(wp);
    }
    
    return 0;
}