import numpy as np
samples_num = 3000    #样本数目
years       = 100    #计算的年份

years_array = np.zeros(years)
for i in range(years):
    years_array[i] = i