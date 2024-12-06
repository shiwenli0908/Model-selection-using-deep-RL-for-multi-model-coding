# -*- coding: utf-8 -*-
"""
Created on Fri May 31 21:06:38 2024

@author: shiwenli
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 假设这是你的8x8混淆矩阵
confusion_matrix = np.array([[ 94,  42,  33,   0,   3,   6,   0,  21],
                             [ 17, 239,   0,   0,  50,   0,   0,   2],
                             [  7,   0,   6,   0,   0,   0,   0,   0],
                             [  0,   0,   0,   0,   0,   0,   0,   1],
                             [  1,   0,   0,   0,   0,   0,   0,   1],
                             [  6,   0,   3,   0,   0,   3,   1,   0],
                             [ 11,   0,   3,   0,   0,   1,   3,   0],
                             [ 23,   4,  12,   0,   1,   2,   1,   3]]

)


cm_nx = np.array([[194, 103,   4,   4],
                [ 12, 132,   2,   0],
                [ 21,  38,  24,   7],
                [ 21,  18,   2,  18]]
)


cm_algo = np.array([[437,  0],
                    [ 50,  13]]

)


# 类别标签
labels = ['Non', 'Sinu', 'Poly-0', 'Poly-1', 'Poly-2', 'Poly-3', 'Poly-4', 'Poly-5']

labels_nx = ['10 bits', '20 bits', '30 bits', '40 bits']

labels_algo = ['DCT', 'DWT']

# 将混淆矩阵归一化
confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

cm_nx_normalized = cm_nx.astype('float') / cm_nx.sum(axis=1)[:, np.newaxis]

cm_algo_normalized = cm_algo.astype('float') / cm_algo.sum(axis=1)[:, np.newaxis]

# 将归一化后的混淆矩阵转换为DataFrame
#df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

df_cm = pd.DataFrame(cm_algo_normalized, index=labels_algo, columns=labels_algo)

# 创建一个图形实例
plt.figure(figsize=(10, 7))

# 使用Seaborn的heatmap函数绘制归一化后的混淆矩阵
heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True, annot_kws={"size": 10})

# 添加标题
plt.title('Confusion Matrix')

# 添加轴标签
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# 显示图形
plt.show()
