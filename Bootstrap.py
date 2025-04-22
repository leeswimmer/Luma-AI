import pandas as pd
from sklearn.utils import resample

# 假设数据存储在 'data.csv' 中
file_path = 'E:/papers/Xu_project1/code/Medicine LLM/3 Machine Learning_Prediction/medicine data/汇总初始数据/OA_tran原始.csv'

# 读取数据
df = pd.read_csv(file_path)

# 第3列是标签列，统计标签数量
label_column = 1  # 第3列，索引从0开始
label_counts = df.iloc[:, label_column].value_counts()

# 打印原始标签分布
print("原始标签数量分布：")
print(label_counts)

# 分离多数类和少数类
majority_class = df[df.iloc[:, label_column] == 0]
minority_class = df[df.iloc[:, label_column] == 1]

# 使用 Bootstrap 对少数类样本进行上采样
minority_upsampled = resample(
    minority_class,
    replace=True,  # 放回采样
    n_samples=len(majority_class),  # 上采样到多数类样本数量
    random_state=42
)

# 合并多数类和上采样的少数类
balanced_df = pd.concat([majority_class, minority_upsampled])

# 打印平衡后的标签分布
print("平衡后标签数量分布：")
print(balanced_df.iloc[:, label_column].value_counts())

# 如果需要，保存平衡后的数据到新文件
balanced_df.to_csv('E:/papers/Xu_project1/code/Medicine LLM/3 Machine Learning_Prediction/medicine data/汇总初始数据/OA_tran.csv', index=False)
print("平衡后的数据已保存为 balanced_data.csv")
