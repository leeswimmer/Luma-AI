import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
import numpy as np

# 确保安装 pytorch-tabnet 和 openpyxl
# pip install pytorch-tabnet openpyxl

# 读取 Excel 表格数据
data_path = 'E:/Medicine LLM/2 important_25_factors/RA.xlsx'  # 替换为你的文件路径
data = pd.read_excel(data_path)

# 定义因变量和自变量
y = data.iloc[:, 0]  # 第三列为因变量
X = data.iloc[:, 3:42]  # 第4~45列为自变量

# 对因变量进行编码（如果是分类任务）
if y.dtype == 'object' or y.nunique() <= 10:  # 判断是否是分类问题
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# **修复：将目标变量转换为二维数组**
y = y.reshape(-1, 1) if len(y.shape) == 1 else y

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对自变量进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化 TabNet 模型
model = TabNetRegressor()  # 如果是分类问题，改用 TabNetClassifier

# 训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],  # 如果是分类任务，改为 ['accuracy']
    max_epochs=100,
    patience=10,
    batch_size=256,
    virtual_batch_size=128
)

# 提取特征重要性
feature_importances = model.feature_importances_

# 创建 DataFrame 来显示特征重要性
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 输出最重要的前 25 个自变量
top_25_features = feature_importance_df.head(25)
print("Top 25 important features:")
print(top_25_features)

# 保存结果到文件
output_path = 'E:/Medicine LLM/2 important_25_factors/keyfactors_RA.csv'  # 替换为保存路径
top_25_features.to_csv(output_path, index=False)



