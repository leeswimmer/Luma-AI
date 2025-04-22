import os

# 允许重复的 OpenMP 库加载
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def prepare_data(df):
    """
    准备数据，包括编码分类变量和分离特征
    """
    # 创建数据副本
    df_processed = df.copy()

    # 存储每列的缺失值位置
    missing_mask = df_processed.isna()

    # 对分类变量进行编码并临时填充缺失值
    encoders = {}
    for column in df_processed.columns:
        # 检查列是否包含非数值数据
        if df_processed[column].dtype == 'object' or df_processed[column].dtype == 'category':
            # 将所有值转换为字符串
            df_processed[column] = df_processed[column].astype(str)
            # 临时填充缺失值
            df_processed[column] = df_processed[column].fillna('MISSING')

            encoders[column] = LabelEncoder()
            df_processed[column] = encoders[column].fit_transform(df_processed[column])
        elif np.issubdtype(df_processed[column].dtype, np.number):
            # 强制转换为数值类型，无法转换的值设置为 NaN
            df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
            # 用该列的平均值临时填充缺失值
            df_processed[column] = df_processed[column].fillna(df_processed[column].mean())
            df_processed[column] = df_processed[column].astype(float)
        else:
            # 对于其他类型，您可以根据需要进行处理
            df_processed[column] = df_processed[column].fillna('MISSING')
            encoders[column] = LabelEncoder()
            df_processed[column] = encoders[column].fit_transform(df_processed[column])

    # 检查是否还有缺失值
    if df_processed.isna().any().any():
        raise ValueError("数据预处理后仍存在缺失值。请检查数据处理步骤。")

    return df_processed, missing_mask, encoders


def train_tabnet_model(X_train, y_train, target_column):
    """
    训练TabNet模型
    """
    model = TabNetRegressor(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        verbose=10
    )

    # 将DataFrame转换为numpy数组
    X_train_values = X_train.values.astype(np.float32)
    # 将目标变量重塑为2D数组
    y_train_values = y_train.values.astype(np.float32).reshape(-1, 1)

    # 确保没有 NaN 值
    if np.isnan(X_train_values).any() or np.isnan(y_train_values).any():
        raise ValueError(f"训练数据中包含 NaN 值。目标列: {target_column}")

    model.fit(
        X_train=X_train_values,
        y_train=y_train_values,
        max_epochs=100,
        patience=10,
        batch_size=256,
        virtual_batch_size=128
    )

    return model


def fill_missing_values(df_path, output_path):
    """
    主函数：读取数据，填补缺失值，保存结果
    """
    print("读取数据...")
    try:
        df = pd.read_csv(df_path, low_memory=False)
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {str(e)}")
        return

    # 准备数据
    try:
        df_processed, missing_mask, encoders = prepare_data(df)
    except ValueError as ve:
        print(f"数据预处理错误: {str(ve)}")
        return

    # 对每个包含缺失值的列进行处理
    for column in df.columns:
        if df[column].isna().any():
            print(f"\n处理列: {column}")

            # 获取该列的非缺失值数据
            train_data = df_processed[~missing_mask[column]].copy()

            # 准备训练数据
            X = train_data.drop(columns=[column])
            y = train_data[column]

            # 分割训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 训练模型
            try:
                model = train_tabnet_model(X_train, y_train, column)
            except ValueError as ve:
                print(f"训练模型时发生错误: {str(ve)}")
                continue

            # 预测缺失值
            missing_rows = df_processed[missing_mask[column]]
            if len(missing_rows) > 0:
                X_missing = missing_rows.drop(columns=[column])
                # 将预测数据也转换为numpy数组
                X_missing_values = X_missing.values.astype(np.float32)

                # 确保预测数据中没有 NaN
                if np.isnan(X_missing_values).any():
                    print(f"缺失值预测数据中包含 NaN。列: {column}")
                    continue

                predictions = model.predict(X_missing_values)

                # 填充预测值
                df_processed.loc[missing_mask[column], column] = predictions

    # 反向转换编码的分类变量
    for column, encoder in encoders.items():
        try:
            df_processed[column] = encoder.inverse_transform(
                df_processed[column].astype(int)
            )
        except Exception as e:
            print(f"反向转换编码时发生错误: {str(e)}")
            # 如果反向转换失败，保留原始编码
            pass

    # 最终检查是否还有缺失值
    if df_processed.isna().any().any():
        print("警告: 填充后数据仍存在缺失值。请检查填充步骤。")
    else:
        print("所有缺失值已成功填充。")

    # 保存结果
    try:
        print("\n保存结果...")
        df_processed.to_csv(output_path, index=False)
        print(f"处理完成！填充后的数据已保存至: {output_path}")
    except Exception as e:
        print(f"保存 CSV 文件时发生错误: {str(e)}")




if __name__ == "__main__":
    input_path = r"E:\data_unfilled.csv"
    output_path = r"E:\data_filled.csv"

    try:
        fill_missing_values(input_path, output_path)
    except Exception as e:
        print(f"发生错误: {str(e)}")