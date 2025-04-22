import os
# ===== 1) 在脚本开头，强制限制底层库并行线程 =====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression, Perceptron, PassiveAggressiveClassifier,
    RidgeClassifier, SGDClassifier, LogisticRegressionCV
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, VotingClassifier,
    HistGradientBoostingClassifier, StackingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import RidgeClassifierCV


# ============================
# 1) 定义 50 种机器学习模型
# ============================
models = {
    # --- 50 种模型 ---
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "SVM (RBF Kernel)": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Extra Trees": ExtraTreesClassifier(),
    "Bagging Classifier": BaggingClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Perceptron": Perceptron(max_iter=1000),
    "Passive-Aggressive": PassiveAggressiveClassifier(max_iter=1000),
    "Ridge Classifier": RidgeClassifier(),
    "SGD Classifier": SGDClassifier(loss='hinge', penalty='l1'),
    "Voting Classifier": VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('svc', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ], voting='soft'),

    "SVC (Linear Kernel)": SVC(kernel='linear', probability=True),
    "SVC (Polynomial Kernel)": SVC(kernel='poly', probability=True),
    "SVC (Sigmoid Kernel)": SVC(kernel='sigmoid', probability=True),
    "LinearSVC": LinearSVC(max_iter=2000),
    "NuSVC": NuSVC(probability=True),
    "MLPClassifier": MLPClassifier(max_iter=2000),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "CalibratedClassifierCV(Perceptron)": CalibratedClassifierCV(
        estimator=Perceptron(max_iter=1000), cv=5
    ),
    "BernoulliNB": BernoulliNB(),
    "MultinomialNB": MultinomialNB(),
    "ComplementNB": ComplementNB(),
    "NearestCentroid": NearestCentroid(),
    "RadiusNeighborsClassifier": RadiusNeighborsClassifier(radius=5.0),
    "LabelPropagation": LabelPropagation(),
    "LabelSpreading": LabelSpreading(),
    # 保留 SelfTrainingClassifier，不再删除
    "SelfTrainingClassifier (DecisionTree)": SelfTrainingClassifier(
        estimator=DecisionTreeClassifier()
    ),
    "StackingClassifier": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=2000)),
            ('svc', SVC(probability=True)),
            ('knn', KNeighborsClassifier())
        ],
        final_estimator=RandomForestClassifier()
    ),
    "CalibratedClassifierCV(LinearSVC)": CalibratedClassifierCV(
        estimator=LinearSVC(max_iter=2000),
        cv=5
    ),
    "RidgeClassifierCV": RidgeClassifierCV(),
    "LogisticRegressionCV": LogisticRegressionCV(max_iter=2000),
    "SGDClassifier (log loss)": SGDClassifier(loss='log_loss'),  # 修正
    "SGDClassifier (modified_huber)": SGDClassifier(loss='modified_huber'),
    "SGDClassifier (elasticnet)": SGDClassifier(penalty='elasticnet'),
    "BaggingClassifier (base=SVC-linear)": BaggingClassifier(
        estimator=SVC(kernel='linear', probability=True)
    ),
    "BaggingClassifier (base=KNN)": BaggingClassifier(
        estimator=KNeighborsClassifier()
    ),
    "BaggingClassifier (base=GNB)": BaggingClassifier(
        estimator=GaussianNB()
    ),
    "OneVsRestClassifier (SVC)": OneVsRestClassifier(SVC(probability=True)),
    "OneVsOneClassifier (LinearSVC)": OneVsOneClassifier(LinearSVC(max_iter=2000)),
    "OutputCodeClassifier (LogisticRegression)": OutputCodeClassifier(
        estimator=LogisticRegression(max_iter=2000),
        code_size=2
    ),
    "ExtraTreeClassifier": ExtraTreeClassifier(),
}


# ===============
# 2) 读取训练集
# ===============
file_path = "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_tran3.csv"
data = pd.read_csv(file_path)

# 定义特征和目标变量
X = data.iloc[:, 2:26]  # 第3到27列是特征
y = data.iloc[:, 1]     # 第2列是目标变量

# 确保目标变量是0和1
assert set(y.unique()) == {0, 1}, "目标变量y必须是0和1的二分类变量。"

# 随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ================================
# 3) 在 (X_test, y_test) 上评估
# ================================
results = []
plt.figure(figsize=(16, 12))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # 若支持 predict_proba，则获取概率绘制 ROC
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", linewidth=2)

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(
    "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_tran3_results_50.csv",
    index=False
)

# 绘制 ROC
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.title("Internal validation dataset", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
plt.grid()
plt.tight_layout()
plt.savefig("Internal validation dataset_50.png", dpi=300)
plt.show()

print(results_df)


# ======================================================
# 4) 测试集 RA_test1.csv
# ======================================================
test_file_path = "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_test3.csv"
test_data_1 = pd.read_csv(test_file_path)
X_test_new_1 = test_data_1.iloc[:, 2:26]
y_test_new_1 = test_data_1.iloc[:, 1]
assert set(y_test_new_1.unique()) == {0, 1}, "测试集1的目标变量y必须是0和1的二分类变量。"

test_results1 = []
plt.figure(figsize=(16, 12))

for name, model in models.items():
    y_pred_new_1 = model.predict(X_test_new_1)
    y_prob_new_1 = model.predict_proba(X_test_new_1)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy_new = accuracy_score(y_test_new_1, y_pred_new_1)
    precision_new = precision_score(y_test_new_1, y_pred_new_1, average="binary")
    recall_new = recall_score(y_test_new_1, y_pred_new_1, average="binary")
    f1_new = f1_score(y_test_new_1, y_pred_new_1, average="binary")

    test_results1.append({
        "Model": name,
        "Accuracy": accuracy_new,
        "Precision": precision_new,
        "Recall": recall_new,
        "F1": f1_new
    })

    if y_prob_new_1 is not None:
        fpr_new, tpr_new, _ = roc_curve(y_test_new_1, y_prob_new_1)
        roc_auc_new = auc(fpr_new, tpr_new)
        plt.plot(fpr_new, tpr_new, label=f"{name} (AUC = {roc_auc_new:.2f})", linewidth=2)

test_results_df1 = pd.DataFrame(test_results1)
test_results_df1.to_csv(
    "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_test3_results_50.csv",
    index=False
)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.title("Internal testing dataset", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
plt.grid()
plt.tight_layout()
plt.savefig("Internal testing dataset_50.png", dpi=300)
plt.show()

print("Test results on new dataset (RA_test1.csv):")
print(test_results_df1)


# ======================================================
# 5) 测试集 RA_test2.csv
# ======================================================
test_file_path = "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_test2.csv"
test_data_2 = pd.read_csv(test_file_path)
X_test_new_2 = test_data_2.iloc[:, 2:26]
y_test_new_2 = test_data_2.iloc[:, 1]
assert set(y_test_new_2.unique()) == {0, 1}, "测试集2的目标变量y必须是0和1的二分类变量。"

test_results2 = []
plt.figure(figsize=(16, 12))

for name, model in models.items():
    y_pred_new_2 = model.predict(X_test_new_2)
    y_prob_new_2 = model.predict_proba(X_test_new_2)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy_new = accuracy_score(y_test_new_2, y_pred_new_2)
    precision_new = precision_score(y_test_new_2, y_pred_new_2, average="binary")
    recall_new = recall_score(y_test_new_2, y_pred_new_2, average="binary")
    f1_new = f1_score(y_test_new_2, y_pred_new_2, average="binary")

    test_results2.append({
        "Model": name,
        "Accuracy": accuracy_new,
        "Precision": precision_new,
        "Recall": recall_new,
        "F1": f1_new
    })

    if y_prob_new_2 is not None:
        fpr_new, tpr_new, _ = roc_curve(y_test_new_2, y_prob_new_2)
        roc_auc_new = auc(fpr_new, tpr_new)
        plt.plot(fpr_new, tpr_new, label=f"{name} (AUC = {roc_auc_new:.2f})", linewidth=2)

test_results_df2 = pd.DataFrame(test_results2)
test_results_df2.to_csv(
    "E:/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_o/O_test2_results_50.csv",
    index=False
)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
plt.title("External testing dataset", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
plt.grid()
plt.tight_layout()
plt.savefig("External testing dataset_50.png", dpi=300)
plt.show()

print("Test results on new dataset (RA_test2.csv):")
print(test_results_df2)
