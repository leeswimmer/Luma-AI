import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier, RidgeClassifier, \
    SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier, BaggingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

# ===================== 新增： 用于捕获旧版本 sklearn 对某些模型报错的辅助函数 =====================
def safe_cross_val_score(model, X, y, cv, scoring):
    """
    封装 cross_val_score, 若某些模型在旧版本 sklearn 中调用 get_tags 出错,
    则用 try-except 跳过, 返回一组 nan 替代, 避免脚本直接崩溃。
    """
    from sklearn.model_selection import cross_val_score
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    except AttributeError as e:
        print(f"[警告] 在 cross_val_score 时，{model} 出现 AttributeError: {e}")
        print("       将跳过此模型的 cross validation 打分……")
        scores = [np.nan] * cv.get_n_splits(X, y)
    return scores

# ===================== 在 Bootstrapping 时也做类似的 try-except =====================
def safe_model_fit_predict(model, X_train, y_train, X_test, y_test):
    """
    封装 model.fit & predict_proba，如果出现 __sklearn_tags__ 相关问题，就跳过。
    返回 (y_bs_prob, fitted_ok)
    """
    try:
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None
        return y_prob, True
    except AttributeError as e:
        print(f"[警告] 在 fit / predict_proba 时，{model} 出现 AttributeError: {e}")
        print("       将跳过此模型的后续流程……")
        return None, False

# ================ 原代码开始（只新增监控进度和使用safe_xxx函数） ================

# 设置环境变量以减少线程警告
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# 忽略警告
warnings.filterwarnings("ignore")

# 定义50种机器学习模型
models = {
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
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver='lsqr'),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
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
        estimator=LinearSVC(max_iter=2000), cv=5
    ),
    "RidgeClassifierCV": RidgeClassifierCV(),
    "LogisticRegressionCV": LogisticRegressionCV(max_iter=2000),
    "SGDClassifier (log loss)": SGDClassifier(loss='log_loss'),
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
        estimator=LogisticRegression(max_iter=2000), code_size=2
    ),
    "ExtraTreeClassifier": ExtraTreeClassifier()
}


# 计算AUC置信区间
def compute_auc_ci(y_true, y_prob, n_bootstraps=1000, random_seed=42):
    np.random.seed(random_seed)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue  # 跳过只有一个类别的情况

        fpr_tmp, tpr_tmp, _ = roc_curve(y_true[indices], y_prob[indices])
        score = auc(fpr_tmp, tpr_tmp)
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
    upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]

    return lower_bound, upper_bound


# 从CSV文件读取数据
file_path = "E:/papers/Xu_project1/code/Medicine LLM/3 Machine Learning_Prediction/3ROC/results_ra/RA_tran.csv"  # 训练集数据
data = pd.read_csv(file_path)

# 定义特征和目标变量
X = data.iloc[:, 2:26]  # 第3到27列是特征
y = data.iloc[:, 1]     # 第2列是目标变量

# 确保目标变量是0和1
assert set(y.unique()) == {0, 1}, "目标变量y必须是0和1的二分类变量。"

# 随机划分数据集：90% 训练集，10% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# 初始化存储结果的列表 (data测试集结果)
results = []
plt.figure(figsize=(16, 12))  # 增加图片尺寸

# 训练和评估每个模型 (data测试集：X_test, y_test)
for name, model in models.items():
    print("="*60)
    print(f"[INFO] 正在处理模型: {name}")

    # ------------------- 1) 10-fold Cross-Validation with 10 repeats -------------------
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    print(f"    -> 执行 10-fold CV (n_splits=10, n_repeats=10) ...")
    cv_scores = safe_cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"       完成。有效得分数量: {np.count_nonzero(~np.isnan(cv_scores))}, 平均得分: {np.nanmean(cv_scores):.4f}")

    # ------------------- 2) Monte Carlo Cross-Validation (MCCV) -------------------
    mccv = ShuffleSplit(n_splits=100, test_size=0.7, random_state=42)
    print(f"    -> 执行 MCCV (n_splits=100, test_size=0.7) ...")
    mccv_scores = safe_cross_val_score(model, X_train, y_train, cv=mccv, scoring='roc_auc')
    print(f"       完成。有效得分数量: {np.count_nonzero(~np.isnan(mccv_scores))}, 平均得分: {np.nanmean(mccv_scores):.4f}")

    # ------------------- 3) Bootstrapping -------------------
    bootstrap_scores = []
    print(f"    -> 执行 Bootstrapping (1000次) ...")
    for b in range(1000):
        # 打印进度
        if b % 100 == 0 and b > 0:
            print(f"       Bootstrapping 进度: {b}/1000 ...")

        indices = np.random.randint(0, len(X_train), size=int(0.7 * len(X_train)))
        X_bs, y_bs = X_train.iloc[indices], y_train.iloc[indices]

        y_bs_prob, fitted_ok = safe_model_fit_predict(model, X_bs, y_bs, X_test, y_test)
        if (y_bs_prob is not None) and fitted_ok:
            fpr_bs, tpr_bs, _ = roc_curve(y_test, y_bs_prob)
            roc_auc_bs = auc(fpr_bs, tpr_bs)
            bootstrap_scores.append(roc_auc_bs)
        else:
            # 如果 fit 失败，直接跳出本模型的 bootstrap 过程
            bootstrap_scores = [np.nan]
            break

    print(f"       Bootstrapping 完成，实际得到的有效打分次数: {len(bootstrap_scores) if not np.isnan(bootstrap_scores).all() else 0}.")

    # ------------------- 4) 最后在完整训练集上 fit & 测试集上评估 -------------------
    print(f"    -> 在整份训练集上 fit 模型，并对测试集进行预测")
    # 同样用 safe_model_fit_predict 来避免报错
    y_prob_final, fitted_ok = safe_model_fit_predict(model, X_train, y_train, X_test, y_test)
    if (y_prob_final is not None) and fitted_ok:
        y_pred = (y_prob_final >= 0.5).astype(int)  # 如果想用 predict() 也行
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary")
        recall = recall_score(y_test, y_pred, average="binary")
        f1 = f1_score(y_test, y_pred, average="binary")

        fpr, tpr, _ = roc_curve(y_test, y_prob_final)
        roc_auc = auc(fpr, tpr)
        auc_lower, auc_upper = compute_auc_ci(y_test.values, y_prob_final)
    else:
        # 如果最终 fit 还是报错，则此模型全部指标置为 nan
        accuracy = precision = recall = f1 = roc_auc = auc_lower = auc_upper = np.nan

    # ------------------- 5) 汇总结果 -------------------
    res_dict = {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": roc_auc,
        "AUC_CI_Lower": auc_lower,
        "AUC_CI_Upper": auc_upper,
        "10-fold CV AUC Mean": np.nanmean(cv_scores),
        "MCCV AUC Mean": np.nanmean(mccv_scores),
        "Bootstrap AUC Mean": np.nanmean(bootstrap_scores)
    }
    results.append(res_dict)

    # ***** 新增：每跑完一个模型，就把这个模型的结果先写到文件，避免中途断掉后全丢失 *****
    csv_path = "E:/papers/Xu_project1/code/Medicine LLM/3 Machine Learning_Prediction/Cross-validation/results_RA/RA_tran3_results1.csv"
    pd.DataFrame([res_dict]).to_csv(csv_path, mode='a', header=(not os.path.exists(csv_path)), index=False)

    # ------------------- 6) 绘制 ROC 曲线 -------------------
    if (y_prob_final is not None) and not np.isnan(roc_auc):
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", linewidth=2)

# ===================== 原逻辑：将所有结果保存到CSV =====================
results_df = pd.DataFrame(results)
results_df.to_csv(
    "E:/papers/Xu_project1/code/Medicine LLM/3 Machine Learning_Prediction/Cross-validation/results_RA/RA_tran3_results_final1.csv",
    index=False
)

# 完成data测试集(X_test, y_test)ROC曲线图
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)  # 添加对角线
plt.title("Internal validation dataset", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
plt.grid()
plt.tight_layout()
plt.savefig("Internal validation dataset.png", dpi=300)  # 保存高分辨率图片
plt.show()

print("=== 最终所有模型结果汇总 ===")
print(results_df)
