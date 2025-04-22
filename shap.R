# 安装所需的R包
install.packages("shapviz")
install.packages("lattice")
install.packages("ggplot2")
install.packages("caret")
install.packages("pROC")
install.packages("tibble")
install.packages("ROCit")
install.packages("xgboost")

# 加载所需的R包
library(shapviz)
library(lattice)
library(ggplot2)
library(caret)
library(pROC)
library(tibble)
library(ROCit)
library(xgboost)

# 设置工作目录
setwd("E:/papers/Xu_project1/code/Medicine LLM/figure_coding/R")

# 读取数据
data = read.csv("OA_99-24_max-min_filled.csv", header = T, check.names = F)
view(data)
str(data)

# 分割训练集和测试集
inTrain <- createDataPartition(y=data[,"oa_yn"], p=0.7, list=F)
traindata <- data[inTrain,]
testdata <- data[-inTrain,]
model_xgboost = xgboost(data = as.matrix(traindata[,c(1:(ncol(traindata)-1))]),label = traindata$oa_yn, max_depth = 3, eta =1, nthread = 2, nrounds = 100, objective = "binary:logistic")
##预测结果，并将预测结果存贮在名为“pred”的新列中
traindata$pred <- predict(model_xgboost, as.matrix(traindata[,c(1:(ncol(traindata)-1))]))
ROC_train <- round(auc(response=traindata$oa_yn, predictor=traindata$pred), 3)
ROC_train
##设置置信区间
CI_train = ci(response=traindata$oa_yn,predictor=traindata$pred)
CI_train
##通过paste0连接
AUC_CI_train=paste0("AUC=",round(CI_train[2],3),",95%ci(", round(CI_train[1],3),"-", round(CI_train[3],3),")")

##绘制ROC曲线
##rocit函数用于计算真正例率（TPR）和假正率（FPR）
#需要提供模型的预测分数（score）和真正类别（class）
ROC_data <- rocit(score=traindata$pred, class = traindata$oa_yn)
##创建包含模型名称、TPR、FPR和AUC值的数据框,name="Model"模型的名称，TPR真正比例，
ml=tibble(name="Model",TPR=ROC_data$TPR,FPR=ROC_data$FPR, AUC=AUC_CI_train)

ggplot(ml,aes(x = FPR, y = TPR)) + geom_path() + labs(title = " ", x = "False Positive Rate (1-Specificity)", y ="Ture Positive Rate (Sensitivity)") + geom_abline(lty=3) + theme_classic() + annotate("text", x =0.6, y = 0.2, label = AUC_CI_train, colour = "black") + scale_y_continuous(expand = c(0,0)) +scale_x_continuous(expand=c(0,0)) + theme(axis.ticks.length = unit(-0.1, "cm"), legend.position = c(0.7, 0.2), legend.title = element_blank(), strip.background = element_blank(), text = element_text(size = 15, color = "black"))

##计算shap值并绘图
view(traindata)
##计算shap值，使用shapvia函数计算并可视化XGBoost模型的SHAP值
shap_xgboost = shapviz(model_xgboost, X_pred = as.matrix(traindata[,c(1:(ncol(traindata)-2))]))
#绘制单个样本瀑布图
sv_waterfall(shap_xgboost, row_id = 10)
#单个样本力图
sv_force(shap_xgboost, row_id = 2)
#绘制变量重要性蜂群图
sv_importance(shap_xgboost, kind = "beeswarm")
#变量重要性柱状图
sv_importance(shap_xgboost) + theme_bw()
#单个变量依赖图，"depression_score"是要分析的特征，alpha = 0.5，点的透明度，color_var = NULL表示不适用颜色变量进行着色，theme_bw()用于设置绘图肢体为白色背景
#sv_dependence(shap_xgboost, "age", alpha = 0.5, size = 1.5, color_var = NULL) + theme_bw()
sv_dependence(shap_xgboost, "age", alpha = 0.5, size = 1.5) + labs(title = "SHAP Dependence Plot for age")
#多个变量偏相关依赖图，这里展示了education、gender、age和weight的偏相关性
sv_dependence(shap_xgboost, v = c("mchc", "age", "tot_protein", "hematocrit", "neutrophil_pct", "glucose", "cholesterol", "platelet_cnt", "triglycerides", "red_cnt", "iron", "eosinophil_pct")) + theme_bw()
# 绘制多个变量偏相关依赖图，并保存为固定大小的图片
p <- sv_dependence(shap_xgboost, v = c("mchc", "age", "tot_protein", "hematocrit", "neutrophil_pct", "glucose", "cholesterol", "platelet_cnt", "triglycerides", "red_cnt", "alt", "tot_bilirubin")) + theme_bw()
ggsave("dependence_plot.png", plot = p, width = 6, height = 3, units = "in")
##色彩版
sv_dependence(shap_xgboost, v = c("mchc", "age", "tot_protein", "hematocrit", "neutrophil_pct", "glucose", "cholesterol", "platelet_cnt", "triglycerides", "red_cnt", "alt", "tot_bilirubin"), color_var = "age") + theme_bw() + scale_color_gradient(low = "blue", high = "red") +   labs(color = "Age")

sv_dependence(shap_xgboost, v = c("health_behavior_score", "poverty", "gender", "marital", "age", "LE8_score", "sleep_score", "education", "bmi_score")) + theme_bw()

# 尝试计算和绘制交互效应
tryCatch({
  shap_interactions <- shapviz(model_xgboost, 
                               X_pred = X_train,
                               X = traindata[,c(1:(ncol(traindata)-2))],
                               interactions = TRUE)
  
  # 绘制交互图
  p7 <- sv_interaction(shap_interactions) + 
    theme_bw() + 
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ggtitle("SHAP Interaction Plot")
  print(p7)
}, error = function(e) {
  message("Note: Could not generate interaction plot due to: ", e$message)
})