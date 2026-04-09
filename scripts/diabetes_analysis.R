# Project: Lifestyle Factors and Diabetes Risk Analysis
# 28-Step Comprehensive Data Analysis Pipeline
# ====================================================

# Note: You may need to install the following packages if you don't have them:
# install.packages(c("ggplot2", "dplyr", "corrplot", "caret", "pROC", "randomForest", "patchwork", "e1071", "rpart", "kknn"))

# --------------------------------
# Step 1: Load Required Libraries
# --------------------------------
library(ggplot2)
library(dplyr)
library(corrplot)
library(caret)
library(pROC)
library(randomForest)
library(patchwork)
library(e1071)
library(rpart)
library(kknn)

# --------------------------------
# Step 2: Load the Dataset
# --------------------------------
data <- read.csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# --------------------------------
# Step 3: Data Cleaning & Type Conversion
# --------------------------------
data$Diabetes_binary <- factor(data$Diabetes_binary, levels = c(0, 1))
data$HighBP <- factor(data$HighBP)
data$HighChol <- factor(data$HighChol)
data$Smoker <- factor(data$Smoker)
data$PhysActivity <- factor(data$PhysActivity)
data$Fruits <- factor(data$Fruits)
data$Veggies <- factor(data$Veggies)
data$HvyAlcoholConsump <- factor(data$HvyAlcoholConsump)
data$GenHlth <- factor(data$GenHlth)
data$DiffWalk <- factor(data$DiffWalk)
data$Sex <- factor(data$Sex)

# --------------------------------
# Step 4: Preliminary Data Exploration
# --------------------------------
cat("\n--- Dataset Structure ---\n")
str(data)
cat("\n--- Summary Statistics ---\n")
summary(data)

# --------------------------------
# Step 5: Checking for Missing Values
# --------------------------------
cat("\n--- Missing Values Check ---\n")
print(colSums(is.na(data)))

# --------------------------------
# Step 6: Detecting and Handling Outliers (BMI)
# --------------------------------
bmi_outliers <- data %>% filter(BMI > 70 | BMI < 12)
cat("\nNumber of extreme BMI outliers detected:", nrow(bmi_outliers), "\n")
data <- data %>% filter(BMI <= 70 & BMI >= 12)

# --------------------------------
# Step 7: Feature Engineering: Lifestyle Score
# --------------------------------
data <- data %>%
  mutate(Lifestyle_Score = (as.numeric(as.character(PhysActivity)) +
                            as.numeric(as.character(Fruits)) +
                            as.numeric(as.character(Veggies))) -
                            (as.numeric(as.character(Smoker)) +
                            as.numeric(as.character(HvyAlcoholConsump))))

# --------------------------------
# Step 8: Statistical Significance Testing (Chi-Square)
# --------------------------------
chi_test <- chisq.test(table(data$HighBP, data$Diabetes_binary))
cat("\n--- Chi-Square Test (HighBP vs Diabetes) ---\n")
print(chi_test)

# --------------------------------
# Step 9: Statistical Significance Testing (T-tests)
# --------------------------------
cat("\n--- T-Test (BMI vs Diabetes Status) ---\n")
t_test_bmi <- t.test(BMI ~ Diabetes_binary, data = data)
print(t_test_bmi)

# --------------------------------
# Step 10: Exploratory Data Visualization (Combined)
# --------------------------------
p1 <- ggplot(data, aes(x = Diabetes_binary, fill = Diabetes_binary)) +
  geom_bar() + labs(title = "Diabetes Dist.", x="Status", y="Count") + theme_minimal()

p2 <- ggplot(data, aes(x = BMI, fill = Diabetes_binary)) +
  geom_density(alpha = 0.5) + labs(title = "BMI Density", x="BMI") + theme_minimal()

p3 <- ggplot(data, aes(x = Lifestyle_Score, fill = Diabetes_binary)) +
  geom_bar(position = "fill") + labs(title = "Lifestyle vs Diabetes", x="Score", y="Prop") + theme_minimal()

print((p1 | p2) / p3)

# --------------------------------
# Step 11: Correlation Analysis
# --------------------------------
numeric_cols <- data %>%
  mutate(Diabetes_binary_numeric = as.numeric(as.character(Diabetes_binary))) %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_cols)
cat("\n--- Correlation Matrix (Top Factors) ---\n")
print(cor_matrix["Diabetes_binary_numeric", ], digits = 2)

# --------------------------------
# Step 12: Data Splitting (Train/Test Sets)
# --------------------------------
set.seed(123)
train_index <- createDataPartition(data$Diabetes_binary, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# --------------------------------
# Step 13: Logistic Regression Model Training + Sample Predictions
# --------------------------------
model_lr <- glm(Diabetes_binary ~ BMI + HighBP + HighChol + Age + Lifestyle_Score,
                data = train_data, family = binomial)
cat("\n--- Logistic Regression Model Summary ---\n")
summary(model_lr)

# -- Inline predictions on 5 sample test cases --
cat("\n--- Logistic Regression: Sample Predictions (first 5 test rows) ---\n")
lr_sample_probs <- predict(model_lr, newdata = head(test_data, 5), type = "response")
lr_sample_preds <- ifelse(lr_sample_probs > 0.5, "Diabetic", "Non-Diabetic")
lr_sample_df <- data.frame(
  Actual          = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  Predicted       = lr_sample_preds,
  Risk_Prob_Pct   = round(lr_sample_probs * 100, 2)
)
print(lr_sample_df)

# --------------------------------
# Step 14: Model Diagnostics (Residual Analysis)
# --------------------------------
# cat("\n--- Model Residuals summary ---\n")
# print(summary(model_lr$residuals))

# --------------------------------
# Step 15: Multi-Model Training (RF, SVM, Tree, kNN) + Sample Predictions
# --------------------------------
set.seed(123)

train_subset <- train_data %>% sample_n(10000)

cat("\n--- Training Multiple Models ---\n")

# A. Random Forest
model_rf <- randomForest(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score,
                         data = train_subset, ntree = 100)
cat("Random Forest: [OK]\n")

# -- RF sample predictions --
rf_sample_probs <- predict(model_rf, newdata = head(test_data, 5), type = "prob")[, 2]
rf_sample_preds <- ifelse(rf_sample_probs > 0.5, "Diabetic", "Non-Diabetic")
cat("\n--- Random Forest: Sample Predictions (first 5 test rows) ---\n")
rf_sample_df <- data.frame(
  Actual        = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  Predicted     = rf_sample_preds,
  Risk_Prob_Pct = round(rf_sample_probs * 100, 2)
)
print(rf_sample_df)

# B. Support Vector Machine (SVM)
model_svm <- svm(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score,
                 data = train_subset, probability = TRUE)
cat("\nSVM: [OK]\n")

# -- SVM sample predictions --
svm_sample_raw   <- predict(model_svm, newdata = head(test_data, 5), probability = TRUE)
svm_sample_probs <- attr(svm_sample_raw, "probabilities")[, 2]
svm_sample_preds <- ifelse(svm_sample_probs > 0.5, "Diabetic", "Non-Diabetic")
cat("\n--- SVM: Sample Predictions (first 5 test rows) ---\n")
svm_sample_df <- data.frame(
  Actual        = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  Predicted     = svm_sample_preds,
  Risk_Prob_Pct = round(svm_sample_probs * 100, 2)
)
print(svm_sample_df)

# C. Decision Tree
model_tree <- rpart(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score,
                    data = train_data, method = "class")
cat("\nDecision Tree: [OK]\n")

# -- Tree sample predictions --
tree_sample_probs <- predict(model_tree, newdata = head(test_data, 5), type = "prob")[, 2]
tree_sample_preds <- ifelse(tree_sample_probs > 0.5, "Diabetic", "Non-Diabetic")
cat("\n--- Decision Tree: Sample Predictions (first 5 test rows) ---\n")
tree_sample_df <- data.frame(
  Actual        = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  Predicted     = tree_sample_preds,
  Risk_Prob_Pct = round(tree_sample_probs * 100, 2)
)
print(tree_sample_df)

# D. k-Nearest Neighbors (kNN)
model_knn <- train(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score,
                   data = train_subset, method = "knn",
                   tuneLength = 5, trControl = trainControl(method = "cv"))
cat("\nkNN: [OK]\n")

# -- kNN sample predictions --
knn_sample_probs <- predict(model_knn, newdata = head(test_data, 5), type = "prob")[, 2]
knn_sample_preds <- ifelse(knn_sample_probs > 0.5, "Diabetic", "Non-Diabetic")
cat("\n--- kNN: Sample Predictions (first 5 test rows) ---\n")
knn_sample_df <- data.frame(
  Actual        = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  Predicted     = knn_sample_preds,
  Risk_Prob_Pct = round(knn_sample_probs * 100, 2)
)
print(knn_sample_df)

# -- Ensemble: side-by-side comparison of all 5 models on same 5 rows --
cat("\n--- All-Model Prediction Comparison (first 5 test rows) ---\n")
ensemble_df <- data.frame(
  Actual      = ifelse(head(test_data$Diabetes_binary, 5) == 1, "Diabetic", "Non-Diabetic"),
  LR_Pred     = lr_sample_preds,
  RF_Pred     = rf_sample_preds,
  SVM_Pred    = svm_sample_preds,
  Tree_Pred   = tree_sample_preds,
  kNN_Pred    = knn_sample_preds,
  LR_Prob     = round(lr_sample_probs  * 100, 1),
  RF_Prob     = round(rf_sample_probs  * 100, 1),
  SVM_Prob    = round(svm_sample_probs * 100, 1),
  Tree_Prob   = round(tree_sample_probs* 100, 1),
  kNN_Prob    = round(knn_sample_probs * 100, 1)
)
print(ensemble_df)

# --------------------------------
# Step 16: Model Performance & Predictions (Full Test Set)
# --------------------------------
lr_probs   <- predict(model_lr,   newdata = test_data, type = "response")
lr_preds   <- factor(ifelse(lr_probs > 0.5, 1, 0), levels = c(0, 1))

rf_probs   <- predict(model_rf,   newdata = test_data, type = "prob")[, 2]

svm_pred_raw <- predict(model_svm, newdata = test_data, probability = TRUE)
svm_probs    <- attr(svm_pred_raw, "probabilities")[, 2]

tree_probs <- predict(model_tree, newdata = test_data, type = "prob")[, 2]
knn_probs  <- predict(model_knn,  newdata = test_data, type = "prob")[, 2]

# --------------------------------
# Step 17: Confusion Matrix Visualization
# --------------------------------
conf_matrix <- confusionMatrix(lr_preds, test_data$Diabetes_binary)
cat("\n--- Confusion Matrix (Logistic Regression) ---\n")
print(conf_matrix$table)

# --------------------------------
# Step 18: Advanced Validation Metrics
# --------------------------------
cat("\n--- Detailed Performance Metrics ---\n")
cat("Accuracy:  ", round(conf_matrix$overall['Accuracy'],   4), "\n")
cat("Precision: ", round(conf_matrix$byClass['Precision'],  4), "\n")
cat("Recall:    ", round(conf_matrix$byClass['Recall'],     4), "\n")
cat("F1-Score:  ", round(conf_matrix$byClass['F1'],         4), "\n")

# --------------------------------
# Step 19: ROC Curve and AUC Score
# --------------------------------
roc_obj <- roc(test_data$Diabetes_binary, lr_probs)
cat("\n--- ROC Analysis ---\n")
cat("AUC Value: ", round(auc(roc_obj), 4), "\n")
plot(roc_obj, main = "ROC Curve (Logistic Regression)", col = "blue")

# --------------------------------
# Step 20: Feature Importance & Interpretation
# --------------------------------
cat("\n--- Feature Importance (Odds Ratios) ---\n")
odds_ratios <- exp(coef(model_lr))
print(sort(odds_ratios, decreasing = TRUE))

# --------------------------------
# Step 21: K-Fold Cross-Validation
# --------------------------------
cat("\n--- 5-Fold Cross-Validation (Logistic Regression) ---\n")
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
cv_model <- train(Diabetes_binary ~ BMI + HighBP + HighChol + Age + Lifestyle_Score,
                  data = train_data, method = "glm", family = "binomial",
                  trControl = train_control)
print(cv_model)

# --------------------------------
# Step 22: Hyperparameter Tuning (Random Forest)
# --------------------------------
cat("\n--- Hyperparameter Tuning (RF mtry) ---\n")
tune_grid <- expand.grid(.mtry = c(2, 3, 4))
rf_tuned <- train(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score,
                  data = train_subset, method = "rf",
                  tuneGrid = tune_grid, trControl = train_control,
                  ntree = 50)
print(rf_tuned$bestTune)

# --------------------------------
# Step 23: Investigating Interaction Effects
# --------------------------------
cat("\n--- Interaction Effect Analysis (BMI * Age) ---\n")
model_int <- glm(Diabetes_binary ~ BMI * Age + HighBP + HighChol,
                 data = train_data, family = binomial)
coeffs <- summary(model_int)$coefficients
print(coeffs[nrow(coeffs), ])

# --------------------------------
# Step 24: Multi-Model ROC Comparison
# --------------------------------
cat("\n--- Visualizing Model Comparison (ROC Curves) ---\n")
roc_rf   <- roc(test_data$Diabetes_binary, rf_probs)
roc_svm  <- roc(test_data$Diabetes_binary, svm_probs)
roc_tree <- roc(test_data$Diabetes_binary, tree_probs)
roc_knn  <- roc(test_data$Diabetes_binary, knn_probs)

plot(roc_obj,  col = "blue",   main = "High-Fidelity Model Comparison")
plot(roc_rf,   add = TRUE, col = "red")
plot(roc_svm,  add = TRUE, col = "green")
plot(roc_tree, add = TRUE, col = "orange")
plot(roc_knn,  add = TRUE, col = "purple")

legend("bottomright",
       legend = c("Logistic Reg", "Random Forest", "SVM", "Decision Tree", "kNN"),
       col    = c("blue", "red", "green", "orange", "purple"), lwd = 2, cex = 0.6)

# --------------------------------
# Step 24b: AUC Summary Table
# --------------------------------
auc_results <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "SVM", "Decision Tree", "kNN"),
  AUC   = c(auc(roc_obj), auc(roc_rf), auc(roc_svm), auc(roc_tree), auc(roc_knn))
)
print(auc_results)

# --------------------------------
# Step 25: Model Calibration Analysis
# --------------------------------
cat("\n--- Calibration (Reliability) Check ---\n")
cal_data <- data.frame(obs = test_data$Diabetes_binary, prob = lr_probs)
cal_plot <- calibration(obs ~ prob, data = cal_data, cuts = 10)
plot(cal_plot)

# --------------------------------
# Step 26: Fairness & Bias Check (By Gender)
# --------------------------------
cat("\n--- Model Performance by Sex (Bias Analysis) ---\n")
test_data$preds <- lr_preds
accuracy_by_sex <- test_data %>%
  group_by(Sex) %>%
  summarise(Accuracy = mean(preds == Diabetes_binary))
print(accuracy_by_sex)

# --------------------------------
# Step 27: Model Serialization (Saving for Production)
# --------------------------------
cat("\n--- Saving Model and Artifacts ---\n")
if (!dir.exists("models")) dir.create("models")
saveRDS(model_lr,  "models/diabetes_logistic_model.rds")
saveRDS(cv_model,  "models/cv_performance_metadata.rds")
saveRDS(model_rf,  "models/diabetes_rf_model.rds")
saveRDS(model_svm, "models/diabetes_svm_model.rds")
saveRDS(model_tree,"models/diabetes_tree_model.rds")
saveRDS(model_knn, "models/diabetes_knn_model.rds")
cat("All 5 models saved to 'models/' directory.\n")

# --------------------------------
# Step 28: Deployment-Ready Prediction Function (All 5 Models)
# --------------------------------
predict_diabetes_risk <- function(bmi, bp, chol, age, lifestyle) {
  input <- data.frame(
    BMI             = bmi,
    HighBP          = factor(bp,   levels = c(0, 1)),
    HighChol        = factor(chol, levels = c(0, 1)),
    Age             = age,
    Lifestyle_Score = lifestyle
  )

  # Logistic Regression
  lr_prob  <- predict(model_lr,   newdata = input, type = "response")
  lr_label <- ifelse(lr_prob > 0.5, "Diabetic", "Non-Diabetic")

  # Random Forest
  rf_prob  <- predict(model_rf,   newdata = input, type = "prob")[, 2]
  rf_label <- ifelse(rf_prob  > 0.5, "Diabetic", "Non-Diabetic")

  # SVM
  svm_raw  <- predict(model_svm,  newdata = input, probability = TRUE)
  svm_prob <- attr(svm_raw, "probabilities")[, 2]
  svm_label <- ifelse(svm_prob > 0.5, "Diabetic", "Non-Diabetic")

  # Decision Tree
  tree_prob  <- predict(model_tree, newdata = input, type = "prob")[, 2]
  tree_label <- ifelse(tree_prob > 0.5, "Diabetic", "Non-Diabetic")

  # kNN
  knn_prob  <- predict(model_knn, newdata = input, type = "prob")[, 2]
  knn_label <- ifelse(knn_prob > 0.5, "Diabetic", "Non-Diabetic")

  # Majority vote
  votes      <- c(lr_label, rf_label, svm_label, tree_label, knn_label)
  ensemble   <- ifelse(sum(votes == "Diabetic") >= 3, "Diabetic", "Non-Diabetic")
  mean_prob  <- mean(c(lr_prob, rf_prob, svm_prob, tree_prob, knn_prob))

  result <- data.frame(
    Model       = c("Logistic Regression","Random Forest","SVM",
                    "Decision Tree","kNN","--- Ensemble (majority vote) ---"),
    Prediction  = c(lr_label, rf_label, svm_label,
                    tree_label, knn_label, ensemble),
    Risk_Prob   = paste0(round(c(lr_prob, rf_prob, svm_prob,
                                 tree_prob, knn_prob, mean_prob) * 100, 2), "%")
  )

  cat("\n====== Diabetes Risk Prediction ======\n")
  cat(sprintf("Input  ->  BMI: %.1f | HighBP: %s | HighChol: %s | Age: %d | Lifestyle Score: %d\n",
              bmi,
              ifelse(bp   == 1, "Yes", "No"),
              ifelse(chol == 1, "Yes", "No"),
              age, lifestyle))
  cat("--------------------------------------\n")
  print(result, row.names = FALSE)
  cat("======================================\n")

  invisible(result)
}

# Example usage:
predict_diabetes_risk(bmi = 28, bp = 1, chol = 1, age = 50, lifestyle = 2)

cat("\nFull 28-Step Analysis Pipeline Complete.\n")