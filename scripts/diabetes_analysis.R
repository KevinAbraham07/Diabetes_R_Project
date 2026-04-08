# Project: Lifestyle Factors and Diabetes Risk Analysis
# 28-Step Comprehensive Data Analysis Pipeline
# ====================================================

# Note: You may need to install the following packages if you don't have them:
# install.packages(c("ggplot2", "dplyr", "corrplot", "caret", "pROC", "randomForest", "patchwork"))

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

# Displaying combined plot using patchwork
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
test_data <- data[-train_index, ]

# --------------------------------
# Step 13: Logistic Regression Model Training
# --------------------------------
model_lr <- glm(Diabetes_binary ~ BMI + HighBP + HighChol + Age + Lifestyle_Score, 
                data = train_data, family = binomial)
cat("\n--- Logistic Regression Model Summary ---\n")
summary(model_lr)

# --------------------------------
# Step 14: Model Diagnostics (Residual Analysis)
# --------------------------------
# cat("\n--- Model Residuals summary ---\n")
# print(summary(model_lr$residuals))

# --------------------------------
# Step 15: Random Forest Model Training (Comparison)
# --------------------------------
train_subset <- train_data[sample(1:nrow(train_data), 10000), ] 
model_rf <- randomForest(Diabetes_binary ~ BMI + HighBP + Age + Lifestyle_Score, 
                         data = train_subset, ntree = 100)
cat("\n--- Random Forest Model Trained (Subset of 10k) ---\n")

# --------------------------------
# Step 16: Model Performance on Test Data
# --------------------------------
lr_probs <- predict(model_lr, newdata = test_data, type = "response")
lr_preds <- factor(ifelse(lr_probs > 0.5, 1, 0), levels = c(0, 1))

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
cat("Accuracy:  ", round(conf_matrix$overall['Accuracy'], 4), "\n")
cat("Precision: ", round(conf_matrix$byClass['Precision'], 4), "\n")
cat("Recall:    ", round(conf_matrix$byClass['Recall'], 4), "\n")
cat("F1-Score:  ", round(conf_matrix$byClass['F1'], 4), "\n")

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
rf_probs <- predict(model_rf, newdata = test_data, type = "prob")[,2]
roc_rf <- roc(test_data$Diabetes_binary, rf_probs)

plot(roc_obj, col = "blue", main = "Multi-Model ROC Comparison")
plot(roc_rf, add = TRUE, col = "red")
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), 
       col = c("blue", "red"), lwd = 2)

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
if(!dir.exists("models")) dir.create("models")
saveRDS(model_lr, "models/diabetes_logistic_model.rds")
saveRDS(cv_model, "models/cv_performance_metadata.rds")
cat("Models saved to 'models/' directory.\n")

# --------------------------------
# Step 28: Deployment-Ready Prediction Function
# --------------------------------
predict_diabetes_risk <- function(bmi, bp, chol, age, lifestyle) {
  input <- data.frame(BMI = bmi, HighBP = factor(bp), 
                      HighChol = factor(chol), Age = age, 
                      Lifestyle_Score = lifestyle)
  prob <- predict(model_lr, newdata = input, type = "response")
  return(paste0("Diabetes Risk Probability: ", round(prob * 100, 2), "%"))
}

# Example usage:
# predict_diabetes_risk(bmi = 28, bp = 1, chol = 1, age = 50, lifestyle = 2)

# Final Conclusion note
cat("\nFull 28-Step Analysis Pipeline Complete.\n")