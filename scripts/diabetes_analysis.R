# ====================================================
# Project: Lifestyle Factors and Diabetes Risk Analysis
# 20-Step Comprehensive Data Analysis Pipeline
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
# Convert categorical indicators to factors
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
# Checking for extreme BMI values (e.g., above 70 or below 12)
bmi_outliers <- data %>% filter(BMI > 70 | BMI < 12)
cat("\nNumber of extreme BMI outliers detected:", nrow(bmi_outliers), "\n")
# Filter out extreme outliers for more robust analysis
data <- data %>% filter(BMI <= 70 & BMI >= 12)

# --------------------------------
# Step 7: Feature Engineering: Lifestyle Score
# --------------------------------
# Create a lifestyle score where higher = healthier habits
# (PhysActivity + Fruits + Veggies - Smoker - HvyAlcoholConsump)
data <- data %>%
  mutate(Lifestyle_Score = (as.numeric(as.character(PhysActivity)) + 
                            as.numeric(as.character(Fruits)) + 
                            as.numeric(as.character(Veggies))) - 
                            (as.numeric(as.character(Smoker)) + 
                            as.numeric(as.character(HvyAlcoholConsump))))

# --------------------------------
# Step 8: Statistical Significance Testing (Chi-Square)
# --------------------------------
# Chi-square test for categorical relationship between HighBP and Diabetes
chi_test <- chisq.test(table(data$HighBP, data$Diabetes_binary))
cat("\n--- Chi-Square Test (HighBP vs Diabetes) ---\n")
print(chi_test)

# --------------------------------
# Step 9: Statistical Significance Testing (T-tests)
# --------------------------------
# T-test to see if BMI significantly differs between diabetics and non-diabetics
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
# Include Diabetes_binary in correlation by temporarily converting it back to numeric
numeric_cols <- data %>% 
  mutate(Diabetes_binary_numeric = as.numeric(as.character(Diabetes_binary))) %>%
  select(where(is.numeric))

cor_matrix <- cor(numeric_cols)
cat("\n--- Correlation Matrix (Top Factors) ---\n")
print(cor_matrix["Diabetes_binary_numeric", ], digits = 2)
# corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)

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
# Using a subset for faster demonstration or full if dataset is reasonable
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

# Final Conclusion note
cat("\nAnalysis Pipeline Complete.\n")