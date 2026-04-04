# ====================================================
# Project: Lifestyle Factors and Diabetes Risk Analysis
# ====================================================



# --------------------------------
# Step 1: Load Required Libraries
# --------------------------------

library(ggplot2)
library(dplyr)
library(corrplot)



# --------------------------------
# Step 2: Load the Dataset
# --------------------------------

data <- read.csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# Convert binary variables to factors
data$Diabetes_binary <- factor(data$Diabetes_binary)
data$Smoker <- factor(data$Smoker)
data$PhysActivity <- factor(data$PhysActivity)
data$Fruits <- factor(data$Fruits)
data$Veggies <- factor(data$Veggies)
data$HvyAlcoholConsump <- factor(data$HvyAlcoholConsump)

head(data)
dim(data)
colnames(data)



# --------------------------------
# Step 3: Dataset Overview
# --------------------------------

str(data)
summary(data)



# --------------------------------
# Step 4: Check for Missing Values
# --------------------------------

colSums(is.na(data))



# --------------------------------
# Step 5: Diabetes Distribution
# --------------------------------

table(data$Diabetes_binary)

ggplot(data, aes(x = Diabetes_binary)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(
    title = "Distribution of Diabetes Cases",
    x = "Diabetes Status",
    y = "Count"
  )


# --------------------------------
# Step 6: BMI Distribution
# --------------------------------

ggplot(data, aes(x = BMI)) +
  geom_histogram(fill = "skyblue", bins = 30) +
  theme_minimal() +
  labs(
    title = "Distribution of BMI in Survey Population",
    x = "BMI",
    y = "Count"
  )


# --------------------------------
# Step 7: BMI vs Diabetes
# --------------------------------

ggplot(data, aes(x = Diabetes_binary, y = BMI, fill = Diabetes_binary)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "BMI vs Diabetes Status",
    x = "Diabetes Status",
    y = "BMI"
  ) +
  guides(fill = "none")



# --------------------------------
# Step 8: Smoking Analysis
# --------------------------------

ggplot(data, aes(x = factor(Smoker), fill = factor(Diabetes_binary))) +
  geom_bar(position = "fill") +
  labs(
    title = "Smoking vs Diabetes",
    x = "Smoking Status",
    y = "Proportion"
  )



# --------------------------------
# Step 9: Diet Factors Analysis
# --------------------------------

ggplot(data, aes(x = factor(Fruits), fill = factor(Diabetes_binary))) +
  geom_bar(position = "fill") +
  labs(
    title = "Fruit Consumption vs Diabetes",
    x = "Fruit Consumption",
    y = "Proportion"
  )

ggplot(data, aes(x = factor(Veggies), fill = factor(Diabetes_binary))) +
  geom_bar(position = "fill") +
  labs(
    title = "Vegetable Consumption vs Diabetes",
    x = "Vegetable Consumption",
    y = "Proportion"
  )



# --------------------------------
# Step 10: Age vs Diabetes
# --------------------------------

ggplot(data, aes(x = factor(Age), fill = factor(Diabetes_binary))) +
  geom_bar(position = "fill") +
  labs(
    title = "Age Group vs Diabetes",
    x = "Age Group",
    y = "Proportion"
  )


# --------------------------------
# Step 11: Correlation Analysis
# --------------------------------

# Reset plotting device
dev.off()

# Select only numeric columns
numeric_data <- data %>% select(where(is.numeric))

# Compute correlation matrix
cor_matrix <- cor(numeric_data)

# Plot heatmap
corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  order = "hclust",
  tl.cex = 0.6,
  tl.col = "black"
)

# --------------------------------
# Step 12: Logistic Regression Model
# --------------------------------

# Convert Diabetes variable back to numeric 0/1
data$Diabetes_binary_num <- as.numeric(as.character(data$Diabetes_binary))

model <- glm(
  Diabetes_binary_num ~ BMI + Age + PhysActivity + Smoker +
    HvyAlcoholConsump + Fruits + Veggies,
  data = data,
  family = binomial
)

summary(model)



# --------------------------------
# Step 13: Model Predictions
# --------------------------------
# Predict probabilities using the trained model

prob_predictions <- predict(model, type = "response")

# Convert predicted probabilities to class labels (0 or 1) using a 0.5 threshold
class_predictions <- ifelse(prob_predictions > 0.5, 1, 0)



# --------------------------------
# Step 14: Confusion Matrix
# --------------------------------

# Generate the confusion matrix (ensuring a 2x2 matrix structure)

conf_matrix <- table(
  Predicted = factor(class_predictions, levels = c(0, 1)), 
  Actual = factor(data$Diabetes_binary_num, levels = c(0, 1))
)

cat("\nConfusion Matrix:\n")
print(conf_matrix)



# --------------------------------
# Step 15: Validation Metrics (Accuracy, Precision, Recall)
# --------------------------------

# Extract values from the confusion matrix
TN <- conf_matrix["0", "0"]
FN <- conf_matrix["0", "1"]
FP <- conf_matrix["1", "0"]
TP <- conf_matrix["1", "1"]

# Calculate Accuracy, Precision, and Recall
accuracy <- (TP + TN) / sum(conf_matrix)
precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0)
f1_score <- ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)

cat("\nModel Evaluation Metrics:\n")
cat("Accuracy:  ", round(accuracy * 100, 2), "%\n")
cat("Precision: ", round(precision * 100, 2), "%\n")
cat("Recall:    ", round(recall * 100, 2), "%\n")
cat("F1 Score:  ", round(f1_score * 100, 2), "%\n")



# --------------------------------
# Step 16: Feature Importance Analysis (Odds Ratios)
# --------------------------------

# Extract exponentiated coefficients (odds ratios)
odds_ratios <- exp(coef(model))

# Sort and print the odds ratios to understand feature impact
# Values > 1 indicate increased odds of diabetes, < 1 indicate decreased odds
cat("\nOdds Ratios (Feature Importance):\n")
print(sort(odds_ratios, decreasing = TRUE))