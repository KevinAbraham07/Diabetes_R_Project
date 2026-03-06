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

ggplot(data, aes(x = factor(Diabetes_binary))) +
  geom_bar(fill = "steelblue") +
  labs(
    title = "Distribution of Diabetes Cases",
    x = "Diabetes Status",
    y = "Count"
  )



# --------------------------------
# Step 6: BMI Analysis
# --------------------------------

ggplot(data, aes(x = factor(Diabetes_binary), y = BMI)) +
  geom_boxplot(fill = "orange") +
  labs(
    title = "BMI vs Diabetes",
    x = "Diabetes Status",
    y = "BMI"
  )



# --------------------------------
# Step 7: Physical Activity Analysis
# --------------------------------

ggplot(data, aes(x = factor(PhysActivity), fill = factor(Diabetes_binary))) +
  geom_bar(position = "fill") +
  labs(
    title = "Physical Activity vs Diabetes",
    x = "Physical Activity",
    y = "Proportion"
  )



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

cor_matrix <- cor(data)

corrplot(cor_matrix, method = "color")



# --------------------------------
# Step 12: Logistic Regression Model
# --------------------------------

model <- glm(
  Diabetes_binary ~ BMI + Age + PhysActivity + Smoker + HvyAlcoholConsump + Fruits + Veggies,
  data = data,
  family = binomial
)

summary(model)