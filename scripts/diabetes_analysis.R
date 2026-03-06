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