library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)

data <- read.csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# UI
ui <- dashboardPage(

  dashboardHeader(title = "Diabetes Risk Dashboard"),

  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview",  tabName = "dashboard", icon = icon("chart-bar")),
      menuItem("Analysis",  tabName = "analysis",  icon = icon("microscope")),

      selectInput(
        "variable",
        "Select Lifestyle Factor",
        choices = c("BMI","Smoker","PhysActivity",
                    "Fruits","Veggies","HvyAlcoholConsump")
      )
    )
  ),

  dashboardBody(

    tabItems(

      # ── Tab 1: Overview ────────────────────────────────────────────────────
      tabItem(tabName = "dashboard",

        fluidRow(
          valueBoxOutput("totalRecords"),
          valueBoxOutput("diabetesCases"),
          valueBoxOutput("nonDiabetes")
        ),

        fluidRow(
          box(title = "Diabetes Distribution",       width = 6,
              status = "primary", solidHeader = TRUE,
              plotOutput("diabetesPlot", height = 300)),
          box(title = "Lifestyle Factor vs Diabetes", width = 6,
              status = "warning", solidHeader = TRUE,
              plotOutput("factorPlot",   height = 300))
        ),

        fluidRow(
          box(title = "Age vs Diabetes",   width = 6,
              status = "success", solidHeader = TRUE,
              plotOutput("agePlot",   height = 300)),
          box(title = "Correlation Heatmap", width = 6,
              status = "danger",  solidHeader = TRUE,
              plotOutput("corrPlot", height = 300))
        )
      ),

      # ── Tab 2: Analysis (6 new visualizations) ────────────────────────────
      tabItem(tabName = "analysis",

        fluidRow(
          # Viz 1 – BMI density
          box(title = "BMI Distribution by Diabetes Status", width = 6,
              status = "primary", solidHeader = TRUE,
              plotOutput("bmiPlot", height = 300)),
          # Viz 2 – Risk factor prevalence
          box(title = "Health Risk Factors Prevalence", width = 6,
              status = "warning", solidHeader = TRUE,
              plotOutput("riskFactorsPlot", height = 300))
        ),

        fluidRow(
          # Viz 3 – Sex breakdown
          box(title = "Diabetes Rate by Sex", width = 6,
              status = "success", solidHeader = TRUE,
              plotOutput("sexPlot", height = 300)),
          # Viz 4 – Income gradient
          box(title = "Income Level vs Diabetes Rate", width = 6,
              status = "danger",  solidHeader = TRUE,
              plotOutput("incomePlot", height = 300))
        ),

        fluidRow(
          # Viz 5 – Physical & Mental health days (violin)
          box(title = "Poor Physical & Mental Health Days", width = 6,
              status = "info",    solidHeader = TRUE,
              plotOutput("healthDaysPlot", height = 300)),
          # Viz 6 – BP × Cholesterol heatmap tile
          box(title = "High Blood Pressure & High Cholesterol", width = 6,
              status = "primary", solidHeader = TRUE,
              plotOutput("bpCholPlot", height = 300))
        )
      )
    )
  )
)

# Server
server <- function(input, output) {

  # ── Value Boxes ─────────────────────────────────────────────────────────────
  output$totalRecords <- renderValueBox({
    valueBox(nrow(data), "Total Survey Records",
             icon = icon("database"), color = "blue")
  })
  output$diabetesCases <- renderValueBox({
    valueBox(sum(data$Diabetes_binary == 1), "Diabetes Cases",
             icon = icon("heartbeat"), color = "red")
  })
  output$nonDiabetes <- renderValueBox({
    valueBox(sum(data$Diabetes_binary == 0), "Non-Diabetic Individuals",
             icon = icon("user"), color = "green")
  })

  # ── Overview Plots ──────────────────────────────────────────────────────────
  output$diabetesPlot <- renderPlot({
    ggplot(data, aes(x = factor(Diabetes_binary))) +
      geom_bar(fill = "#2c7fb8") +
      theme_minimal(base_size = 14) +
      labs(x = "Diabetes Status", y = "Count")
  })

  output$factorPlot <- renderPlot({
    ggplot(data, aes_string(x = input$variable,
                            fill = "factor(Diabetes_binary)")) +
      geom_bar(position = "fill") +
      theme_minimal(base_size = 14) +
      labs(x = input$variable, y = "Proportion", fill = "Diabetes") +
      scale_fill_manual(values = c("#2c7fb8","#d73027"),
                        labels = c("No","Yes"))
  })

  output$agePlot <- renderPlot({
    ggplot(data, aes(x = factor(Age), fill = factor(Diabetes_binary))) +
      geom_bar(position = "fill") +
      theme_minimal(base_size = 14) +
      labs(x = "Age Group", y = "Proportion", fill = "Diabetes") +
      scale_fill_manual(values = c("#2c7fb8","#d73027"),
                        labels = c("No","Yes"))
  })

  output$corrPlot <- renderPlot({
    corrplot(cor(data), method = "color", tl.cex = 0.7)
  })

  # ── Analysis: Viz 1 – BMI Density ──────────────────────────────────────────
  output$bmiPlot <- renderPlot({
    df <- data %>%
      mutate(DiabetesLabel = ifelse(Diabetes_binary == 1,
                                    "Diabetic","Non-Diabetic"))
    ggplot(df, aes(x = BMI, fill = DiabetesLabel, colour = DiabetesLabel)) +
      geom_density(alpha = 0.45, linewidth = 0.8) +
      scale_fill_manual(values   = c("Diabetic"="#d73027","Non-Diabetic"="#2c7fb8")) +
      scale_colour_manual(values = c("Diabetic"="#d73027","Non-Diabetic"="#2c7fb8")) +
      theme_minimal(base_size = 14) +
      labs(x = "BMI", y = "Density", fill = NULL, colour = NULL) +
      theme(legend.position = "top")
  })

  # ── Analysis: Viz 2 – Risk Factor Prevalence ───────────────────────────────
  output$riskFactorsPlot <- renderPlot({
    risk_vars   <- c("HighBP","HighChol","Smoker","PhysActivity","Fruits","Veggies")
    risk_labels <- c("High BP","High Chol","Smoker","Phys Active","Eats Fruits","Eats Veggies")

    risk_df <- data.frame(
      Factor = factor(risk_labels, levels = risk_labels),
      Diabetic    = sapply(risk_vars, function(v)
        mean(data[[v]][data$Diabetes_binary == 1]) * 100),
      NonDiabetic = sapply(risk_vars, function(v)
        mean(data[[v]][data$Diabetes_binary == 0]) * 100)
    )

    risk_long <- pivot_longer(risk_df, -Factor,
                              names_to  = "Group",
                              values_to = "Prevalence")

    ggplot(risk_long, aes(x = Factor, y = Prevalence, fill = Group)) +
      geom_col(position = "dodge", width = 0.65) +
      scale_fill_manual(values = c("Diabetic"="#d73027","NonDiabetic"="#2c7fb8"),
                        labels = c("Diabetic","Non-Diabetic")) +
      theme_minimal(base_size = 13) +
      labs(x = NULL, y = "Prevalence (%)", fill = NULL) +
      theme(axis.text.x = element_text(angle = 25, hjust = 1),
            legend.position = "top")
  })

  # ── Analysis: Viz 3 – Diabetes Rate by Sex ─────────────────────────────────
  output$sexPlot <- renderPlot({
    sex_df <- data %>%
      mutate(SexLabel = ifelse(Sex == 1, "Male", "Female")) %>%
      group_by(SexLabel) %>%
      summarise(DiabetesRate = mean(Diabetes_binary) * 100, .groups = "drop")

    ggplot(sex_df, aes(x = SexLabel, y = DiabetesRate, fill = SexLabel)) +
      geom_col(width = 0.5, show.legend = FALSE) +
      geom_text(aes(label = sprintf("%.1f%%", DiabetesRate)),
                vjust = -0.5, size = 5, fontface = "bold") +
      scale_fill_manual(values = c("Female"="#e08080","Male"="#2c7fb8")) +
      scale_y_continuous(limits = c(0, 60)) +
      theme_minimal(base_size = 14) +
      labs(x = "Sex", y = "Diabetes Rate (%)")
  })

  # ── Analysis: Viz 4 – Income vs Diabetes Rate ──────────────────────────────
  output$incomePlot <- renderPlot({
    income_labels <- c("<$10K","$10-15K","$15-20K","$20-25K",
                       "$25-35K","$35-50K","$50-75K",">$75K")
    income_df <- data %>%
      group_by(Income) %>%
      summarise(DiabetesRate = mean(Diabetes_binary) * 100, .groups = "drop") %>%
      mutate(IncomeLabel = factor(income_labels[Income], levels = income_labels))

    ggplot(income_df, aes(x = IncomeLabel, y = DiabetesRate)) +
      geom_line(aes(group = 1), colour = "#d73027", linewidth = 1.2) +
      geom_point(colour = "#d73027", size = 3.5) +
      theme_minimal(base_size = 13) +
      labs(x = "Income Bracket", y = "Diabetes Rate (%)") +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
  })

  # ── Analysis: Viz 5 – Physical & Mental Health Days (Violin) ───────────────
  output$healthDaysPlot <- renderPlot({
    hd_long <- data %>%
      mutate(DiabetesLabel = ifelse(Diabetes_binary == 1,
                                    "Diabetic","Non-Diabetic")) %>%
      select(DiabetesLabel, PhysHlth, MentHlth) %>%
      pivot_longer(c(PhysHlth, MentHlth),
                   names_to  = "HealthType",
                   values_to = "Days") %>%
      mutate(HealthType = recode(HealthType,
               PhysHlth = "Poor Physical Health Days",
               MentHlth = "Poor Mental Health Days"))

    ggplot(hd_long, aes(x = DiabetesLabel, y = Days, fill = DiabetesLabel)) +
      geom_violin(trim = TRUE, alpha = 0.7) +
      geom_boxplot(width = 0.12, outlier.shape = NA,
                   colour = "white", fill = "white", alpha = 0.6) +
      facet_wrap(~ HealthType) +
      scale_fill_manual(values = c("Diabetic"="#d73027","Non-Diabetic"="#2c7fb8")) +
      theme_minimal(base_size = 13) +
      labs(x = NULL, y = "Days (last 30)", fill = NULL) +
      theme(legend.position = "none")
  })

  # ── Analysis: Viz 6 – BP × Cholesterol Heatmap Tile ───────────────────────
  output$bpCholPlot <- renderPlot({
    bp_chol <- data %>%
      mutate(
        BPLabel   = ifelse(HighBP   == 1, "High BP",   "Normal BP"),
        CholLabel = ifelse(HighChol == 1, "High Chol", "Normal Chol")
      ) %>%
      group_by(BPLabel, CholLabel) %>%
      summarise(DiabetesRate = mean(Diabetes_binary) * 100, .groups = "drop")

    ggplot(bp_chol, aes(x = BPLabel, y = CholLabel, fill = DiabetesRate)) +
      geom_tile(colour = "white", linewidth = 1.5) +
      geom_text(aes(label = sprintf("%.1f%%", DiabetesRate)),
                size = 6, fontface = "bold", colour = "white") +
      scale_fill_gradient(low = "#74add1", high = "#d73027",
                          name = "Diabetes\nRate (%)") +
      theme_minimal(base_size = 14) +
      labs(x = "Blood Pressure", y = "Cholesterol") +
      theme(panel.grid = element_blank())
  })
}

shinyApp(ui, server)