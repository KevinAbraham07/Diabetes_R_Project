library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(corrplot)

data <- read.csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# UI
ui <- dashboardPage(
  
  dashboardHeader(title = "Diabetes Risk Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "dashboard", icon = icon("chart-bar")),
      
      selectInput(
        "variable",
        "Select Lifestyle Factor",
        choices = c(
          "BMI",
          "Smoker",
          "PhysActivity",
          "Fruits",
          "Veggies",
          "HvyAlcoholConsump"
        )
      )
    )
  ),
  
  dashboardBody(
    
    fluidRow(
      
      valueBoxOutput("totalRecords"),
      valueBoxOutput("diabetesCases"),
      valueBoxOutput("nonDiabetes")
      
    ),
    
    fluidRow(
      
      box(
        title = "Diabetes Distribution",
        width = 6,
        status = "primary",
        solidHeader = TRUE,
        plotOutput("diabetesPlot", height = 300)
      ),
      
      box(
        title = "Lifestyle Factor vs Diabetes",
        width = 6,
        status = "warning",
        solidHeader = TRUE,
        plotOutput("factorPlot", height = 300)
      )
      
    ),
    
    fluidRow(
      
      box(
        title = "Age vs Diabetes",
        width = 6,
        status = "success",
        solidHeader = TRUE,
        plotOutput("agePlot", height = 300)
      ),
      
      box(
        title = "Correlation Heatmap",
        width = 6,
        status = "danger",
        solidHeader = TRUE,
        plotOutput("corrPlot", height = 300)
      )
      
    )
    
  )
)

# Server
server <- function(input, output) {
  
  output$totalRecords <- renderValueBox({
    
    valueBox(
      value = nrow(data),
      subtitle = "Total Survey Records",
      icon = icon("database"),
      color = "blue"
    )
    
  })
  
  output$diabetesCases <- renderValueBox({
    
    valueBox(
      value = sum(data$Diabetes_binary == 1),
      subtitle = "Diabetes Cases",
      icon = icon("heartbeat"),
      color = "red"
    )
    
  })
  
  output$nonDiabetes <- renderValueBox({
    
    valueBox(
      value = sum(data$Diabetes_binary == 0),
      subtitle = "Non-Diabetic Individuals",
      icon = icon("user"),
      color = "green"
    )
    
  })
  
  output$diabetesPlot <- renderPlot({
    
    ggplot(data, aes(x = factor(Diabetes_binary))) +
      geom_bar(fill = "#2c7fb8") +
      theme_minimal(base_size = 14) +
      labs(
        x = "Diabetes Status",
        y = "Count"
      )
    
  })
  
  output$factorPlot <- renderPlot({
    
    ggplot(data, aes_string(x = input$variable, fill = "factor(Diabetes_binary)")) +
      geom_bar(position = "fill") +
      theme_minimal(base_size = 14) +
      labs(
        x = input$variable,
        y = "Proportion"
      )
    
  })
  
  output$agePlot <- renderPlot({
    
    ggplot(data, aes(x = factor(Age), fill = factor(Diabetes_binary))) +
      geom_bar(position = "fill") +
      theme_minimal(base_size = 14)
    
  })
  
  output$corrPlot <- renderPlot({
    
    cor_matrix <- cor(data)
    
    corrplot(
      cor_matrix,
      method = "color",
      tl.cex = 0.7
    )
    
  })
}

shinyApp(ui, server)