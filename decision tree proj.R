df = read.csv("weatherAUS.csv", header= TRUE, stringsAsFactors = TRUE); 
df$Date <- as.numeric(as.Date(df$Date))
dim(df)
df1 = na.omit(df)
dim(df1)

# Function to remove outliers based on IQR
remove_outliers_iqr <- function(df, cols) {
  for (col in cols) {
    q1 <- quantile(df[[col]], 0.25)
    q3 <- quantile(df[[col]], 0.75)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    
    outliers <- (df[[col]] < lower_bound) | (df[[col]] > upper_bound)
    df <- df[!outliers, ]
  }
  return(df)
}
all_vars = c("Date", "Location", "MinTemp", "MaxTemp",  
             "Rainfall", "Evaporation", "Sunshine", 
             "WindGustDir", "WindGustSpeed", 
             "WindDir9am", "WindDir3pm", 
             "WindSpeed9am", "WindSpeed3pm",  
             "Humidity9am", "Humidity3pm", 
             "Pressure9am", "Pressure3pm", 
             "Cloud9am", "Cloud3pm", 
             "Temp9am", "Temp3pm")


num_vars = c("MinTemp", "MaxTemp",
             "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
             "WindSpeed9am", "WindSpeed3pm",  
             "Humidity9am", "Humidity3pm", 
             "Pressure9am", "Pressure3pm", 
             "Cloud9am", "Cloud3pm", 
             "Temp9am", "Temp3pm","Date")

df2 <- remove_outliers_iqr(df1, num_vars)
dim(df2)

str(df2)

library(rpart)
library(rpart.plot)



# Assuming your dataframe is named 'df'
# Selecting only the relevant variables
df_subset1 <- df2[, c("RainTomorrow", "MaxTemp", "MinTemp", "Rainfall",
                      "Evaporation", "Sunshine", 
                      "WindGustDir", "WindGustSpeed", 
                      "WindDir9am", "WindDir3pm", 
                      "WindSpeed9am", "WindSpeed3pm",  
                      "Humidity9am", "Humidity3pm", 
                      "Pressure9am", "Pressure3pm", 
                      "Cloud9am", "Cloud3pm", 
                      "Temp9am", "Temp3pm","Date")]

set.seed(69)
trainIndex1 <- sample(seq_len(nrow(df_subset1)), size = 0.7 * nrow(df_subset1))
training_data1 <- df_subset1[trainIndex1, ]
testing_data1 <- df_subset1[-trainIndex1, ]

# Build a decision tree model using the training set
tree_model1 <- rpart(RainTomorrow ~ WindGustSpeed + Evaporation + Sunshine +
                       MaxTemp + WindSpeed3pm + MinTemp + Temp3pm + Temp9am + Cloud9am +
                       Humidity9am + Humidity3pm + Pressure9am + Pressure3pm + Cloud3pm +
                       WindSpeed9am + Date + Rainfall,
                     data = training_data1,
                     method = "class",
                     parms = list(split = "gini"),
                     control = rpart.control(xval = 20, cp = 0.001, minsplit = 120))

# Visualize the decision tree
rpart.plot(tree_model1, extra = 1, type =1)

# Make predictions on the testing set
predictions <- predict(tree_model1, newdata = training_data1, type = "class")
actual <-  training_data1$RainTomorrow

# Evaluate the model performance
cm <- table(predictions, actual); cm
pt<- prop.table(cm); pt
options(digits = 5) 
accuracy = pt[1,1] + pt[2,2]; accuracy
sensitivity = cm[2,2] / (cm[1,2] + cm[2,2]); sensitivity # TP/P
specificity = cm[1,1] / (cm[1,1] + cm[2,1]); specificity # TN/N

FPR = cm[2,1] / (cm[2,1] + cm[1,1]); FPR
FNR = cm[1,2] / (cm[1,2] + cm[2,2]); FNR

install.packages("caret")
library(caret)
conf_matrix <- confusionMatrix(as.factor(actual), as.factor(predictions))
fourfoldplot(cm, color = c("#CC6666", "#99CC99"),
conf.level = 0,main = "Confusion Matrix for Train Data")

predictions <- predict(tree_model1, newdata = testing_data1, type = "class")
actual <-  testing_data1$RainTomorrow

cm1 <- table(predictions, actual); cm1
pt<- prop.table(cm1); pt
accuracy = pt[1,1] + pt[2,2]; accuracy
sensitivity = cm1[2,2] / (cm1[1,2] + cm1[2,2]); sensitivity # TP/P
specificity = cm1[1,1] / (cm1[1,1] + cm1[2,1]); specificity # TN/N
FPR = cm1[2,1] / (cm1[2,1] + cm1[1,1]); FPR
FNR = cm1[1,2] / (cm1[1,2] + cm1[2,2]); FNR


conf_matrix <- confusionMatrix(as.factor(actual), as.factor(predictions))
fourfoldplot(cm1, color = c("#CC6666", "#99CC99"),
             conf.level = 0,main = "Confusion Matrix for Test Data")

