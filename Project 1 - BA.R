install.packages("Hmisc")
library(Hmisc)

install.packages("fastDummies")
library(fastDummies)

install.packages("caret")
library(caret)


df = read.csv("weatherAUS.csv")
head(df)
dim(df)
str(df)
summary(df)



# Evaporation, Sunshine, Cloud9am, Cloud3pm has more than 50k missing Values
# So drop the columns

# Removing NA values from other columns
df1 = na.omit(df)
dim(df1)

all_cols = c("Location", "MinTemp", "MaxTemp",  
             "Rainfall", "Evaporation", "Sunshine", 
             "WindGustDir", "WindGustSpeed", 
             "WindDir9am", "WindDir3pm", 
             "WindSpeed9am", "WindSpeed3pm",  
             "Humidity9am", "Humidity3pm", 
             "Pressure9am", "Pressure3pm", 
             "Cloud9am", "Cloud3pm", 
             "Temp9am", "Temp3pm")

cat_cols = c("Location", "WindGustDir", "WindDir9am", "WindDir3pm" )

num_cols = c("Year", "Month", "Day", "MinTemp", "MaxTemp",  
             "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
             "WindSpeed9am", "WindSpeed3pm",  
             "Humidity9am", "Humidity3pm", 
             "Pressure9am", "Pressure3pm", 
             "Cloud9am", "Cloud3pm", 
             "Temp9am", "Temp3pm")

# Converting Categorical Variables to Factors
for (col in cat_cols) {
  df1[,col] = factor(df1[,col])
}

# converting Date variable to numeric
#df1$Date_num = as.numeric(as.Date(df1$Date))
df1$Month = as.numeric(format(as.Date(df1$Date), "%m"))
df1$Year = as.numeric(format(as.Date(df1$Date), "%y"))
df1$Day = as.numeric(format(as.Date(df1$Date), "%d"))

# Function to remove outliers based on IQR
remove_outliers <- function(df, cols) {
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

df2 <- remove_outliers(df1, num_cols)
dim(df2)

# Removing duplicated rows
df2 <- df2[!duplicated(df2),]
dim(df2)

#### LOGISTIC REGRESSION ####

# Scaling
df3 <- df2
min_max_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

df3[, num_cols] <- lapply(df3[, num_cols], min_max_scale)

str(df3)
head(df3)
dim(df3)

# One Hot Encoding
df3$RainTomorrow_1hot <- ifelse(df3$RainTomorrow == "Yes", 1, 0)

length(df3$RainTomorrow_1hot)
length(df3$RainTomorrow)

df3$RainToday_1hot <- ifelse(df3$RainToday == "Yes", 1, 0)

# Train - Test Split
set.seed(69)
train = sample(1:nrow(df3), nrow(df3)*(2/3))
df3.train = df3[train,]
df3.test = df3[-train,]

dim(df3.train)
dim(df3.test)

# Logit

logit.reg <- glm(RainTomorrow_1hot ~ Location + MinTemp + MaxTemp+ Rainfall
                 + Evaporation+ Sunshine + WindGustSpeed + WindSpeed9am + WindSpeed3pm
                 + Humidity9am+ Humidity3pm + Pressure9am + Pressure3pm+ Cloud9am + Cloud3pm
                 + Temp9am + Temp3pm,data = df3.train, family = "binomial")
                  
summary(logit.reg)


# Performance on Train

logitPredict <- predict(logit.reg, df3.train, type = "response")
logitPredictClass <- ifelse(logitPredict> 0.5, "Yes", "No")

actual <- df3.train$RainTomorrow
predict <- logitPredictClass
cm <- table(predict, actual); cm
pt <- prop.table(cm); pt

options(digits = 3)

accuracy = pt[1,1] + pt[2,2]; accuracy
sensitivity = cm[2,2] / (cm[1,2] + cm[2,2]); sensitivity # TP/P
specificity = cm[1,1] / (cm[1,1] + cm[2,1]); specificity # TN/N
FPR = cm[2,1] / (cm[2,1] + cm[1,1]); FPR # FP / FP + TN
FNR = cm[1,2] / (cm[1,2] + cm[2,2]); FNR # FN / FN + TP

sprintf("Performance on Train Data: Accuracy: %.2f, Sensitivity: %.2f, Specificity: %.2f", 
        accuracy, sensitivity, specificity);

conf_matrix <- confusionMatrix(as.factor(actual), as.factor(predict))
fourfoldplot(cm, color = c("#CC6666", "#99CC99"), 
             conf.level = 0, main = "Confusion Matrix for Train Data")

# Performance on Test

logitPredict <- predict(logit.reg, df3.test, type = "response")
logitPredictClass <- ifelse(logitPredict> 0.5, "Yes", "No")

actual <- df3.test$RainTomorrow
predict <- logitPredictClass
cm <- table(predict, actual); cm
pt <- prop.table(cm); pt

options(digits = 3)

accuracy = pt[1,1] + pt[2,2]; accuracy
sensitivity = cm[2,2] / (cm[1,2] + cm[2,2]); sensitivity # TP/P
specificity = cm[1,1] / (cm[1,1] + cm[2,1]); specificity # TN/N
FPR = cm[2,1] / (cm[2,1] + cm[1,1]); FPR # FP / FP + TN
FNR = cm[1,2] / (cm[1,2] + cm[2,2]); FNR # FN / FN + TP

sprintf("Performance on Test Data: Accuracy: %.2f, Sensitivity: %.2f, Specificity: %.2f", 
        accuracy, sensitivity, specificity);

conf_matrix <- confusionMatrix(as.factor(actual), as.factor(predict))
fourfoldplot(cm, color = c("#CC6666", "#99CC99"), 
             conf.level = 0, 
             main = "Confusion Matrix for Test Data")
