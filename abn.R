# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)

# Load the dataset
data <- read.csv("~/Documents/CL_prediction/data/merged_data.csv")

# Convert client_nr to a factor (random effect grouping variable)
data$client_nr <- as.factor(data$client_nr)

# Fit a mixed-effects logistic regression model
# Predicting 'credit_application' with fixed effects (e.g., nr_credit_trx, min_balance)
# and a random intercept for 'client_nr'
data$nr_credit_trx_scaled <- scale(data$nr_credit_trx)
data$min_balance_scaled <- scale(data$min_balance)
data$volume_credit_trx_scaled<- scale(data$volume_credit_trx)
data$volume_debit_trx_scaled<- scale(data$volume_debit_trx)


data$lag_nr_credit_applications <- lag(data$nr_credit_applications, n = 1)

data$lag_nr_credit_applications[is.na(data$lag_nr_credit_applications)] <- 0

# Fit the mixed-effects logistic regression model again
model <- glmer(
  credit_application ~  (1 | yearmonth) + (1 | client_nr) + nr_credit_trx_scaled
  +min_balance_scaled+
    volume_credit_trx_scaled + volume_debit_trx_scaled +lag_nr_credit_applications ,
  data = data,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)



# Summary of the model
summary(model)

# Predict probabilities for visualization
data$predicted_prob <- predict(model, type = "response")
hist(data$predicted_prob)

# Convert probabilities to binary outcomes (threshold = 0.5)
data$predicted_class <- ifelse(data$predicted_prob > 0.5, 1, 0)

# Confusion Matrix to evaluate model performance
table(Predicted = data$predicted_class, Actual = data$credit_application)

# Plot observed vs. predicted probabilities
ggplot(data, aes(x = predicted_prob, y = credit_application)) +
  geom_jitter(height = 0.02, width = 0) +
  geom_smooth(method = "loess", color = "blue") +
  labs(
    title = "Predicted vs Observed Credit Application",
    x = "Predicted Probability",
    y = "Observed Value"
  )

# Optional: Diagnostic plots
plot(model)
