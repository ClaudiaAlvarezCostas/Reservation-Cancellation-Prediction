# ℹ️ Overview
A few months ago, I applied for a Data Science role at a hotel company. To demonstrate that I was qualified for the position, they asked me to solve the following problem:

*Could it be possible to design a model that optimizes overbooking to compensate for reservation cancellations in a hotel?*

The hotel is located in Paris, and the company operates several other hotels in the city. This is an important detail because, in cases of overbooking, the hotel has
the option to redirect guests to one of their close properties.

The main objective, therefore, is to design a model that allows the hotel to accept more reservations than the available rooms, in anticipation of cancellations.
The key challenge is to ensure that the model is more inclined to overbook rather than under-predict cancellations.

Okay, fine, I understood the task... but where should I start?

# 🌟 Methodology and Process Overview
##  📦 The Data
They delivered me a huge data base (".xlsx" file) where I could find variables of different classes (date, integer, character...). The dependent variable I need to predict is called *IsCanceled*, as the ultimate goal is to determine whether a customer will cancel a reservation or not. 

The first step must be to find possible problems to avoid mistakes during the test. In particular, I was confussed by the meaning of the columns *StaysInWeekendNights* and *StaysInWeekNights* since they were described to me as *number of nights during the week* and *number of nights during the weekend*. I determined, by doing a quick comprobation, that it was due to a mistake writing the exercise statement and the varibles were okay in the file (the title and the range of each variable were coordinated).

After that, I went on by doing a descriptive analysis of the data.

### Descriptive analysis results
 During this first section of the project, I classified the original variables in type groups (numeric integer, numeric decimal, date, categorical...) and applied format transformations when they were needed. This step is especially important before generating graphs, as visualization tools interpret data types differently. For instance, categorical variables require distinct treatment compared to numeric ones, with each type influencing the choice of visualization.

### Redundant variables
 After analyzing the dataset, I tried to reduced the number of variables by removing those that were redundant, specifically through correlation matrix analysis and examining collinearity. In addition to reducing variables, I also examined the class balance across categorical variables. For predictive modeling, balanced data is crucial because imbalanced classes can lead to biased models, particularly in classification tasks (as this one). 

## ✏️ Model Selection and Evaluation
By following the steps outlined above, I have gained a thorough understanding of the dataset. Now, I am ready to proceed with predicting the *IsCanceled* variable and working toward a solution to the problem.

### Explanatory variables
Before considering which models might best solve the task, it's crucial to reduce the number of potential explanatory variables. By focusing only on the most relevant variables, we can improve model performance and interpretability.

To determine which variables have meaningful relationships with the target variable (*IsCanceled*), I used statistical tests. For numerical variables, I conducted T-tests and for categorical variables, I applied Pearson's Chi-squared test.

### Models

* **SVM (Support Vector Machine)**: This could be interesting because it handles imbalanced classes relatively well.
* **Decision Trees**: Not suitable in this case because of the imbalance in class distributions, which could make the model biased toward the majority class.
* **Logistic Regression**: Not ideal because the variables show collinearity and the class distribution is imbalanced. Logistic regression tends to perform poorly under these conditions.
* **Neural Networks**: While potentially interesting, neural networks are complex "black-box" models, making it difficult to interpret how they arrive at their decisions.
* **Random Forest**: Could be a good choice because it can handle imbalanced data and does not require the data to be scaled.
* **Gradient Boosting**: Sensitive to outliers, and since we observed many outliers in the boxplots, this model might not perform well here.
* **Naive Bayes**: Not ideal as it requires assuming specific data distributions, which may not hold in our dataset.
* **PCA (Principal Component Analysis)**: PCA may not be the best option here due to different data scales, a high number of features, and its sensitivity to outliers.
* **Lasso Regression**: Not very useful in this case, as we only observed high correlation between two variables. Hence, Lasso wouldn't add significant penalization.
  
Based on these considerations, I decided to test with Random Forest (RF) and Support Vector Machine (SVM). I opted to avoid neural networks due to their "black-box" nature, prioritizing more interpretable models.

### Model Evaluation
In this analysis, we implemented several key steps to evaluate and refine our model:

1. **Model Selection and Initial Results**: We confirmed that the Random Forest model was the best performer among the models tested. While it showed promise, further tuning with various hyperparameters would likely improve its performance.

2. **Evaluation**:

     * Accuracy and Kappa Statistic: The model achieved an accuracy close to 90%. However, a Kappa statistic of around 0.2 raised concerns, as this low value suggests that the high accuracy might be due to random chance rather than meaningful patterns.
     * Sensitivity vs. Specificity: We observed a significant discrepancy between sensitivity and specificity. The model predicted non-cancellations accurately (high specificity) but struggled with cancellations (low sensitivity). This is likely due to the dataset's imbalance, with far more non-cancelled reservations than cancelled ones. Consequently, the model’s Balanced Accuracy was only around 0.60, which indicates a moderate, but not ideal, performance.
     * Adjusting Class Weights: We experimented with adjusting class weights to give more importance to the minority class (cancellations). Despite this, there was minimal improvement, as the model continued to prioritize non-cancellations.

3. **Consideration of a Neural Network Model**: Given the limitations of the Random Forest model, we proceeded to test a neural network as the next step. However, this approach did not bring any improvement.

4. **Balancing the Dataset**: Recognizing that the dataset's imbalance contributed to the model's difficulties in predicting cancellations, we proposed balancing the classes by reducing the number of non-cancelled cases. This approach would create a more even class distribution, potentially allowing the model to improve its detection of cancellations by reducing its bias toward predicting non-cancellations.

## ✅ Results
As I suspected, balancing the dataset so that the response variable categories have approximately equal numbers of observations improved the model's performance. Although accuracy scores are slightly lower, this adjustment creates a model that performs better for our specific case. Here’s what we can observe:

* **Error Measurement**: For this project, the most critical error to minimize is when the model predicts "False" (no cancellation) when the actual outcome is "True" (cancellation). After balancing, the model achieved an error rate of 8% for the subset and 12% for the full dataset.
* **Kappa Statistic and Balanced Accuracy**: The Kappa value rose to 0.45, indicating less randomness in predictions, and the Balanced Accuracy improved to around 0.7, showing that the model handles both classes more effectively.
* **Real-World Implementation Consideration**: With this model, the hotel would likely be overprepared for cancellations, as it predicts more cancellations than might actually occur. This situation could allow for better management of resources and potentially overbooking strategies, knowing that some expected cancellations might not happen.

An ideal model would:

* Maintain or improve accuracy measures.
* Show a bias in prediction errors toward "True" (predicting a cancellation even when it doesn't occur) or, ideally, place a higher proportion of correct predictions on the diagonal of the confusion matrix with minimal additional error for "True."

The Random Forest model with a balanced subset of the original dataset performs best. While a perfectly equal number of cancelled and non-cancelled observations is not required, reducing imbalance was beneficial.

Note that I did not apply cross-validation or conduct an exhaustive evaluation here to keep the project concise. Ideally, further steps would include variable selection and fine-tuning.

# ✍️ Author

Claudia Álvarez Costas
