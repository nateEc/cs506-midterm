# Predicting Amazon Movie Review Ratings with XGBoost

## Introduction

In this project, I aimed to predict the star ratings of Amazon movie reviews using an XGBoost classifier. Given the complexity of text data in reviews, I implemented several strategies to enhance model performance. This write-up details the algorithm, special tricks applied, and the assumptions made throughout the process.

## Data Description

The dataset consists of user reviews for various Amazon products, which includes features such as:

- `ProductId`: Unique identifier for the product
- `UserId`: Unique identifier for the user
- `HelpfulnessNumerator`: Number of users who found the review helpful
- `HelpfulnessDenominator`: Total number of users who indicated whether they found the review helpful
- `Score`: Target variable representing ratings from 1 to 5
- `Summary`: Brief summary of the review
- `Text`: Detailed text of the review

The goal was to predict the `Score` based on various extracted features, including textual data.

## Algorithm Overview

I chose the XGBoost classifier due to its efficiency and performance with structured data, especially when combined with the high-dimensional nature of text data. XGBoost implements gradient boosting algorithms that optimize for both speed and model accuracy.

### Feature Engineering

Feature engineering played a crucial role in improving model performance. I implemented the following features:

1. **Helpfulness Metrics**: Created a `Helpfulness` feature by calculating the ratio of `HelpfulnessNumerator` to `HelpfulnessDenominator`. This metric indicates how helpful a review is perceived to be.

2. **Text Length Metrics**: Measured both the length of the review text and summary in words and characters to capture verbosity.

3. **Time Features**: Extracted year and month from the review timestamp to see if there were any trends over time.

4. **Sentiment Analysis**: Used the `TextBlob` library for sentiment analysis, extracting polarity and subjectivity scores from both the review text and summary. These features provide insight into the emotional tone of the reviews.

5. **Caps and Unique Word Count**: Counted the number of capitalized words and unique words in the review text to identify emphasis and diversity in vocabulary.

### Vectorization

For transforming the textual data, I utilized the **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization method, which effectively captures the importance of words in the context of the entire dataset. The following steps were taken:

- Vectorized the `Text` column with a maximum of 5000 features.
- Vectorized the `Summary` column with a maximum of 1000 features.
- Combined these features with numeric features using `scipy.sparse.hstack`, which efficiently handles high-dimensional data.

### Model Training and Evaluation

To evaluate the model, I used an 80/20 train-validation split:

1. **XGBoost Configuration**: Configured the XGBoost model with parameters such as `n_estimators`, `learning_rate`, `max_depth`, and others. I initially set `n_estimators` to 600 to allow for sufficient training while managing overfitting.

2. **Cross-Validation**: To ensure robust performance, I implemented k-fold cross-validation. This method provides a better estimate of the model's performance across different subsets of the data.

3. **Performance Metrics**: The primary metric for evaluation was accuracy, but I also monitored other metrics like precision, recall, and F1-score, especially given the potential for class imbalance in ratings.

## Special Tricks for Performance Improvement

Throughout the modeling process, I implemented several special tricks to enhance performance:

- **Sampling**: Initially, I worked with a sample of the training data (50%) to speed up the experimentation process. This allowed me to iterate quickly on feature engineering and model tuning without waiting for long training times.

- **Hyperparameter Tuning**: Although I did not implement Grid Search in this initial version, I plan to use it in future iterations to systematically explore combinations of hyperparameters for optimal performance.

- **Feature Selection**: I monitored feature importance after training the model to determine which features contributed most to predictions. This process led to the removal of less impactful features, streamlining the model.

## Assumptions Made

Several assumptions were made during the development of this model:

1. **Data Quality**: It was assumed that the data provided was clean and reliable, which could affect the model's performance if there were significant noise or outliers.

2. **Linear Relationships**: The modeling approach assumes that the relationships between features and the target variable (ratings) can be captured effectively by the boosting algorithm.

3. **Feature Independence**: It was assumed that the features are independent of each other, which is a common assumption in many statistical models. However, real-world data may have multicollinearity.

## Conclusion

The implementation of the XGBoost algorithm for predicting Amazon movie review ratings proved effective, achieving an accuracy of approximately 57.7% on the sampled dataset, around 63% Through careful feature engineering, effective vectorization of textual data, and strategic model evaluation, I was able to leverage patterns within the dataset to enhance predictive performance. Future work will focus on further tuning, exploring more advanced text preprocessing techniques, and potentially experimenting with deep learning approaches.




