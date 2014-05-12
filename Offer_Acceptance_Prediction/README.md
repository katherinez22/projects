Project: Offer Acceptance Prediction and Profit Optimization
==================

### Instructions ###
There are three datasets used for this projects. The `training.csv` contains observations of product offered to customers. Observation is labeled as 1 if the customer accepted the offer and is labeled as 0 if the customer declined the offer. The `testing.csv` contains fewer observations, but has the same information as in the `training.csv`. The third file `other.csv` contains randomly selected records from `training.csv`. Data set is not be included in this page to protect the data provider.

### Approach of this project ###
(1) I built 5 different models for this project using `Python` and performed data manipulation in `R`. To evaluate the model performance, I used the following measures:
```
a. Area Under the ROC Curve (AUC) greater than a certain threshold on training dataset using N-folds cross-validation (The value of N is chosen when a model yields a high AUC and an insignificant difference of AUC between training and validation data sets.)
b. No overfitting
c. Predict reasonable results
```

(2) By using the model constructed in (1) and a for-loop algorithm, I found the optimal product price for the records in the `other.csv` file. Then I segmented the customers into two groups based on their price sensitivity using K-means clustering algorithm. Finally, I visualized the profit as a function of price and the two groups of customers using `matplotlib` in Python. 





