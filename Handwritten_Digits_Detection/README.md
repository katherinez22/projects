Project: Handwritten Digits Detection
==================

This project is to to detect whether a handwritten digit is less than ’9’ (label = 1) or equal to nine (label = -1). In this project, I trained a Support Vector Machine classifier to detect these handwritten digits. 

### Approach of this project ###

(1) Load the file letters_training.csv and set all but the last column as features, and the last column as the target. Open the file in universal line mode for reading using the flags ’rU’. 

(2) Train the SVM classifier using a 70/15/15 random partition and a RBF kernel. Investigate the classifier performance on the test set and compare it with a linear SVM classifier.

(3) Read the file 9_54.txt convert the row-major ordered matrix into a one-dimensional array. N.B. row-major ordering is equivalent to western reading order, i.e. from left to right and then down to the next line etc. Check that your classifier predicts that this file represents a handwritten ’9’. 

(4) Note which features are constant over the training set. Using this observation, reduce the dimension of the feature vectors, retrain the SVM classifier and measure the performance of the SVM classifier using either a rbf or a linear kernel.

### Results ###
The SVM classifier I built successfully classified handwritten digits with 95% accuracy. Also, I reduced the dimentionality of feature space to optimize the classification. For more details about this project, please find the code in `answer.py`.

