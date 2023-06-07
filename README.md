# A2
# So, we have Breast Cancer Wisconsin (Diagnostic) dataset and I used two different algorithms: Support Vector Machines (SVM) and Random Forest Classifier as two different models.

# First, I imported the necessary libraries like pandas, numpy, and sci-kit-learn modules for data manipulation, preprocessing, and model evaluation. 

# Next, I loaded the dataset from the provided URL. The dataset contains information about various features of breast cancer cells and the corresponding diagnosis (benign or malignant). The dataset is read into a pandas DataFrame, and I gave them the column names . Then I checked the overview of the dataset.  

# As there are missing values in the 'bare_nuclei' column I replaced the occurences of '?' with NaN values using NumPy's np.nan and dropped these rows usin dropna() method. Then I converted 'bare_nuclei' column to numeric type so that the data is in suitable format for training the model.
# Then I split the dataset into features(X) and target variable(y). To split the dataset into training and testing datasets i used train_test_split function, where I used 80% of data as training and 20% of data as testing datasets.
# For Scaling the features I used StandardScalar() to make sure all the features are on a similar scale which improves the performance of the model.
# I used SVM model as model1. Then I used fit() method to train the SVM model using the training data. I finally calculated the accuracy of the model and also created a confusion matrix to evaluate the model's performance.

# For the second model, I used RandomForestClassifier(), trained it and calculated the accuracy, and created a confusion matrix.