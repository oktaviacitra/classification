# Classification
This is the documentation of my learning journey about classification

### StrokePrediction.ipynb [<a href="https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset">Brain stroke prediction dataset</a>]
The file contains few steps. Those are preprocessing data to generate normalization of each row and column, change the unknown values in the 'smoking_status' column with prediction results using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression</a>, and predict stroke status using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html">SVM SVC</a> from sklearn.

### VillainClassification.ipynb [<a href="https://www.kaggle.com/datasets/rogeriovaz/villains-image-classification">Villains - Image Classification</a>]
Look for the optimum <a href="https://www.tensorflow.org/tutorials/images/cnn">Convolutional Neural Networks</a> model step by step using <a href="https://www.tensorflow.org/tutorials/keras/keras_tuner">Keras Tuner</a> to get hyperparameter tuning. There are preprocessing images, splitting data, tuning model, building model, training model, and evaluating model. 

### BreastCancer.ipynb [<a href="https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset">Breast Cancer Dataset</a>]
Find best features to create a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">Random Forest Classifier</a> model using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html">SelectKBest</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a>. There are preprocessing dataframe, looking for k of features selection, splitting data, building model, model fit, and evaluating model. 

### WaterQuality.ipynb [<a href="https://www.kaggle.com/datasets/adityakadiwal/water-potability">Water Quality</a>]
Look for the optimum <a href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html">Neural Networks</a> model using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> to get hyperparameter tuning. There are preprocessing dataframe, looking for k of features selection, splitting data, building model, model fit, and evaluating model. 
