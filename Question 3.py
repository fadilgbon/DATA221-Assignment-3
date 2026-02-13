import pandas
from sklearn.model_selection import train_test_split
kidney_disease_dataframe = pandas.read_csv("kidney_disease.csv")

feature_matrix_X = kidney_disease_dataframe.drop("classification", axis=1)
feature_matrix_Y = kidney_disease_dataframe["classification"]

X_train, X_test, Y_train, Y_test = train_test_split(feature_matrix_X, feature_matrix_Y,test_size=0.3, random_state=20)
# if we train and test the same dataset then instead of learning from the data the model will just memorize it leading to overfitting
#the test set is used to see how well the model is trained. Using unseen data will give a realistic view on how well it works in real world scenerios