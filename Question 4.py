import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

kidney_disease_dataframe = pandas.read_csv("kidney_disease.csv")  #read the dataset into a dataframe

kidney_disease_dataframe["classification"] = kidney_disease_dataframe["classification"].str.strip()  #remove extra spaces from labels
kidney_disease_dataframe["classification"] = kidney_disease_dataframe["classification"].replace("?", "notckd")  #replace missing labels with 'notckd'

feature_matrix_X = pandas.get_dummies(kidney_disease_dataframe.drop("classification", axis=1))  #convert categorical features to numeric
vector_label_y = kidney_disease_dataframe["classification"]  #extract the target labels

imputer = SimpleImputer(strategy="mean")  #create imputer to fill missing numeric values
feature_matrix_X_imputed = imputer.fit_transform(feature_matrix_X)  #apply imputer to feature matrix

X_train, X_test, Y_train, Y_test = train_test_split(
    feature_matrix_X_imputed, vector_label_y, test_size=0.3, random_state=20, stratify=vector_label_y
)  #split data into training and testing sets

knn_model = KNeighborsClassifier(n_neighbors=5)  #create KNN classifier with k=5
trained_knn_model = knn_model.fit(X_train, Y_train)  #train the model on training data
predicted_y = trained_knn_model.predict(X_test)  #predict labels for test data

cm = confusion_matrix(Y_test, predicted_y)  #compute confusion matrix
accuracy = accuracy_score(Y_test, predicted_y)  #calculate accuracy
precision = precision_score(Y_test, predicted_y, average='binary', pos_label='ckd')  #calculate precision for 'ckd'
recall = recall_score(Y_test, predicted_y, average='binary', pos_label='ckd')  #calculate recall for 'ckd'
f1 = f1_score(Y_test, predicted_y, average='binary', pos_label='ckd')  #calculate F1-score for 'ckd'

print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

"""
Q1: A true positive means that when the model predicts that the patient has kidney disease, 
they actually have kidney disease. A true negative is when the model predicts that the patient 
does not have kidney disease, and they actually do not. A false positive is when the model predicts 
that the patient has kidney disease when they in fact do not. A false negative is when the model 
predicts that the patient does not have kidney disease when they actually do.

Q2: Accuracy measures the proportion of correct predictions, but if the dataset is imbalanced, 
it may be misleading. For example, if there are more healthy patients and the model predicts 
everyone is healthy, the accuracy will still be high despite missing all kidney disease cases.

Q3: In kidney disease prediction, recall (sensitivity) is the most important metric because it 
tells us the proportion of actual kidney disease cases the model correctly identifies. Missing a 
kidney disease case (a false negative) could have serious consequences, so it is better to catch 
as many true cases as possible, even if some healthy patients are incorrectly flagged.
"""
