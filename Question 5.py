import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

kidney_disease_dataframe = pandas.read_csv("kidney_disease.csv")  #load the dataset into a dataframe

kidney_disease_dataframe["classification"] = kidney_disease_dataframe["classification"].str.strip()  #remove extra spaces from labels
kidney_disease_dataframe["classification"] = kidney_disease_dataframe["classification"].replace("?", "notckd")  #replace missing labels with 'notckd'

feature_matrix_X = pandas.get_dummies(kidney_disease_dataframe.drop("classification", axis=1))  #convert categorical columns to numeric
vector_label_y = kidney_disease_dataframe["classification"]  #extract the target labels

imputer = SimpleImputer(strategy="mean")  #create imputer to fill missing numeric values
feature_matrix_X_imputed = imputer.fit_transform(feature_matrix_X)  #apply imputer to feature matrix

X_train, X_test, Y_train, Y_test = train_test_split(
    feature_matrix_X_imputed, vector_label_y, test_size=0.3, random_state=20, stratify=vector_label_y
)  #split data into training and testing sets

knn_model = KNeighborsClassifier(n_neighbors=5)  #create KNN classifier with k=5
trained_knn_model = knn_model.fit(X_train, Y_train)  #train the model
predicted_y = trained_knn_model.predict(X_test)  #predict labels for test data

k_values = [1, 3, 5, 7, 9]  #list of k values to test
accuracy_results = {}  #dictionary to store accuracy for each k

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)  #create KNN model with current k
    knn.fit(X_train, Y_train)  #train the model
    y_pred_k = knn.predict(X_test)  #predict using current k
    acc = accuracy_score(Y_test, y_pred_k)  #compute accuracy
    accuracy_results[k] = acc  #store accuracy in dictionary

accuracy_table = pandas.DataFrame(list(accuracy_results.items()), columns=["k", "Test Accuracy"])  #convert results to a table
print("\nKNN Test Accuracy for Different k values:")
print(accuracy_table)

best_k = max(accuracy_results, key=accuracy_results.get)  #find k with highest accuracy
print(f"\nThe value of k with the highest test accuracy is: {best_k}")

"""
1. Changing k affects the model's complexity and sensitivity to training data.
2. Very small k values (like k=1) can cause overfitting because the model follows noise too closely.
3. Very large k values can cause underfitting because the model smooths predictions too much.
4. Choosing the right k balances bias and variance, improving performance on new data.
"""
