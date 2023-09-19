# step 1 to import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier



# step 2 to read csv files and create dataframes
legitimate_df = pd.read_csv("structured_data_legitimate.csv")
phishing_df = pd.read_csv("structured_data_phishing.csv")

# step 3 to combine both legi and phish df and shuffle them
df = pd.concat([legitimate_df,phishing_df], axis=0)

df = df.sample(frac=1)

# step 4 to remove url column and duplicates and create x and y values for model
df=df.drop('URL', axis=1)
df=df.drop_duplicates()

X = df.drop('label', axis = 1)
y = df['label']

# step 5 split data to create test and train models
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.2,random_state=10)

# step 6 crete ml models using sk learn
svm_model = svm.LinearSVC()

# Random Forest
rf_model = RandomForestClassifier(n_estimators=60)

# Decision Tree
dt_model = tree.DecisionTreeClassifier()

# AdaBoost
ab_model = AdaBoostClassifier()

# Gaussian Naive Bayes
nb_model = GaussianNB()

# Neural Network
nn_model = MLPClassifier(alpha=1)

# KNeighborsClassifier
kn_model = KNeighborsClassifier()

# Gaussian Process
#gp_model = GaussianProcessClassifier(1.0 * RBF(1.0))

# step 7 train the model
# Convert columns with strings to numeric values
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with a default value
X_train.fillna(0, inplace=True)
y_train.fillna(0, inplace=True)

svm_model.fit(X_train, y_train)



# step 8 make prediction
# Convert columns with strings to numeric values
X_test = X_test.apply(pd.to_numeric, errors='coerce')
y_test = y_test.apply(pd.to_numeric, errors='coerce')
# Fill NaN values with a default value
X_test.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)

# Continue with your code
predictions = svm_model.predict(X_test)


# step 9 cretae confusion matrix and calculate tn tp fn fp
tn, tp, fn, fp = confusion_matrix(y_test, y_pred=predictions).ravel()




# step 10 calculate accuracy precision and recall
accuracy = (tp + tn)/(tp+tn+fp+fn)
precision = tp/ (tp+tn)
recall = tp / (tp+fn)


print("accuracy -->", accuracy)
print("precision -->", precision)
print("recall -->", recall)

# K fold cross validation K=5
K = 5
total = X.shape[0]
index = int(total/ K)

#1
X_1_test = X.iloc[:index]
X_1_train = X.iloc[index:]
y_1_test = y.iloc[:index]
y_1_train = y.iloc[index:]

#2
X_2_test = X.iloc[index:index*2]
X_2_train = X.iloc[np.r_[:index, index*2:]]
y_2_test = y.iloc[index:index*2]
y_2_train = y.iloc[np.r_[:index, index*2:]]

#3
X_3_test = X.iloc[index*2:index*3]
X_3_train = X.iloc[np.r_[:index*2, index*3:]]
y_3_test = y.iloc[index*2:index*3]
y_3_train = y.iloc[np.r_[:index*2, index*3:]]

#4
X_4_test = X.iloc[index*3:index*4]
X_4_train = X.iloc[np.r_[:index*3, index*4:]]
y_4_test = y.iloc[index*3:index*4]
y_4_train = y.iloc[np.r_[:index*3, index*4:]]

#5
X_5_test = X.iloc[index*4:]
X_5_train = X.iloc[np.r_[:index*4]]
y_5_test = y.iloc[index*4:]
y_5_train = y.iloc[np.r_[:index*4]]


# X and y train test list
X_train_list = [X_1_train, X_2_train, X_3_train, X_4_train, X_5_train]
X_test_list = [X_1_test, X_2_test, X_3_test, X_4_test, X_5_test]

y_train_list = [y_1_train, y_2_train, y_3_train, y_4_train, y_5_train]
y_test_list = [y_1_test, y_2_test, y_3_test, y_4_test, y_5_test]


def calculate_measures(TN, TP, FN, FP):
    model_accuracy = (TP+TN)/(TP + TN + FP + FN)
    model_precision = TP / (TP + FP)
    model_recall = TP / (TP + FN)
    return model_accuracy, model_precision, model_recall

rf_accuracy_list, rf_precision_list, rf_recall_list = [], [], []
dt_accuracy_list, dt_precision_list, dt_recall_list = [], [], []
ab_accuracy_list, ab_precision_list, ab_recall_list = [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list = [], [], []
svm_accuracy_list, svm_precision_list, svm_recall_list = [], [], []
nb_accuracy_list, nb_precision_list, nb_recall_list = [], [], []
nn_accuracy_list, nn_precision_list, nn_recall_list = [], [], []
kn_accuracy_list, kn_precision_list, kn_recall_list = [], [], []
#gp_accuracy_list, gp_precision_list, gp_recall_list = [], [], []

for i in range(0, K):
    #-------Random Forest---------#
    # Convert columns with strings to numeric values
    X_train_list[i] = X_train_list[i].apply(pd.to_numeric, errors='coerce')
    y_train_list[i] = y_train_list[i].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with a default value
    X_train_list[i].fillna(0, inplace=True)
    y_train_list[i].fillna(0, inplace=True)

    rf_model.fit(X_train_list[i], y_train_list[i])

    # Convert columns with strings to numeric values
    X_test_list[i] = X_test_list[i].apply(pd.to_numeric, errors='coerce')
    y_test_list[i] = y_test_list[i].apply(pd.to_numeric, errors='coerce')
    # Fill NaN values with a default value
    X_test_list[i].fillna(0, inplace=True)
    y_test_list[i].fillna(0, inplace=True)

    rf_predictions = rf_model.predict(X_test_list[i])
    tn, tp, fn, fp = confusion_matrix(y_true=y_test_list[i],y_pred=rf_predictions).ravel()
    rf_accuracy, rf_precision, rf_recall = calculate_measures(tn,tp,fn,fp)
    rf_accuracy_list.append(rf_accuracy)
    rf_precision_list.append(rf_precision)
    rf_recall_list.append(rf_recall)

    # ----- DECISION TREE ----- #
    dt_model.fit(X_train_list[i], y_train_list[i])
    dt_predictions = dt_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=dt_predictions).ravel()
    dt_accuracy, dt_precision, dt_recall = calculate_measures(tn, tp, fn, fp)
    dt_accuracy_list.append(dt_accuracy)
    dt_precision_list.append(dt_precision)
    dt_recall_list.append(dt_recall)

    # ----- SUPPORT VECTOR MACHINE ----- #
    svm_model.fit(X_train_list[i], y_train_list[i])
    svm_predictions = svm_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=svm_predictions).ravel()
    svm_accuracy, svm_precision, svm_recall = calculate_measures(tn, tp, fn, fp)
    svm_accuracy_list.append(svm_accuracy)
    svm_precision_list.append(svm_precision)
    svm_recall_list.append(svm_recall)

    # ----- ADABOOST ----- #
    ab_model.fit(X_train_list[i], y_train_list[i])
    ab_predictions = ab_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=ab_predictions).ravel()
    ab_accuracy, ab_precision, ab_recall = calculate_measures(tn, tp, fn, fp)
    ab_accuracy_list.append(ab_accuracy)
    ab_precision_list.append(ab_precision)
    ab_recall_list.append(ab_recall)

    # ----- GAUSSIAN NAIVE BAYES ----- #
    nb_model.fit(X_train_list[i], y_train_list[i])
    nb_predictions = nb_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=nb_predictions).ravel()
    nb_accuracy, nb_precision, nb_recall = calculate_measures(tn, tp, fn, fp)
    nb_accuracy_list.append(nb_accuracy)
    nb_precision_list.append(nb_precision)
    nb_recall_list.append(nb_recall)

    # ----- NEURAL NETWORK ----- #
    nn_model.fit(X_train_list[i], y_train_list[i])
    nn_predictions = nn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=nn_predictions).ravel()
    nn_accuracy, nn_precision, nn_recall = calculate_measures(tn, tp, fn, fp)
    nn_accuracy_list.append(nn_accuracy)
    nn_precision_list.append(nn_precision)
    nn_recall_list.append(nn_recall)
"""
    # ----- K-NEIGHBOURS CLASSIFIER ----- #
    # Convert columns with strings to numeric values


    kn_model.fit(X_train_list[i], y_train_list[i])

    X_test_list[i] = X_test_list[i].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with a default value
    X_test_list[i].fillna(0, inplace=True)
    
    kn_predictions = kn_model.predict(X_test_list[i])
    tn, fp, fn, tp = confusion_matrix(y_true=y_test_list[i], y_pred=kn_predictions).ravel()
    kn_accuracy, kn_precision, kn_recall = calculate_measures(tn, tp, fn, fp)
    kn_accuracy_list.append(kn_accuracy)
    kn_precision_list.append(kn_precision)
    kn_recall_list.append(kn_recall)
"""

RF_accuracy = sum(rf_accuracy_list) / len(rf_accuracy_list)
RF_precision = sum(rf_precision_list) / len(rf_precision_list)
RF_recall = sum(rf_recall_list) / len(rf_recall_list)

print("Random Forest Accuracy ==>", RF_accuracy)
print("Random Forest Precision ==>", RF_precision)
print("Random Forest Recall ==>", RF_recall)

DT_accuracy = sum(dt_accuracy_list) / len(dt_accuracy_list)
DT_precision = sum(dt_precision_list) / len(dt_precision_list)
DT_recall = sum(dt_recall_list) / len(dt_recall_list)

print("Decision Tree accuracy ==> ", DT_accuracy)
print("Decision Tree precision ==> ", DT_precision)
print("Decision Tree recall ==> ", DT_recall)


AB_accuracy = sum(ab_accuracy_list) / len(ab_accuracy_list)
AB_precision = sum(ab_precision_list) / len(ab_precision_list)
AB_recall = sum(ab_recall_list) / len(ab_recall_list)

print("AdaBoost accuracy ==> ", AB_accuracy)
print("AdaBoost precision ==> ", AB_precision)
print("AdaBoost recall ==> ", AB_recall)


SVM_accuracy = sum(svm_accuracy_list) / len(svm_accuracy_list)
SVM_precision = sum(svm_precision_list) / len(svm_precision_list)
SVM_recall = sum(svm_recall_list) / len(svm_recall_list)

print("Support Vector Machine accuracy ==> ", SVM_accuracy)
print("Support Vector Machine precision ==> ", SVM_precision)
print("Support Vector Machine recall ==> ", SVM_recall)


NB_accuracy = sum(nb_accuracy_list) / len(nb_accuracy_list)
NB_precision = sum(nb_precision_list) / len(nb_precision_list)
NB_recall = sum(nb_recall_list) / len(nb_recall_list)

print("Gaussian Naive Bayes accuracy ==> ", NB_accuracy)
print("Gaussian Naive Bayes precision ==> ", NB_precision)
print("Gaussian Naive Bayes recall ==> ", NB_recall)

NN_accuracy = sum(nn_accuracy_list) / len(nn_accuracy_list)
NN_precision = sum(nn_precision_list) / len(nn_precision_list)
NN_recall = sum(nn_recall_list) / len(nn_recall_list)

print("Neural Network accuracy ==> ", NN_accuracy)
print("Neural Network precision ==> ", NN_precision)
print("Neural Network recall ==> ", NN_recall)

"""
KN_accuracy = sum(kn_accuracy_list) / len(kn_accuracy_list)
KN_precision = sum(kn_precision_list) / len(kn_precision_list)
KN_recall = sum(kn_recall_list) / len(kn_recall_list)

print("K-Neighbours Classifier accuracy ==> ", KN_accuracy)
print("K-Neighbours Classifier precision ==> ", KN_precision)
print("K-Neighbours Classifier recall ==> ", KN_recall)

GP_accuracy = sum(gp_accuracy_list) / len(gp_accuracy_list)
GP_precision = sum(gp_precision_list) / len(gp_precision_list)
GP_recall = sum(gp_recall_list) / len(gp_recall_list)

print("Gaussian Process accuracy ==> ", GP_accuracy)
print("Gaussian Process precision ==> ", GP_precision)
print("Gaussian Process recall ==> ", GP_recall)
"""

data = {'accuracy': [NB_accuracy, SVM_accuracy, DT_accuracy, RF_accuracy, AB_accuracy, NN_accuracy],
        'precision': [NB_precision, SVM_precision, DT_precision, RF_precision, AB_precision, NN_precision],
        'recall': [NB_recall, SVM_recall, DT_recall, RF_recall, AB_recall, NN_recall]
        }

index = ['NB', 'SVM', 'DT', 'RF', 'AB', 'NN']

df_results = pd.DataFrame(data=data, index=index)


# visualize the dataframe
ax = df_results.plot.bar(rot=0)
plt.show()