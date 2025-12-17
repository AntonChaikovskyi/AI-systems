import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score 

input_file = r'D:\politeh\ai\lab2\income_data.txt' 

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue
            
            data = line[:-1].split(', ')
            
            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1
except FileNotFoundError:
    print(f"Помилка: Файл {input_file} не знайдено.")
    exit()

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)

for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000)) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
classifier.fit(X_train, y_train)

y_test_pred = classifier.predict(X_test)

f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

from sklearn.metrics import accuracy_score, precision_score, recall_score
print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 2))
print("Precision:", round(precision_score(y_test, y_test_pred, average='weighted'), 2))
print("Recall:", round(recall_score(y_test, y_test_pred, average='weighted'), 2))

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        try:
            input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        except:
             input_data_encoded[i] = 0 
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1) 

predicted_class = classifier.predict(input_data_encoded)
print(f"Predicted class (encoded): {predicted_class[0]}")