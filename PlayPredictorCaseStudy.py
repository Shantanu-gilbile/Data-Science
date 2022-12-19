import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = pandas.read_excel(r'C:\Users\Vijay\Desktop\MarvellousAssignmant\PlayPredictor.xlsx')

data = dataset.iloc[:,[1,2]].values
target = dataset.iloc[:,[3]].values

print(data)
print(target)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)

Classifier = KNeighborsClassifier()

Classifier.fit(x_train, y_train)

Predictions = Classifier.predict(x_test)

Accuracy = accuracy_score(y_test, Predictions)

print("Accuracy is  : ",Accuracy * 100," %")
