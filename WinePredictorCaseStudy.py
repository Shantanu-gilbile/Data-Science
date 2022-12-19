import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def WinePridector():
    data = pd.read_excel(r'C:\Users\Vijay\Desktop\ML\Ball\Iris\WinePredictor.xlsx',skiprows=1)

    features = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]].values
    labels = data.iloc[:,[0]].values

    print(features)
    print(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    Classifier = DecisionTreeClassifier()

    Classifier.fit(x_train, y_train)

    Predictions = Classifier.predict(x_test)

    Accuracy = accuracy_score(y_test, Predictions)


    print("Accuracy is  : ",Accuracy * 100," %")

def main():
    print("Supervised Machine Learning")
    print("Wine Predictor using Decision Tree Claasifier")
    WinePridector()

if __name__ == "__main__":
    main()
