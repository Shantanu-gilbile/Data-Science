from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def ShantanuKNeighborsClassfier():

        Dataset = load_iris()   # 1 Load the data
        Data = Dataset.data
        Target = Dataset.target

        # 2 : Manipulate the data
        Data_Train , Data_Test , Target_Train , Target_Test = train_test_split(Data,Target,test_size=0.5)

        Classifier = KNeighborsClassifier()

        # 3 : Built the model
        Classifier.fit(Data_Train,Target_Train)

        # 4 : Test the model
        Predictions = Classifier.predict(Data_Test)

        Accuracy = accuracy_score(Target_Test , Predictions)

        #5 : Improve --Missing

        return Accuracy


def main():
    Ret = ShantanuKNeighborsClassfier()

    print("Accuracy of Iris Dataset with KNN is : ",Ret*100)

if __name__ =="__main__":
    main()