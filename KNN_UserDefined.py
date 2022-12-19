from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ShantanuKNeighborsClassfier:

    def fit(self,TrainingData, TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget

    def closest(self,row):
        minimumDistance = euc(row,self.TrainingData[0])
        minimumIndex  = 0

        for i in range(1,len(self.TrainingData)):
            Distance = euc(row , self.TrainingData[i])
            if Distance < minimumDistance:
                minimumDistance = Distance
                minimumIndex = i

        return self.TrainingTarget[minimumIndex]

    def predict(self,TestData):
        predictions = []
        for value in TestData:
            result = self.closest(value)
            predictions.append(result)
        return predictions



def ShantanuML():

        Dataset = load_iris()   # 1 Load the data
        Data = Dataset.data
        Target = Dataset.target

        # 2 : Manipulate the data
        Data_Train , Data_Test , Target_Train , Target_Test = train_test_split(Data,Target,test_size=0.5)

        Classifier = ShantanuKNeighborsClassfier()

        # 3 : Built the model
        Classifier.fit(Data_Train,Target_Train)

        # 4 : Test the model
        Predictions = Classifier.predict(Data_Test)

        Accuracy = accuracy_score(Target_Test , Predictions)

        #5 : Improve --Missing

        return Accuracy


def main():
    Ret = ShantanuML()

    print("Accuracy of Iris Dataset with KNN is : ",Ret*100)

if __name__ =="__main__":
    main()