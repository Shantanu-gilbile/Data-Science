# K-Nearest-Neighbors-Classification-Machine-Learning

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()

features=iris.data
labels=iris.target

classifier=KNeighborsClassifier()

classifier.fit(features,labels)

sl=int(input("Enter sepal length : "))
sw=int(input("Enter sepal width : "))
pl=int(input("Enter petal length : "))
pw=int(input("Enter petal width : "))

prediction=classifier.predict([[sl,sw,pl,pw]])

if(prediction==0):
    print("setosa")
    
elif(prediction==1):
    print("Versicolour")
    
else:
    print("Virginica")
    
    