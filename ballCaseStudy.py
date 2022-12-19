# Import required library
from sklearn import tree

# Rough 1
# Smooth 0

# Tennis 1
# Cricket 2

def ballPredictor(w,s):
    # Load the dataset
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]

    labels =[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    # Decide the ML Algorithm
    obj = tree.DecisionTreeClassifier()

    #Perform the training
    obj = obj.fit(Features,labels)

    #Perform the testing
    ans = obj.predict([[w,s]])

    for a in ans:
        if a==1:
            print("Your Object Looks like Tennis Ball")
        else:
            print("Your Object Looks like Cricket Ball")

def main():
    weight=int(input("Please enter the weight of your object"))
    surface = input("Enter the Type of your object(Rough / Smooth")

    if surface.lower()=="rough":
        surface = 1

    elif surface.lower()=="smooth":
        surface = 0

    ballPredictor(weight,surface)
if __name__ =="__main__":
    main()

