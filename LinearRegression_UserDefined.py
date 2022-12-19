import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def HeadBrainPredictor():
    data = pd.read_csv(r'C:\Users\Vijay\Desktop\ML\Ball\Iris\MarvellousHeadBrain.csv')

    print("Size of Dataset : ",data.shape)

    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values

    #Least Square Method

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)

    numerator = 0
    denomerator = 0

    #Equation of line is y = mx + c

    for i in range(n):
        numerator += (X[i]-mean_x)*(Y[i]-mean_y)
        denomerator += (X[i]-mean_x)**2

    m = numerator / denomerator

    c = mean_y - (m * mean_x)

    print("Slope of Regression line is : ",m)
    print("Y intercept of Regression line is",c)

    max_x = np.max(X) + 100
    min_x = np.min(X)  - 100

    x = np.linspace(min_x,max_x,n)

    y = m * x + c

    plt.plot(x,y,color = '#58b970',label='Regression Line')
    plt.scatter(X,Y,color='#ef5423',label = 'scatter plot')

    plt.xlabel('Head size in cm3')
    plt.ylabel("Brain weight in gram")

    plt.legend()
    plt.show()

    ss_t = 0
    ss_r = 0

    for i in range(n):
        y_pred = c + m*X[i]
        ss_t += (Y[i] - mean_y)**2
        ss_r += (Y[i] - y_pred)**2
    
    r2 = 1 - (ss_r/ss_t)

    print("Valur of r2 is : ",r2)

def main():
    print("Supervised Machine Learning")
    print("Linear Regression on Head and Breain size dataset")
    HeadBrainPredictor()


if __name__ =="__main__":
    main()
    
