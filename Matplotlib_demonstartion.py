import pandas as pd
import matplotlib.pyplot as plt

def Matplotlib_Plotting():
    data = pd.read_excel(r'C:\Users\Vijay\Desktop\ML\Ball\Iris\Marvellous.xlsx')

    print("All data from excel")
    print(data)

    print("First 4 rows from File")
    print(data.head(4))

    print("Last 4 rows from file")
    print(data.tail(4))

    print("Shape of File",data.shape)

    data['Age'].plot(kind="hist")
    plt.show()

    data['Age'].plot(kind="barh")
    plt.show()


def main():
    print("Demonstration of MatplotLib for Plotting")
    Matplotlib_Plotting()
if __name__ == "__main__":
    main()