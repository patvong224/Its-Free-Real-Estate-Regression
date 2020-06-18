import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.linear_model import Ridge
import matplotlib.pyplot as pyplot
from matplotlib import style

data = pd.read_csv("RealEstateData.csv")
data = data[["No","X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores",
             "X5 latitude", "X6 longitude", "Y house price of unit area"]]

pred = "Y house price of unit area"

x = np.array(data.drop([pred],1))
y = np.array(data[pred])

print(data.head())

norm_X = preprocessing.normalize(x)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(norm_X, y, test_size=0.10)


best = 0

for _ in range (30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(norm_X, y, test_size=0.10)

    lin = Ridge()

    lin = linear_model.LinearRegression()

    lin.fit(x_train, y_train)

    acc = lin.score(x_train, y_train)

    #print("Accuracy is ", str(acc * 100), "%")

    if acc>best:
        best = acc
        print("The best score is out of 30 is ", str(best*100),"%")
        with open("realestate.pickle", "wb") as f:
            pickle.dump(lin, f)



pickle_in = open("realestate,pickle", "rb")
lin = pickle.load(pickle_in)

print("Coefficent: ",lin.coef_)
print("\nIntercept: ",lin.intercept_)

for x in range(len(pred)):
    print(pred[x], x_test[x], y_test[x])

p ="No"

style.use("ggplot")
pyplot.scatter(data[p], data[pred])
pyplot.title("Price of housing ")
pyplot.xlabel("Number")
pyplot.ylabel("Price per unit of area")
pyplot.show()













