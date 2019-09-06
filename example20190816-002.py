import csv
import numpy as np


def read_data():
    x = []
    y = []
    with open('csv/京Q2H1S5_2019_07_15_00_45_18_mvv_Loc1.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            x_line = []
            for s in line[2:8]:
                x_line.append(float(s))
            x.append(x_line)
            y.append(float(line[len(line)-2]))
    return x, y


X0, y0 = read_data()
# print(X0)
# print(y0)
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0)-2
X = np.array(X0[:d])
y = np.transpose(np.array([y0[:d]]))
# print(X)
# print(y)

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt, X)
Xty = np.dot(Xt, y)
beta = np.linalg.solve(XtX, Xty)
print(beta)

# Make predictions for the last 10 rows in the data set
for data, actual in zip(X0[d:], y0[d:]):
    x_data = np.array([data])
    prediction = np.dot(x_data, beta)
    print('prediction = '+str(prediction[0, 0])+' actual = '+str(actual))
