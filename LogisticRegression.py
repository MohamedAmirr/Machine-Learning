import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv("magic04.data", names=cols)

# Printing dataset

print(df.head())
df['class'] = (df['class'] == "g").astype(int)

# Plotting dataset
for label in cols[:-1]:
    plt.hist(df[df["class"] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    # plt.show()

# Train,  Validation, test dataset


train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])


def scaleDataset(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # taking more of the less class to increase the size of out dataset of that smaller class
    # so they not match
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y


# if we print len of dataset of class 0 and 1 without resampling
# "print(len(train[train["class"]==1])), print(len(train[train["class"]==0])),
# we will see that the number of class 1 is very different from class 0
# so we make over sample in function scalDataset

train, xTrain, yTrain = scaleDataset(train, oversample=True)
valid, xvalid, yvalid = scaleDataset(valid, oversample=False)
test, xTest, yTest = scaleDataset(test, oversample=False)

# after OverSampling to see result of reSample{
# print(len(yTrain))
# print(sum(yTrain == 1))
# print(sum(yTrain == 0))
# }
# Before reSample
# print(len(train[train["class"] == 1]))
# print(len(train[train["class"] == 0]))

# Logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

lgModel = LogisticRegression()
lgModel = lgModel.fit(xTrain, yTrain)

yPredi = lgModel.predict(xTest)

print(classification_report(yTest, yPredi))
