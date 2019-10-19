# -*- coding: utf-8 -*-
"""
https://blog.amedama.jp/entry/2017/03/18/140238
Created on Thu Dec  6 15:55:40 2018

@author: Akitaka
"""

from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris_dataset = datasets.load_iris()

    features = iris_dataset.data
    targets = iris_dataset.target

    predicted_labels = []

    loo = LeaveOneOut()
    for train, test in loo.split(features):
        train_data = features[train]
        target_data = targets[train]

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(train_data, target_data)

        predicted_label = model.predict(features[test])
        predicted_labels.append(predicted_label)

    score = accuracy_score(targets, predicted_labels)
    print(score)


if __name__ == '__main__':
    main()