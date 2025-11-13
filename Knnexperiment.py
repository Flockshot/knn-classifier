import pickle
import time
import numpy as np
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import StratifiedKFold



def calc_lower_bound(mean, std):
    return mean - ((1.96 * std) / np.sqrt(10))


def calc_upper_bound(mean, std):
    return mean + ((1.96 * std) / np.sqrt(10))


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))

k_values = [5, 10, 30]
distance_functions = [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance,
                      Distance.calculateMahalanobisDistance]
iterations = 5
k_folds = 10

stats = []

start_time = time.time()
for k in k_values:
    for distance_function in distance_functions:
        one_start_time = time.time()
        model_accuracies = []
        for i in range(iterations):
            k_values = StratifiedKFold(n_splits=k_folds, shuffle=True)

            fold_accuracies = []
            for train_index, test_index in k_values.split(dataset, labels):
                train_dataset, test_dataset = dataset[train_index], dataset[test_index]
                train_labels, test_labels = labels[train_index], labels[test_index]
                S_1 = None

                if distance_function == Distance.calculateMahalanobisDistance:
                    S_1 = np.linalg.inv(np.cov(train_dataset.T))

                knn = KNN(train_dataset, train_labels, distance_function, S_1, k)

                correct = 0
                for j in range(len(test_dataset)):
                    prediction = knn.predict(test_dataset[j])
                    # print("Prediction: %d - Actual: %d" % (prediction, test_labels[j]))
                    if prediction == test_labels[j]:
                        correct += 1
                fold_accuracies.append((correct / len(test_dataset))*100)
            model_accuracies.append(np.mean(fold_accuracies))

        mean = np.mean(model_accuracies)
        std = np.std(model_accuracies)
        lower_bound = calc_lower_bound(mean, std)
        upper_bound = calc_upper_bound(mean, std)

        stat = [k, distance_function.__name__, mean, std, lower_bound, upper_bound]
        stats.append(stat)

        print("Statistics for K: %d - Distance Function: %s" % (k, distance_function.__name__))
        print("Mean: %.2f" % mean)
        print("Std: %.2f" % std)
        print("Lower Bound: %.2f" % lower_bound)
        print("Upper Bound: %.2f" % upper_bound)
        print("Time Taken: %.2f seconds" % (time.time() - one_start_time))
        print("------------------------------------------------------------")


stats.sort(key=lambda x: x[2], reverse=True)

print("\n\nBest Model: K: %d - Distance Function: %s" % (stats[0][0], stats[0][1]))
print("Mean: %.2f" % stats[0][2])
print("Std: %.2f" % stats[0][3])
print("Lower Bound: %.2f" % stats[0][4])
print("Upper Bound: %.2f" % stats[0][5])


elapsed_time = time.time() - start_time
print("\n\nTotal Time Taken: %.2f seconds" % elapsed_time)
