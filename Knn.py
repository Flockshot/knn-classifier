import numpy as np

class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):

        distances = []

        # calculate the distance from the instance to each data sample in the dataset
        for i in range(len(self.dataset)):
            if self.similarity_function_parameters is None:
                distances.append([self.dataset_label[i], self.similarity_function(self.dataset[i], instance)])
            else:
                distances.append([self.dataset_label[i], self.similarity_function(self.dataset[i], instance, self.similarity_function_parameters)])


        # sort the distances
        distances.sort(key=lambda x: x[1])
        k_nearest = distances[:self.K]
        # if self.K == 5:
        #    print(k_nearest)
        # print(k_nearest)
        # print(k_nearest)
        # count the labels of the k nearest neighbors
        label_count = {}
        for data in k_nearest:
            if data[0] in label_count:
                label_count[data[0]] += 1
            else:
                label_count[data[0]] = 1

        # print(label_count)
        # print(sorted(label_count.items(), key=lambda x: x[1]))
        # return the label with the highest count
        return sorted(label_count.items(), key=lambda x: x[1], reverse=True)[0][0]




