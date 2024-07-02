# import operator
# from collections import Counter

# import numpy as np
# class Knn:

#     def __init__(self,k=5):
#         self.n_neighbors = k
#         self.X_train=None
#         self.y_train = None

#     def fit(self,X_train,y_train):
#         self.X_train = X_train
#         self.y_train = y_train.to_numpy()
#     def predict(self,X_test):
#         y_pred = []
#         for i in X_test:
#             # calculate distance with each training point
#             distances = {}
#             counter =  1
#             for j in self.X_train:
#                 distances[counter]= self.calculate_distance(i,j)
#                 counter=counter+1
#             distances= sorted(distances.items(),key=operator.itemgetter(1))
#             valid_neighbors = []
#             for i in distances:
#                 idx = i[0]
#                 if idx in self.y_train:
#                     valid_neighbors.append(idx)
#             neighbors = valid_neighbors[0:self.n_neighbors]

#             # calculating majority count
#             label = self.majority_count(neighbors)
#             y_pred.append(label)
#         return np.array(y_pred) 

#     def calculate_distance(self,point_A,point_B):
#         return np.linalg.norm(point_A-point_B)
#     # def majority_count(self,neighbors):
#     def majority_count(self, neighbors):
#         votes = []
#         for i in neighbors:
#             votes.append(self.y_train[i])

#         # print(votes)
#         # # now calculating most common element in python
#         votes = Counter(votes)
#         return votes.most_common()[0][0]


import numpy as np
from collections import Counter

class Knn:
    def __init__(self, k=5):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train.to_numpy()  # Convert y_train to numpy array

    def predict(self, X_test):
        y_pred = []
        for i in X_test:
            distances = []
            for j in self.X_train:
                distances.append(self.calculate_distance(i, j))
            neighbors = sorted(list(enumerate(distances)), key=lambda x: x[1])
            top_neighbors = [self.y_train[idx] for idx, _ in neighbors[:self.n_neighbors]]
            label = self.majority_count(top_neighbors)
            y_pred.append(label)
        return np.array(y_pred)

    def calculate_distance(self, point_A, point_B):
        return np.linalg.norm(point_A - point_B)

    def majority_count(self, neighbors):
        votes = Counter(neighbors)
        return votes.most_common(1)[0][0]



      