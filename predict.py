import numpy as np
import csv
import sys

from validate import validate

train_X_file_path = "train_X_knn.csv"
train_Y_file_path = "train_Y_knn.csv"
validation_split_percent = 20

def compute_ln_norm_distance(vector1, vector2, n):
    sum = 0
    for i in range(min(len(vector1), len(vector2))):
        sum += abs(vector1[i] - vector2[i])**n
    return (sum**(1/n))

def find_k_nearest_neighbors(train_X, test_example, k, n):
    kNN = []
    kNNdists = []
    for i in range(len(train_X)):
        dist = compute_ln_norm_distance(test_example, train_X[i], n)
        if len(kNN) < k:
            kNN.append(i)
            kNNdists.append(dist)
        elif max(kNNdists) > dist:
            toDelDist = kNNdists.index(max(kNNdists))
            toDel = kNN[toDelDist]
            kNN.remove(toDel)
            kNNdists.remove(kNNdists[toDelDist])
            kNN.append(i)
            kNNdists.append(dist)
    z = list(zip(kNNdists, kNN))
    z.sort()
    z = [i for (d,i) in z]
    return z 

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_example in test_X:
        kNN = find_k_nearest_neighbors(train_X, test_example, k, n)
        Ys = [train_Y[knn] for knn in kNN]
        test_Y.append(max(set(Ys), key=Ys.count))
    return test_Y

def calculate_accuracy(predicted_Y, actual_Y):
    correct = 0
    for i in range(min(len(predicted_Y), len(actual_Y))):
        correct += (predicted_Y[i] == actual_Y[i])
    return correct/min(len(predicted_Y), len(actual_Y))

def get_best_k_n_using_validation_set(train_X, train_Y, validation_split_percent):
    import math 
    train_inst_num = math.floor(((100-validation_split_percent)/100)*len(train_X))
    test_X = train_X[train_inst_num:]
    test_Y = train_Y[train_inst_num:]
    train_X = train_X[:train_inst_num]
    train_Y = train_Y[:train_inst_num]
    bestAcc = 0 
    bestK = 0 
    bestN = 0
    for k in range(1,train_inst_num+1):
        for n in range(1,5):
            pred_Y = classify_points_using_knn(train_X, train_Y, test_X, n, k)
            acc = calculate_accuracy(pred_Y, test_Y)
            if acc >= bestAcc:
                bestAcc = acc
                bestK = k
                bestN = n
    return (bestK, bestN)

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def predict_target_values(test_X):
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.
    train_X = import_data(train_X_file_path)[1:]
    train_Y = import_data(train_Y_file_path)
    k, n = get_best_k_n_using_validation_set(train_X, train_Y, validation_split_percent)
    return np.array(classify_points_using_knn(train_X, train_Y, test_X, k, n), dtype=int)
    
    
def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 