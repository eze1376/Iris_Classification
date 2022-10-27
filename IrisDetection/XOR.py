
import csv
import random
import math

random.seed(123)

result_values = {}

def initialization(isTest, X1, X2):
# Load datase
    global test_y, test_X, train_X, train_y
    with open('/home/mohammadreza/Development/Python/Django/Iris_UI/IrisDetection/XOR.csv') as csvfile:
        csvreader = csv.reader(csvfile)

        dataset = list(csvreader)


    for row in dataset:
        row[2] = int(row[2])
        row[:2] = [int(row[j]) for j in range(len(row))]

    random.shuffle(dataset)
    datatrain = dataset[:int(len(dataset) * 0.75)]
    datatest = dataset[int(len(dataset) * 0.75):]
    train_X = [data[:2] for data in datatrain]
    train_y = [data[2] for data in datatrain]

    if isTest:
        test_X = [[int(X1), int(X2)]]
        test_y = [1]
    else:
        test_X = [data[:2] for data in datatest]
        test_y = [data[2] for data in datatest]


def matrix_mul_bias(A, B, bias): #for test
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))] #makin a zeros matrix with size (number of inputs)*(numnber of weights)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C


def vec_mat_bias(A, B, bias):  #value of output fo reach node of layer
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(A)):
            C[j] += A[k] * B[k][j]

            C[j] += bias[j]
    return C


def mat_vec(A, B):  #
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C


def sigmoid(A, deriv=False):
    if deriv:
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A

def predictor(Alpha, Epoch, Hidden_Neuron):


    result_values['cost_total'] = []

    alfa = Alpha
    # alfa = 0.002
    epoch = Epoch
    # epoch = 500
    neuron = [2, 2, 1]  # number of neuron each layer
    # neuron = [4, 5, 3]  # number of neuron each layer


    weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
    weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
    bias = [0 for i in range(neuron[1])]
    bias_2 = [0 for i in range(neuron[2])]

    for i in range(neuron[0]):
        for j in range(neuron[1]):
            weight[i][j] = 2 * random.random() - 1

    for i in range(neuron[1]):
        for j in range(neuron[2]):
            weight_2[i][j] = 2 * random.random() - 1
    for e in range(epoch):
        cost_total = 0
        for idx, x in enumerate(train_X):

            # Forward propagation
            h_1 = vec_mat_bias(x, weight, bias)
            X_1 = sigmoid(h_1)
            h_2 = vec_mat_bias(X_1, weight_2, bias_2)
            X_2 = sigmoid(h_2)

            target = [0, 0, 0]
            target[int(train_y[idx])] = 1

            eror = 0

            eror += 0.5 * (target[0] - X_2[0]) ** 2
            cost_total += eror

            # Backward propagation
            delta_2 = []
            for j in range(neuron[2]):
                delta_2.append(-1 * (target[j] - X_2[j]) * X_2[j] * (1 - X_2[j]))

            for i in range(neuron[1]):
                for j in range(neuron[2]):
                    weight_2[i][j] -= alfa * (delta_2[j] * X_1[i])
                    bias_2[j] -= alfa * delta_2[j]

            # Update weight and bias
            delta_1 = mat_vec(weight_2, delta_2)
            for j in range(neuron[1]):
                delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

            for i in range(neuron[0]):
                for j in range(neuron[1]):
                    weight[i][j] -= alfa * (delta_1[j] * x[i])
                    bias[j] -= alfa * delta_1[j]

        cost_total /= len(train_X)
        if (e % 100 == 0):
            # print(cost_total)
            result_values['cost_total'].append(cost_total)
    print(weight,'\n')
    print(weight_2,'\n')



    res = matrix_mul_bias(test_X, weight, bias)
    res_2 = matrix_mul_bias(res, weight_2, bias)


    preds = []
    for r in res_2:
        preds.append(max(enumerate(r), key=lambda x: x[1])[0])

    # Print prediction
    # print(preds)
    result_values['predictions'] = preds
    # Calculate accuration
    acc = 0.0
    for i in range(len(preds)):
        if preds[i] == int(test_y[i]):
            acc += 1
    # print(acc / len(preds) * 100, "%")
    result_values['accuracy'] = str(acc / len(preds) * 100) + '%'

    return result_values

def main(isTest, Alpha = .1, Epoch = 1000, Hidden_Neuron = 5, X1 = 0, X2 = 0):
    result_values.clear()
    initialization(isTest, X1, X2)
    return predictor(Alpha, Epoch,2)
# print(main(True,SepalLengthCm = 5.8, SepalWidthCm = 2.7, PetalLengthCm = 5.1, PetalWidthCm= 1.9))
# print(main(False, Epoch= 500, Alpha= .002, Hidden_Neuron= 5))
# print(main(False, Epoch= 500, Alpha= .002, Hidden_Neuron= 5))
print(main(True, X1 = 1, X2= 1))