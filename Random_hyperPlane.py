import csv
import random
import sys
from sklearn.svm import LinearSVC
import hinge_loss_cp1

def read_file(filename):
    with open(filename) as datafile:
        data = [line.split() for line in datafile]
        return data

def create_random_hyperplane(cols):
    w = []
    for j in range(cols-1):
        # w.append(0.00002 * random.uniform(0,1) - 0.00001)
        w.append(0.02 * random.random() - 0.01)
    w.append(0)
    return w

def sign(x): return 1 if x >= 0 else -1


def create_projection(data,w):
    rows = len(data)
    cols = len(data[0])
    z = []
    for i in range(rows):
        dotproduct = 0
        for j in range(cols):
            dotproduct += w[j] * float(data[i][j])
        z.append(sign(dotproduct))
    return z

def run_svc_classifier(data,labels,testdata,testlabels):
    clf = LinearSVC(max_iter=150000, tol=0.000001).fit(data, labels)
    predictedlabels = clf.predict((testdata))
    for i in range(0, len(testdata), 1):
        print(predictedlabels[i] , testlabels[i])


def main():
    data = read_file(sys.argv[1]);
    data_labels = read_file(sys.argv[2]);
    labels = {int(x[1]): int(x[0]) for x in data_labels}

    newmatrix = []
    for i in range(2000):
        w = create_random_hyperplane(len(data[0]))
        z = create_projection(data,w)
        newmatrix.append(z)
    zmatrix = [list(x) for x in zip(*newmatrix)]
    testdata =[]
    testlabels = []
    traindata =[]
    trainlabes = []
    for i in range(len(z)):
        if (labels.get(i) == None):
            testdata.append(zmatrix[i])
            testlabels.append(i)
        else:
            traindata.append(zmatrix[i])
            trainlabes.append(labels.get(i))
    #hinge_loss_cp1.wInitialization(traindata,trainlabes,testdata,testlabels)
    run_svc_classifier(traindata,trainlabes,testdata,testlabels)



main()