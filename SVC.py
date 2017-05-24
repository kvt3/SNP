import sys

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm



def creatingDataset():
    '''''data= [[-1, -1], [-2, -1], [1, 1], [2, 1]]
    labels = [1, 1, 2, 2]
    testdata = [[2,2]]
    testlabels = [1]
    '''
    file1 = sys.argv[1]
    rawdata =[]
    with open(file1) as datafile:
        for line in datafile:
            line = line.replace("[", '')
            line = line.replace("]", '')
            line = line.replace(",", '')
            line =line.split()
            data_value =[]
            for i in range(len(line)):
                data_value.append(float(line[i]))
            rawdata.append(data_value)
    #print(rawdata)

    file2 = sys.argv[2]
    with open(file2) as datafile:
        data_labels = [line.split() for line in datafile]
    rawlabels = {int(x[1]): int(x[0]) for x in data_labels}
    labels = [int(x[0]) for x in data_labels]

    testdata =[]
    data = []
    testlabels =[]
    for i in range (len(rawdata)):
        if(rawlabels.get(i) == None):
            testdata.append(rawdata[i])
            testlabels.append(i)
        else:
            data.append(rawdata[i])

    '''file3 = sys.argv[3]
    testdata = []
    with open(file3) as testfile:
        for line in testfile:
            line = line.replace("[", '')
            line = line.replace("]", '')
            line = line.replace(",", '')
            line = line.split()
            data_value = []
            for i in range(len(line)):
                data_value.append(float(line[i]))
            testdata.append(data_value)
    '''
    CallSVC(data,labels,testdata,testlabels)

def CallSVC(data,labels,testdata,testlabels):
    #print("I am here")
    #svc = svm.SVC(kernel='linear',C=1,verbose=0,tol=0.000001).fit(data, labels)
    svc = svm.LinearSVC(tol=0.000001,max_iter=1000000).fit(data, labels)

    for i in range(len(testdata)):
        print(svc.predict([testdata[i]]), testlabels[i])
        #print(svc.predict([testdata[i]]),i)


creatingDataset()