import sys
from Naive_bays_classifier import calculateMean
from  hinge_loss_cp1 import wInitialization
from SNP.SVC import CallSVC
######################
#CREATING DATA MATRIX#
######################

def creatingDataset():
    # opening files
    file1 = sys.argv[1]
    with open(file1) as datafile:
        data = [line.split() for line in datafile]
    file2 = sys.argv[2]
    with open(file2) as datafile:
        data_labels = [line.split() for line in datafile]
    labels = {int(x[1]): int(x[0]) for x in data_labels}
    labels_list = [int(x[0]) for x in data_labels]
    features = [list(map(float,x)) for x in zip(*data)]
    print("creating dataset")
    return data, features, labels,labels_list

def calculateFscore(data, features, labels):
    positive = 0
    negative = 0
    num_feat = len(features)
    num = len(features[0])
    F_score =[]
    for i in range(num_feat):
        pos_list = []
        neg_list = []
        for j in range(len(labels)):
            if(labels.get(j)!=None):
                if(labels.get(j) == 1):
                    positive +=1
                    pos_list.append(features[i][j])
                else:
                    negative +=1
                    neg_list.append(features[i][j])
        mean = sum(features[i])/num
        mean_pos = sum(pos_list)/positive
        mean_neg = sum(neg_list)/negative
        vpos=0
        for x in pos_list:
            vpos +=(x - mean_pos) ** 2
        var_pos = vpos/positive
        vneg = 0
        for x in neg_list:
            vneg += (x - mean_neg) ** 2
        var_neg = vneg / negative
        fscore = ((mean_pos - mean)**2+(mean_neg - mean)**2)/(var_pos + var_neg)
        F_score.append(fscore)
    print("calculate score")
    return F_score

def selectFeature(features, F_score,labels,labelsdict,labels_list):
    num_feat = len(features)
    print("I am in Select Feature")
    seldata = [features[i] for i in range(num_feat) if F_score[i] > 27800]
    secdata = [list(x) for x in zip(*seldata)]
    testdata = []
    data = []
    testlabels = []
    print(len(secdata),len(secdata[0]))
    datafile = open("/home/kalyani/PycharmProjects/machine learning/data/train3.txt", 'w')
    for item in secdata:
        datafile.write("%s\n" % item)
    '''for i in range(len(secdata)):
        if (labelsdict.get(i) == None):
            testdata.append(secdata[i])
            testlabels.append(i)
        else:
            data.append(secdata[i])
    #callClasifier(labels,secdata,testdata,testlabels,labelsdict)



def callClasifier(labels,data,testdata,testlabels,labelsdict):
    for i in range(len(labelsdict)):
        if (labelsdict.get(i) == 0):
            labelsdict[i] = -1
    print("I am in callClasifier")
    wInitialization(data,labelsdict,testdata,testlabels)
'''
