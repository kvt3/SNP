################################################################
################################################################
###                                                          ###
###     PROJECT AUTHORS: VISHAL KULKARNI & KALYANI TARU      ###
###                                                          ###
################################################################
################################################################
import random
import time
import sys
from sklearn.svm import LinearSVC

#######################################
####### reading data from args ########
#######################################

def readfile(x):
    data = []
    c =0
    starttimeinmillis = int(round(time.time()))
    print("Reading data from file[",x,"]",sep='',end='',flush=True)
    try:
        file = sys.argv[x]
    except IndexError:
        print()
        print ("#"*104)
        print("###### Improper Syntax: try - python3 SNP.py traindata trainlabels testdata {i.e. 3 dataset files} #####")
        print ("#"*104)
        sys.exit()

    with open(file) as datafile:
        for line in datafile:
            c+=1
            if(c%100==0):
                print(".",sep='', end='',flush=True)
            data.append([int(l) for l in line.split()])

    #rows = len(data)
    #cols = len(data[0])
    # print(data)

    #print("rows=", rows, " cols=", cols)
    print("time took:",int(round(time.time()))-starttimeinmillis,"seconds")
    return data

###########################################################
####### creating train dataset for cross validation #######
###########################################################

def createTrainlabel(data, labels):
    org_data = [x for x in data]
    org_labels = [x for x in labels]
    print("Original data:",len(org_data))
    rows = len(data)
    print(int(rows*0.1))

    print("starting...")
    predictdata = []
    rangef=int(rows*0.1)
    predictlabels =[]
    if rangef<1:
        rangef = 1
    for x in range(0, rangef ,1):
        randomnum = random.randrange(0, (len(data)-1))
        predictlabels.append(randomnum)
        predictdata.append(data[randomnum])
        del data[randomnum]
        del labels[randomnum]
    return org_data, org_labels, data, predictdata, labels, predictlabels

##########################################
############ Feature selection ###########
##########################################

def calFetures(data, labels):

    print("Now selecting the bestfeatures....................",sep='', end='',flush=True)
    linearSVC = LinearSVC(C=0.002, penalty='l1', dual=False).fit(data,[x[0] for x in labels])
    score = linearSVC.coef_

    feature = []
    feature_cols = []
    for k in range(len(score[0])):
        if(score[0][k] != 0.0):
            feature_cols.append(int(k))
            rowdata = []
            for i in range(len(data)):
                rowdata.append(data[i][k])
            feature.append(rowdata)
    bestfeatures_data = [list(map(float, x)) for x in zip(*feature)]
    #return bestfeatures_data
    #print(feature_cols)
    #---------------------------------------------------
    #------ writing selected feature data to file ------
    #---------------------------------------------------


    print("Number of Features selected in traindata:",len(bestfeatures_data[0]))


    return bestfeatures_data, feature_cols

###################################################
##### selecting the best feature of testdata ######
###################################################

def getFetures(testdata,feature_cols):
    print("Now getting the bestfeatures for testdata.........",sep='', end='',flush=True)

    bestfeature_testdata = []


    for rows in range(len(testdata)):
        col = []
        k=0
        for cols in range(len(testdata[0])):
            if(k<len(feature_cols) and feature_cols[k] == cols):
                col.append(testdata[rows][cols])
                k+=1
        bestfeature_testdata.append(col)
    #print(bestfeature_testdata)

    print("Number of Features selected in testdata:",len(bestfeature_testdata[0]))

    return bestfeature_testdata


#######################################
########### classify data #############
#######################################

def predictLabels(traindata, trainlabels, testdata):
    print("Classifying data",sep='',end='',flush=True)

    #--------------------------------------
    #-------- classification method -------
    #--------------------------------------

    clf = LinearSVC(max_iter=15000000, verbose=1,tol=0.00000001).fit(traindata,[x[0] for x in trainlabels])
    #clf = SVC(C=10, kernel='linear', verbose=1, tol=0.00000001).fit(data,[x[0] for x in labels])
    #clf = tree.DecisionTreeClassifier().fit(X,[x[0] for x in y])

    #--------------------------------------
    #------- data prediction methods ------
    #--------------------------------------

    predictedlabels=clf.predict((testdata))
    #predictedl=clf.predict(scale(predict))     #For LinearSVC

    #---------------------------------------
    #----- writing and printing data -------
    #---------------------------------------
    print('/n')

    predictedlabels_file = open('/home/kalyani/PycharmProjects/machine learning/predictedlabels', 'w')

    for i in range(0, len(testdata), 1):
        print(str(predictedlabels[i])+" "+str(i))
        predictedlabels_file.write(str(predictedlabels[i])+" "+str(i)+'\n')

    predictedlabels_file.close()
    return True

def main():
    startTime = time.time()

    #traindata = threading.Thread(target=readfile(1)).start()
    #trainlabels = threading.Thread(target=readfile(2)).start()
    #testdata = threading.Thread(target=readfile(3)).start()

    traindata = readfile(1)
    trainlabels = readfile(2)
    testdata = readfile(3)

    bestfeatures_data, feature_cols = calFetures(traindata,trainlabels)
    bestfeature_testdata = getFetures(testdata,feature_cols)
    isPredicted = predictLabels(bestfeatures_data,trainlabels,bestfeature_testdata)

    if(isPredicted):
        print ("#"*80)
        print("####### Data classified successfully and stored in 'predictedlabels' file ######")
        print ("#"*80)

    totalTime = (time.time()-startTime)

    print("Total Execution Time: ",int(totalTime/60), "minutes",int(totalTime%60),"seconds")

main()