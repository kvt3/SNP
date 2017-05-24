import sys
import traceback
import importlib
import PCA

if __name__=='__main__':

    data = None
    labels = None
    features = None
    testdata = None
    testlabels = None
    is_first_iteration = 1

    while True:
        if not data:
            data, labels, testdata = PCA.creatingDataset()
            #data,features,labels,labels_list = Fscore.creatingDataset()
            #F_score = Fscore.calculateFscore(data, features, labels)
            #data, labels, testdata, testlabels = SVC.creatingDataset()
        try:
            if is_first_iteration == 0:
                importlib.reload(PCA)

            is_first_iteration = 0
            #Fscore.selectFeature(features,F_score,labels_list,labels,labels_list)
            #SVC.CallSVC(data,labels,testdata,testlabels)
            PCA.calculateF(data, labels, testdata)
        except Exception as e:
            print('*' * 64)
            print('Exception raised in tested module')
            print(traceback.print_exc())
            print('*' * 64)

        print("Press enter to re-run script or CTRL-C to exit")
        sys.stdin.readline()
