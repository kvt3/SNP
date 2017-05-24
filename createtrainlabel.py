import random


def createtrainlabel(data, labels):
    temp = [x for x in data]
    templ = [x for x in labels]
    print("temp:",len(temp))
    tl = len(data)
    print(int(tl*0.1))
    #rand = [random.randrange(0, (tl-1)) for x in range(0,(tl),1)]
    #print(rand)
    #print(x for x in range(1,10,1))
    print("starting...")
    predict = []
    rangef=int(tl*0.1)
    predictlabels =[]
    if rangef<1:
        rangef = 1
    for x in range(0, rangef ,1):
        randomnum = random.randrange(0, (len(data)-1))
        predictlabels.append(randomnum)
        predict.append(data[randomnum])
        del data[randomnum]
        del labels[randomnum]


    #####################################################
    ############ print the 90% random labels ############
    #####################################################
    #for x in range(0,len(data),1):
    #    print(data[x], labels[x][0])
    print(predictlabels,len(predict))
    #print(str(int(x[0]))+" "+str(int(x[1])))

    return temp, templ, data, predict, labels, predictlabels