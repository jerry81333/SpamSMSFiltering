from numpy import *
import random
from sklearn import svm
import time

def creatVocabList(Data):
    List = set([])  #create empty set
    for document in Data:
        List = List | set(document) #union of the two sets
    return list(List)

def textList(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def dataToList(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainSVM(trainData,trainLabel):
	classifier = svm.SVC(C=10.0,kernel='linear') #Penalty parameter C is too high, may overfit
	return classifier.fit(trainData,trainLabel)

def mailfilter(K):
    txtList=[]; classList = []; fullText =[]
    for i in range(1,K+1):
        wordList = textList(open('spam/%d.txt' % i).read())
        #print (wordList)
        txtList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textList(open('ham/%d.txt' % i).read())
        txtList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = creatVocabList(txtList) #create vocabulary
    trainingSet = list(range(2*K))

    #create test set
    testSet=[] 
    for i in range(int(K/2)):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) 
    trainMat=[]; trainClasses = []

    start = time.clock()
    for i in trainingSet:#train the classifier (get probs) naiveBayesTrain
        trainMat.append(dataToList(vocabList, txtList[i]))
        trainClasses.append(classList[i])
    clf = trainSVM(array(trainMat),array(trainClasses))

    right = 0
    errorCount = 0
    shouldSpam = 0
    shouldHam = 0
    for i in testSet:        #classify the remaining items
        wordVector = dataToList(vocabList, txtList[i])
        if(clf.predict(array(wordVector)) == classList[i]):
            right+=1
        elif (clf.predict(array(wordVector))==0 and classList[i]==1):
            errorCount += 1
            shouldSpam += 1
            #print ("Classification error",txtList[i])
        elif (clf.predict(array(wordVector))==1 and classList[i]==0):
            errorCount += 1
            shouldHam +=1
            #print ("Classification error",txtList[i])
    Recall=right/(right+shouldSpam)
    Precision=right/(right+shouldHam)
    Error=errorCount/len(testSet)
    #print('Recall:',Recall)
    #print('Precision:',Precision)
    #print ('Error rate: ',Error)
    end = time.clock()
    Time=end-start
    #print("Time:",Time,"s")
    #return vocabList,fullText
    return Recall,Precision,Error,Time

i=0
MaxRecall=0
MaxPrecision=0
MinError=1
AveRecall=0
AvePrecision=0
AveError=0
AveTime=0
K=10
while K<=80:
    while i<10:
        Recall,Precision,Error,Time=mailfilter(K)
        AveRecall=AveRecall+Recall
        AvePrecision=AvePrecision+Precision
        AveError=AveError+Error
        AveTime=AveTime+Time
        if(Recall>MaxRecall):
            MaxRecall=Recall
        if(Precision>MaxPrecision):
            MaxPrecision=Precision
        if(MinError>Error):
            MinError=Error
        i+=1
    print('***************************')
    print('K=',K)
    print('MaxRecall:',MaxRecall)
    print('MaxPrecision:',MaxPrecision)
    print('MinError:',MinError)
    print('AveRecall:',AveRecall/i)
    print('AvePrecision:',AvePrecision/i)
    print('AveError:',AveError/i)
    print('AveTime:',AveTime/i)
    i=0
    K+=10
    MaxRecall=0
    MaxPrecision=0
    MinError=1
    AveRecall=0
    AvePrecision=0
    AveError=0
    AveTime=0