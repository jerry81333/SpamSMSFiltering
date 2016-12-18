import random
from numpy import *
import os
import time

'''
def createData(Max):
    data   = []  
    labels = []
    i=1
    j=1

    with open("SMSSpamCollection.data") as ifile:  
        for line in ifile:
            rowDict = {}
            tokens = line.strip().split('\t')
            #print(tokens)
            rowDict['label']=tokens[0]
            rowDict['message']=tokens[1]
            if(i==Max and j==Max):
                break
            if(tokens[0]=="spam" and i<Max):
                f = open('spam/%d.txt' % i, "w+")
                f.write(tokens[1])
                f.close
                i=i+1
            if(tokens[0]=="ham" and j<Max):
                f = open('ham/%d.txt' % j, "w+")
                f.write(tokens[1])
                f.close
                j=j+1
'''

def creatVocabList(Data):
    List = set([])  #create empty set
    for document in Data:
        List = List | set(document) #union of the two sets
    #print(len(List))
    return list(List)

def naiveBayesTrain(trainData,trainLabel):
    N = len(trainData)
    NWords = len(trainData[0])
    pAbusive = sum(trainLabel)/N
    p0Num = ones(NWords); p1Num = ones(NWords) #Laplace smoothing 
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(N):
        if trainLabel[i] == 1:        
            p1Num += trainData[i]
            p1Denom += sum(trainData[i])
        else:
            p0Num += trainData[i]
            p0Denom += sum(trainData[i])
    p1Vect = log(p1Num/p1Denom) #use log()
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

def naiveBayesClf(testVec, p0Vec, p1Vec, pClass1):
    p1 = sum(testVec * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(testVec * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

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
        
def mailfilter(K):
    #createData(81)
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
    p0V,p1V,pSpam = naiveBayesTrain(array(trainMat),array(trainClasses))

    right = 0
    errorCount = 0
    shouldSpam = 0
    shouldHam = 0
    for i in testSet:        #classify the remaining items
        wordVector = dataToList(vocabList, txtList[i])
        if(naiveBayesClf(array(wordVector),p0V,p1V,pSpam) == classList[i]):
            right+=1
        elif (naiveBayesClf(array(wordVector),p0V,p1V,pSpam)==0 and classList[i]==1):
            errorCount += 1
            shouldSpam += 1
            #print ("Classification error",txtList[i])
        elif (naiveBayesClf(array(wordVector),p0V,p1V,pSpam)==1 and classList[i]==0):
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
