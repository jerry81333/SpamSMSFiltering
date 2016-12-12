from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    word = set([])  #create empty set
    for document in dataSet:
        word = word | set(document) #union of the two sets
    print("word:",word)
    return list(word)

def setOfWords2Vec(vocabList, inputWord):
    returnVec = [0]*len(vocabList)
    for word in inputWord:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: 
            print ("the word: %s is not in my Vocabulary!" % word)
        #print("returnVec:",returnVec)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    print("len(trainMatrix)",len(trainMatrix))
    print("trainMatrix",trainMatrix)
    numWords = len(trainMatrix[0])
    print("len(trainMatrix[0]",len(trainMatrix[0]))
    print("trainMatrix[0]",trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #Laplace Smoothing, change to 2.0
    for i in range(numTrainDocs): #prior probability
        if trainCategory[i] == 1:        
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print("p0Num:",p0Num)
    print("p1Num:",p1Num)
    print("p0Denom:",p0Denom)
    print("p1Denom:",p1Denom)

    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()

    print("p1Num/p1Denom:",p1Num/p1Denom)
    print("p1Vect:",p1Vect,"p0Vect:",p0Vect)
    print("pAbusive",pAbusive)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    print(pAb)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage','worthless']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()