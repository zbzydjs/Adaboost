import numpy as np


def loadData(filename):
    numFeat = len(open(filename).readline().split("\t"))
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat



def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == "lt":
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ["lt", "gt"]:
                threshVal = rangeMin + float(j)*stepSize
                predictedVal = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVal == labelMat] = 0
                weightedError = D.T*errArr
                print("split: dimen %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" 
                      %(i, threshVal, inequal, weightedError))
                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVal.copy()
                    bestStump["dim"] = i
                    bestStump["thresh"] = threshVal
                    bestStump["ineq"] = inequal
    return bestStump, minError, bestClasEst
    

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []             # 加入由buildStump选出的弱分类器
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)      # 初始化每个样本点的权重为1/m
    aggClassEst = np.mat(np.zeros((m, 1)))   # 初始化预测结果
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump["alpha"] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))/np.sum(D)
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("errorRate: ", errorRate, "\n")
        if errorRate == 0.0:
            break
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    dataMatrix = np.mat(dataToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]["dim"], classifierArr[i]["thresh"],
                                 classifierArr[i]["ineq"])
        aggClassEst += classEst * classifierArr[i]["alpha"]
        print(aggClassEst)
    return np.sign(aggClassEst)

                
        
        
        

if __name__ == "__main__":
    dataMatrix, classLabels = loadData(r"./horseColicTraining2.txt")
    a = adaBoostTrainDS(dataMatrix, classLabels, 10)
#    b = adaClassify([0,0], a)
    print(a)
    

