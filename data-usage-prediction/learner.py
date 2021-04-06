from sklearn.svm import SVR
import numpy as np
import math

# Grubbs statistics
T = 20

def grubbs_test(arr):
    mu = np.mean(arr)
    tsum = 0
    for item in arr:
        tsum += abs(item - mu)
    tsum /= len(arr) - 1
    s = math.sqrt(tsum)
    for i in range(len(arr)):
        gval = abs(arr[i] - mu) / s
        if gval > T:
            # label the value as an outlier
            arr[i] = -1

def neighbor_mean_interpolation(arr):
    for i in range(len(arr)):
        if arr[i] == -1:
            left = i-1
            right = i+1
            while not (left >= 0 and arr[left] >= 0):
                left -= 1
                if left < 0:
                    break
            while not (right < len(arr) and arr[right] >= 0):
                right += 1
                if right >= len(arr):
                    break

            if left < 0 and right >= len(arr):
                arr[i] = 0
            else:
                if left < 0:
                    arr[i] = arr[right]
                elif right >= len(arr):
                    arr[i] = arr[left]
                else:
                    arr[i] = (arr[left] + arr[right]) / 2

def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def learner(trainX, trainY):
    svr_rbf = SVR(kernel='rbf')
    svr_rbf.fit(trainX, trainY)
    return svr_rbf

def predict(id, testX, testY, clf):
    predY = clf.predict(testX)
    if predY[0] < 0:
        predY[0] = 0
    if testY[0] == 0 and predY[0] == 0:
        acc = 1
    else:
        acc = 1 - (abs(predY[0] - testY[0]) / max(predY[0], testY[0]))
    print("用户" + id + "的次月流量预测值为：" + str(predY[0]))
    print("用户" + id + "的次月流量真实值为：" + str(testY[0]))
    print("预测准确率为：" + str(acc))

if __name__ == "__main__":
    # 样本数据集的数据和预测效果仅供参考，模型具体参数视数据集不同而调整
    du_file = open("../data/samples-data-usage.txt", 'r')
    du_lines = du_file.readlines()
    for line in du_lines:
        sarr = line.split(',')
        id = sarr[0]
        arr = []
        flag = False
        for i in range(1, len(sarr)):
            if sarr[i] != "0":
                flag = True
            if flag:
                arr.append(float(sarr[i]))
        arr = np.array(arr)
        if len(arr) < 4:
            print("用户" + id + "的流量序列过短，不予预测")
            continue

        # 异常检测和插值，实际上本仓库里的样本数据集已经过清洗，所以无需执行下述函数
        #grubbs_test(arr)
        #neighbor_mean_interpolation(arr)

        dataX, dataY = create_dataset(arr)
        trainX = dataX[:-1]
        trainY = dataY[:-1]
        testX = dataX[-1:]
        testY = dataY[-1:]
        print("训练集：", trainX, trainY)
        print("测试集：", testX, testY)

        clf = learner(trainX, trainY)
        predict(id, testX, testY, clf)