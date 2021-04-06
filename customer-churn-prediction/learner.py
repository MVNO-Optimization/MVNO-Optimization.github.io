from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import OneSidedSelection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def learner(trainX, trainY):
    # OneSidedSelection 欠采样方法，由于样本数据集较小，因此可不进行重采样；全量数据集则需要使用
    # oss = OneSidedSelection()
    # trainX, trainY = oss.fit_resample(trainX, trainY)
    brf = RandomForestClassifier(random_state=42)
    brf.fit(trainX, trainY)
    return brf

def predict(testX, testY, clf):
    predY = clf.predict(testX)
    acc = accuracy_score(testY, predY)
    recall = recall_score(testY, predY)
    precision = precision_score(testY, predY)
    f1 = f1_score(testY, predY)
    print("准确率：" + str(acc) + ",召回率：" + str(recall) + ",精度：" + str(precision) + ",F1：" + str(f1))

if __name__ == "__main__":
    # 样本数据集的数据和效果仅供参考，模型参数还需根据具体数据集来选择
    file = open("../data/samples-churn-features.txt", 'r')
    lines = file.readlines()[1:]
    datasetX = []
    datasetY = []
    ids = []
    for line in lines:
        arr = line.split(',')
        ids.append(arr[0])
        if len(arr) < 3:
            continue
        tmp = []
        for i in range(1, len(arr) - 1):
            tmp.append(float(arr[i]))
        datasetX.append(tmp)
        datasetY.append(int(arr[-1].strip('\n')))

    # 实际应用时应采用十折交叉验证，此处由于样本数据集较小略有简化
    train_X, test_X, train_y, test_y = train_test_split(datasetX, datasetY, test_size=0.2, random_state=5, shuffle=True)

    clf = learner(train_X, train_y)
    predict(test_X, test_y, clf)


