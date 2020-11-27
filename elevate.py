from model.model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from tqdm import tqdm,trange
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,roc_curve,f1_score,precision_recall_fscore_support

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

if __name__ == '__main__':
    batch_size = 1  # 一个一个测
    test_dataset = mnist.MNIST(root='./DataSet', train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = Model()
    model.load_state_dict(torch.load('model/mnist.pth'))
    threshold = 0.00001

    # 初始化
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    # 1w个数据，980个0
    accurance = 0   #准确率
    correct = 0     #正确的数量
    nums = 0        #0的个数
    y_scores = []
    y_true = []
    y_pred = []
    for (test_x, test_label) in tqdm(test_loader):
        '''在这里，取用0作为我们的正例，1—9作为我们的反例'''
        predict_y = model(test_x.float()).detach()
        predict_ys = predict_y.numpy().squeeze()
        predict_ys = normalization(predict_ys)[0]  # 是0的概率
        y_scores.append(predict_ys)
        if test_label.numpy()==0:
            nums+=1
            correct += 1 if np.argmax(predict_y, axis=-1).item() == 0 else 0

        y_true.append(1 if test_label.numpy()[0]==0 else 0)    #还是用0来进行PR评估,1 表示为 0，0 表示为非 0
        y_pred.append(1 if np.argmax(predict_y, axis=-1).item()==0 else 0)

    accurance = correct*1.0/nums     #980个0
    print("Accuracy:",accurance)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    fpr, tpr, tresholds = roc_curve(y_true, y_scores)
    F1= f1_score(y_true, y_pred)
    print("F1-Score:",F1)

    plt.figure(figsize=(50,100))
    plt.subplot(1,2,1)
    plt.plot(precision, recall)
    plt.xlabel(r'Recall')  # 坐标
    plt.ylabel(r'Precision')
    plt.title("figure of PR-Curve")
    plt.subplot(1,2,2)
    plt.plot(fpr, tpr)
    plt.title("figure of PS")
    plt.xlabel(r'False Positive Rate')  # 坐标
    plt.ylabel(r'True Positive Rate')
    plt.show()


    #     if predict_ys >= threshold:                 # 预测为正例，如果超过这个阈值，预测为正，
    #         TP += test_label.numpy()==0                    # TP(真正例，标签正，预测正)
    #         TN += test_label.numpy()!=0                    # TN(真反例，标签反，预测正)
    #     else:                                       # 预测为反例
    #         FP += test_label.numpy()!=0                    # FP(假正例，标签反，预测反)
    #         FN += test_label.numpy()==0                    # FN(假反例，标签正，预测反)
    #
    # P = TP*1.0/(TP+FP)
    # R = TP*1.0/(TP+FN)
    # print(TP,FN,FP,TN,P,R)