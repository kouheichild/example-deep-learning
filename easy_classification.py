import numpy as np
import pandas as pd
from sklearn import datasets
# 教師データと訓練データの分割
from sklearn.model_selection import train_test_split
import torch
# 自動微分を行うためのpytorchの形式
from torch.autograd import Variable as V
# ニューラルネットワークのモデル
import torch.nn as nn
# 活性化関数
import torch.nn.functional as F
# 最適化，確率的勾配降下法など
import torch.optim as optim

# データセット
iris = datasets.load_iris()
# one-hot-vector表現のベクトルに変換
    # 0,1,2 の分類であるため
y = np.zeros((len(iris.target),1 + iris.target.max()),dtype=int)
y[np.arange(len(iris.target)),iris.target]=1

#train_test_split(data, label, test_size=)
X_train, X_test, y_train, y_test = train_test_split(iris.data,y,test_size=0.25)

#Pytorchで取り扱える形式に変換,V(data,requires_grad=Trueで自動微分)
x=V(torch.from_numpy(X_train).float(), requires_grad = True)
y=V(torch.from_numpy(y_train).float())

# ニューラルネットワークのモデル構成
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(4,10)#入力層4,隠れ層1層目10
        self.fc2 = nn.Linear(10,8)#隠れ層1層目10, 隠れ層2層目8
        self.fc3 = nn.Linear(8,3)#隠れ層２層目8, 出力層3

    def forward(self,x):
        x = F.relu(self.fc1(x))#隠れ層1->2 活性化関数
        x = F.relu(self.fc2(x))#隠れ層2->出力 活性化関数
        x = self.fc3(x)        #出力の活性化関数, ex) シグモイド，恒等関数，ソフトマックス関数
        return x

# 学習
net = Net()
# 最適化の方法，optim.SGD(net.parameters(), lr = 学習率)
                #SGD..確率的勾配降下法
optimizer = optim.SGD(net.parameters(),lr = 0.01)
# 誤差関数, nn.__()
        #__ ..MSELoss():平均二乗誤差, CrossEntrooyLoss():交差エントロピー誤差
criterion = nn.MSELoss()
# エポック数
epochs = 3000

# START LEARNING!!!
losses=[]
for epoch in range(epochs):
    # 勾配の初期化
    optimizer.zero_grad()
    # 予測
    y_pred = net(x)
    #　誤差の計算, criterion(y_pred=予測値, y=正解ラベル)
    loss = criterion(y_pred,y)
    # 誤差逆伝播
    loss.backward()
    # 勾配の更新
    optimizer.step()
    # 収束の確認
    losses.append(loss.data)

# TEST
outputs = net(V(torch.from_numpy(X_test).float()))
_, predicted = torch.max(outputs.data,1)
y_predicted = predicted.numpy()
y_true=np.argmax(y_test,axis=1)
accuracy = 100*np.sum(y_predicted==y_true) / len(y_predicted)
print("accuracy: {}%".format(round(accuracy, 3)))
