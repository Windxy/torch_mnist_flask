# torch_mnist_flask
基于Pytorch和Flask web框架来预测手写数字，using Flask and torch and to predict mnist

## 主要的依赖包 Requirements
1.torch
2.torchvision
3.flask

## 使用方法 How to Use
- 1.clone本git，然后下载MNIST数据集，MNIST文件夹解压到DataSet目录下 （Download DataSet by Baidu Netdisk，The MNIST folder is extracted into the DataSet directory）

链接：https://pan.baidu.com/s/13yaI3R4Oun2UF0eoLpLfeQ 
提取码：vnyc 

- 2.模型搭建 Net building
进入model/model.py 进行更改你需要的model，这里使用的是LeNet5（into file"Model/model.py" then make changes to the model you need,LeNet5 here）

- 3.运行train.py训练数据集（Run train.py to train Mnist with LeNet5,）

- 3.运行elevate.py评估数据集（Run elevate.py to evaluate test of DataSet）

- 5.运行app.py，然后点击出现的链接（run app.py then click the link that appears）

## 网页设计参考 refer to
https://github.com/ybsdegit/Keras_flask_mnist

## 博客link
https://blog.csdn.net/qq_33952811/article/details/110227932

## 效果如下：
![image](https://github.com/Windxy/torch_mnist_flask/blob/main/static/show.jpg)
