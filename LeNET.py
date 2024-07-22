import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import d2l.torch as d2l
import tqdm
from torch import device


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Flatten()
        self.L1 = nn.Linear(16 * 5 * 5 ,120)  # 输入是16 * 5 * 5
        self.L2 = nn.Linear(120,84)
        self.L3 = nn.Linear(84,num_classes)

    def forward(self, X):
        X = F.sigmoid(self.conv1(X))
        X = self.avgpool1(X)
        X = F.sigmoid(self.conv2(X))
        X = self.avgpool2(X)
        X = self.fc1(X)
        X = F.sigmoid(self.L1(X))
        X = F.sigmoid(self.L2(X))
        X = self.L3(X)
        return X

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # shuffle 打乱数据

def data_to_device(data_iter, device='cpu'):
    # 初始化一个空的列表来存储所有的数据和标签
    all_data = []
    all_labels = []
    # 遍历 DataLoader，收集所有的数据和标签
    for inputs, labels in data_iter:
        all_data.append(inputs)
        all_labels.append(labels)
    batch_size = len(all_data)
    # 使用 torch.cat 将所有的数据和标签合并为一个大的张量
    combined_data = torch.cat(all_data, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    combined_data = combined_data.to(device)
    combined_labels = combined_labels.to(device)
    return combined_data, combined_labels, load_array((combined_data, combined_labels), batch_size, is_train=False)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  # @save
    """计算在指定数据集上模型的精度"""
    n = 0
    accu = 0.
    net.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        for X, y in data_iter:
            accu += accuracy(net(X), y)
            n += y.numel()
    return accu / n

def train(net, train_iter, tr_feat, tr_lab, te_feat, te_lab, test_iter, loss, optimizer):
    train_loss_hist = []
    test_loss_hist = []
    for epoch in tqdm.tqdm(range(epochs)):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            optimizer.step()
        train_loss_hist.append(loss(net(tr_feat), tr_lab).cpu().item())
        test_loss_hist.append(loss(net(te_feat), te_lab).cpu().item())
    train_accuracy = evaluate_accuracy(net, train_iter)
    test_accuracy = evaluate_accuracy(net, test_iter)
    print(f'Train Accuracy: {train_accuracy * 100:.3f}%\n')
    print(f'Test Accuracy: {test_accuracy * 100:.3f}%\n')
    return train_loss_hist, test_loss_hist

def net_eval(net, train_iter, tr_feat, tr_lab, te_feat, te_lab, test_iter, loss, optimizer):
    net.eval()
    train_loss_hist = loss(net(tr_feat), tr_lab).cpu().item()
    test_loss_hist = loss(net(te_feat), te_lab).cpu().item()
    train_accuracy = evaluate_accuracy(net, train_iter)
    test_accuracy = evaluate_accuracy(net, test_iter)
    print(f'Train Accuracy: {train_accuracy * 100:.3f}%\n')
    print(f'Test Accuracy: {test_accuracy * 100:.3f}%\n')
    return train_loss_hist, test_loss_hist
if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    train_features, train_labels, train_iter = data_to_device(train_iter, device='cuda')
    test_features, test_labels, test_iter = data_to_device(test_iter, device='cuda')

    net = LeNet()
    net.apply(init_weights)
    #net = torch.load('LeNet.pth')
    net.to(device='cuda')

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 100

    tr_loss, te_loss = train(net, train_iter, train_features, train_labels, test_features, test_labels, test_iter, loss, optimizer)
    torch.save(net, 'LeNet.pth')
    #tr_loss, te_loss = net_eval(net, train_iter, train_features, train_labels, test_features, test_labels, test_iter, loss, optimizer)
    n = len(tr_loss)
    n = np.arange(n)
    plt.plot(n,tr_loss,color='blue',label='train loss')
    plt.plot(n,te_loss,color='red',label='test loss')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('train and test loss with iterations')
    plt.show()