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
torch.multiprocessing.set_sharing_strategy('file_system')

timer = d2l.Timer()
timer.start()
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=11, padding=1,stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Flatten()
        self.L1 = nn.Linear(256 * 5 * 5 ,4096)  # 输入是16 * 5 * 5
        self.fc2 = nn.Dropout(p=0.5)
        self.L2 = nn.Linear(4096,4096)
        self.fc3 = nn.Dropout(p=0.5)
        self.L3 = nn.Linear(4096,num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = self.maxpool1(X)
        X = F.relu(self.conv2(X))
        X = self.maxpool2(X)
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))
        X = F.relu(self.conv5(X))
        X = self.maxpool3(X)
        X = self.fc1(X)
        X = F.relu(self.L1(X))
        X = self.fc2(X)
        X = F.relu(self.L2(X))
        X = self.fc3(X)
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
    # 使用 torch.cat 将所有的数据和标签合并为一个大的张量
    combined_data = torch.cat(all_data, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    combined_data = combined_data.to(device).clone().detach()
    combined_labels = combined_labels.to(device).clone().detach()
    batch_size = len(all_data)
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

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    n = 0
    accu = 0.
    net.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device='cuda'), y.to(device='cuda')
            accu += accuracy(net(X), y)
            n += y.numel()
    return accu / n


def train(net, train_iter, tr_feat, tr_lab, te_feat, te_lab, test_iter, loss, optimizer,epochs):
    train_loss_hist = []
    test_loss_hist = []
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss1 = 0
        num_samples = 0
        for X, y in train_iter:
            X, y = X.to(device='cuda'), y.to(device='cuda')
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            optimizer.step()
            train_loss1 += l.item() * X.shape[0]
            num_samples += X.shape[0]
        loss_avg = train_loss1 / num_samples
        train_loss_hist.append(loss_avg)


    train_accuracy = evaluate_accuracy(net, train_iter)
    test_accuracy = evaluate_accuracy(net, test_iter)
    print(f'Train Accuracy: {train_accuracy * 100:.3f}%\n')
    print(f'Test Accuracy: {test_accuracy * 100:.3f}%\n')

    return train_loss_hist

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
    batch_size = 32
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
    train_features, train_labels, train_iter = data_to_device(train_iter, device='cpu')
    test_features, test_labels, test_iter = data_to_device(test_iter, device='cpu')

    net=AlexNet()
    net = nn.Sequential(
        # 这里使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))
    net.apply(init_weights)
    net.to(device='cuda')

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    epochs = 5

    tr_loss = train(net, train_iter, train_features, train_labels, test_features, test_labels, test_iter, loss, optimizer,epochs)
    torch.save(net, 'AlexNet.pth')
    #print(l)
    #tr_loss, te_loss = net_eval(net, train_iter, train_features, train_labels, test_features, test_labels, test_iter, loss, optimizer)
    n = len(tr_loss)
    n = np.arange(n)
    plt.plot(n,tr_loss,color='blue',label='train loss')
    #plt.plot(n,te_loss,color='red',label='test loss')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('train and test loss with iterations')
    plt.show()
    timer.stop()
    print(f'time_cost:  {timer.sum()}\n')