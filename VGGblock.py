import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import torch.utils.data as data
import d2l.torch as d2l
torch.multiprocessing.set_sharing_strategy('file_system')
timer = d2l.Timer()
timer.start()
class VGGBlock(nn.Module):
    def __init__(self, conv_arch):
        super().__init__()
        self.conv_blks = nn.ModuleList()
        in_channels = 1
        # 卷积层部分
        for (num_convs, out_channels) in conv_arch:
            self.conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        # 全连接层部分
        self.net_out = nn.Sequential(nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

    def forward(self, X):
        for blk in self.conv_blks:
            X = blk(X)
        X = self.net_out(X)
        return X

    def vgg_block(self, num_convs,in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

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

def evaluate_accuracy(net, data_iter):  # @save
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
if __name__ == '__main__':
    batch_size = 32
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    train_features, train_labels, train_iter = data_to_device(train_iter, device='cpu')
    test_features, test_labels, test_iter = data_to_device(test_iter, device='cpu')

    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = VGGBlock(conv_arch)
    net.apply(init_weights)
    net.to(device='cuda')

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    epochs = 5

    tr_loss = train(net, train_iter, train_features, train_labels, test_features, test_labels, test_iter, loss, optimizer,
              epochs)
    torch.save(net, 'VGG.pth')
    n = len(tr_loss)
    n = np.arange(n)
    plt.plot(n, tr_loss, color='blue', label='train loss')
    # plt.plot(n,te_loss,color='red',label='test loss')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('train and test loss with iterations')
    plt.show()
    timer.stop()
    print(f'time_cost:  {timer.sum()}\n')