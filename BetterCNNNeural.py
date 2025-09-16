
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from torch.optim.lr_scheduler import StepLR

# 超参数设置
Layer1 = 512          # 全连接层1
Layer2 = 256          # 全连接层2
Layer3 = 10           # 输出层（10类）
lr=0.001
alpha = 1e-3          # L2正则化参数
dropout_rate = 0.2
epochs = 10
batch_size = 128
test_interval = 3     # 每3个epoch测试一次

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用设备:", device)

# 数据预处理（增加数据增强）
print("Loading CIFAR-10 dataset...")
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, Layer1, Layer2, Layer3, dropout_rate):
        super(CNN, self).__init__()
        # 卷积层1：输入3通道（RGB），输出32通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化，针对2D卷积
        # 卷积层2：32输入通道，64输出通道，3x3卷积核，padding=1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # 池化层：2x2最大池化，步幅2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层：输入是64*8*8（经过两次池化后特征图为8x8），输出Layer1
        self.fc1 = nn.Linear(64 * 8 * 8, Layer1)
        self.bn_fc1 = nn.BatchNorm1d(Layer1)
        self.fc2 = nn.Linear(Layer1, Layer2)
        self.bn_fc2 = nn.BatchNorm1d(Layer2)
        self.fc3 = nn.Linear(Layer2, Layer3)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [batch_size, 3, 32, 32]
        x = F.relu(self.bn1(self.conv1(x)))  # 卷积 -> 批归一化 -> ReLU
        x = self.pool(x)  # 池化：32x32 -> 16x16
        x = F.relu(self.bn2(self.conv2(x)))  # 卷积 -> 批归一化 -> ReLU
        x = self.pool(x)  # 池化：16x16 -> 8x8
        x = x.view(x.size(0), -1)  # 展平为[batch_size, 64*8*8]
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练和测试函数（保持不变）
def train_epoch(epoch, model, train_loader, scheduler, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch {epoch}/{epochs}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}')
    
    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_time

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = correct / total
    return acc

if __name__ == '__main__':
    # 数据加载
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = CNN(Layer1, Layer2, Layer3, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练循环
    print("Starting training...")
    total_start_time = time.time()
    epoch_times = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss, epoch_acc, epoch_time = train_epoch(epoch, model, train_loader, scheduler, optimizer, criterion)
        epoch_times.append(epoch_time)
        
        # 计算剩余时间
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (epochs - epoch)
        
        print(f'Epoch {epoch}/{epochs} completed - Avg Loss: {epoch_loss:.4f}, Avg Acc: {epoch_acc:.4f}, Time: {epoch_time:.2f}s')
        print(f'Estimated remaining time: {remaining_time:.2f}s')
        
        # 每3个epoch测试一次
        if epoch % test_interval == 0:
            test_acc = test_model(model, test_loader)
            print(f'========== Epoch {epoch} 测试完成 - 测试准确率: {test_acc:.4f} ==========')

    total_time = time.time() - total_start_time
    print(f'Total training time: {total_time:.2f}s')

    # 最终测试
    print("Testing model...")
    final_acc = test_model(model, test_loader)
    print(f'Final Test Accuracy: {final_acc:.4f}')