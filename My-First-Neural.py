import torch
import time
eps = 1e-8

# 神经网络类
class NeuralNet:
    def __init__(self, Layer0, Layer1, Layer2, Layer3, lr, alpha, dropout_rate):
        # 初始化参数
        self.w1 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer0)), (Layer0, Layer1), requires_grad=True, dtype=torch.float32)
        self.b1 = torch.full((1, Layer1), 0.05, requires_grad=True, dtype=torch.float32)
        self.w2 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer1)), (Layer1, Layer2), requires_grad=True, dtype=torch.float32)
        self.b2 = torch.full((1, Layer2), 0.05, requires_grad=True, dtype=torch.float32)
        self.w3 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer2)), (Layer2, Layer3), requires_grad=True, dtype=torch.float32)
        self.b3 = torch.full((1, Layer3), 0.05, requires_grad=True, dtype=torch.float32)
        
        # 超参数
        self.lr = lr # 学习率
        self.alpha = alpha # 正则化超参数
        self.dropout_rate = dropout_rate # Dropout率
        
        # Batch_Norm 参数
        self.BN_gamma1 = torch.normal(1, 0.02, (1, Layer1), requires_grad=True, dtype=torch.float32)
        self.BN_beta1 = torch.normal(0, 0.02, (1, Layer1), requires_grad=True, dtype=torch.float32)
        self.BN_gamma2 = torch.normal(1, 0.02, (1, Layer2), requires_grad=True, dtype=torch.float32)
        self.BN_beta2 = torch.normal(0, 0.02, (1, Layer2), requires_grad=True, dtype=torch.float32)
        
        # Batch_Norm 均值和方差
        self.running_mean1 = torch.zeros((1, Layer1), dtype=torch.float32)
        self.running_var1 = torch.ones((1, Layer1), dtype=torch.float32)
        self.running_mean2 = torch.zeros((1, Layer2), dtype=torch.float32)
        self.running_var2 = torch.ones((1, Layer2), dtype=torch.float32)
        self.BN_momentum = 0.9
        
        # Adam 优化算法
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                       self.BN_gamma1, self.BN_beta1, self.BN_gamma2, self.BN_beta2]
        self.lr_decay = 0.85
        self.Adam_beta1 = 0.9
        self.Adam_beta2 = 0.99
        self.t = 0
        # 初始化 Adam 的一阶矩和二阶矩
        self.G1 = {}  # 一阶矩
        self.G2 = {}  # 二阶矩
        # 为每个参数初始化 Adam 状态
        for param in self.params:
            self.G1[param] = torch.zeros_like(param)
            self.G2[param] = torch.zeros_like(param)
        
        self.iftraining = True

        # GPU 运行
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        # 将参数移到GPU
        for param in self.params:
            param.data = param.data.to(self.device)
        # 将BN移到GPU
        self.running_mean1 = self.running_mean1.to(self.device)
        self.running_var1 = self.running_var1.to(self.device)
        self.running_mean2 = self.running_mean2.to(self.device)
        self.running_var2 = self.running_var2.to(self.device)
        # 将Adam优化器状态也移到GPU（新增这部分）
        for param in self.params:
            self.G1[param] = self.G1[param].to(self.device)
            self.G2[param] = self.G2[param].to(self.device)

    def TrainMode(self, iftraining: bool):
        self.iftraining = iftraining

    def forward(self, x: torch.Tensor):
        def FC(x, w, b):
            return torch.matmul(x, w) + b
        def ReLU(x):
            return torch.where(x > 0, x, 0.01 * x)
        def BN(x, running_mean, running_var, BN_gamma, BN_beta):
            if self.iftraining:
                mean = torch.mean(x, dim=0, keepdim=True) # 对所有样本的同一维度 进行归一化
                var = torch.var(x, dim=0, keepdim=True)
                running_mean = self.BN_momentum * running_mean + (1 - self.BN_momentum) * mean
                running_var = self.BN_momentum * running_var + (1 - self.BN_momentum) * var
            else:
                mean = running_mean
                var = running_var
            x_norm = (x - mean) / (torch.sqrt(var) + 1e-5)
            x_norm = BN_gamma * x_norm + BN_beta
            return x_norm, running_mean, running_var
        def Dropout(a):
            mask = torch.rand_like(a) > self.dropout_rate
            a = a * mask / (1 - self.dropout_rate)
            return a
        
        a1 = FC(x, self.w1, self.b1)
        a1, self.running_mean1, self.running_var1 = BN(a1, self.running_mean1, self.running_var1, self.BN_gamma1, self.BN_beta1)
        a1 = ReLU(a1)
        a1 = Dropout(a1) if self.iftraining else a1

        a2 = FC(a1, self.w2, self.b2)
        a2, self.running_mean2, self.running_var2 = BN(a2, self.running_mean2, self.running_var2, self.BN_gamma2, self.BN_beta2)
        a2 = ReLU(a2)
        a2 = Dropout(a2) if self.iftraining else a2
        
        a3 = FC(a2, self.w3, self.b3)
        # 注意: 不要在forward中应用softmax，因为交叉熵损失需要原始logits. qwq...
        return a3

    def H_loss(self, y_pred, y_true):
        # def Softmax(x):
        #     exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values) # 这里优化了: exp(x_i) / Σexp(x_j) = exp(x_i - c) / Σexp(x_j - c)
        #     return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
        N = y_true.shape[0]
        H_loss = -1/N * torch.sum(y_true * torch.log_softmax(y_pred, dim=1)) # 使用log_softmax, 我希望这意味着更低的运算花费
        return H_loss

    def ACC(self, y_pred, y_true):
        pred_labels = torch.argmax(y_pred, dim=1)
        true_labels = torch.argmax(y_true, dim=1)
        correct = torch.sum(pred_labels == true_labels).item()
        return correct / y_true.shape[0]



    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def train_step(self, x_batch, y_batch, epoch):
        # 将数据移到GPU
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        def Adam():
            """Adam 优化器更新规则"""
            self.t += 1
            current_lr = self.lr * (self.lr_decay ** epoch)  # 指数衰减
            with torch.no_grad():
                for param in self.params:
                    if param.grad is None:
                        continue
                    # 更新一阶矩和二阶矩估计
                    self.G1[param] = self.Adam_beta1 * self.G1[param] + (1 - self.Adam_beta1) * param.grad
                    self.G2[param] = self.Adam_beta2 * self.G2[param] + (1 - self.Adam_beta2) * (param.grad ** 2)
                    # 偏差修正
                    G1_hat = self.G1[param] / (1 - self.Adam_beta1 ** self.t)
                    G2_hat = self.G2[param] / (1 - self.Adam_beta2 ** self.t)
                    # 更新参数
                    param.data -= current_lr * G1_hat / (torch.sqrt(G2_hat) + eps)

        # 前向传播
        outputs = self.forward(x_batch)
        
        # 反向传播
        L2_loss = self.alpha * (self.w1.pow(2).sum() + self.w2.pow(2).sum() + self.w3.pow(2).sum())
        loss = self.H_loss(outputs, y_batch) + L2_loss
        self.zero_grad()
        loss.backward()
        
        # 更新参数
        Adam()
        
        # 计算准确率
        loss_value = loss.item()
        acc = self.ACC(outputs, y_batch)
        
        return loss_value, acc

    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.device)  # 将输入数据移到GPU
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)
    
def Loader():
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x.view(-1))  # 展平图像
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader

# def Loader(use_augmentation=True):
#     from torchvision import datasets, transforms
    
#     # 定义基础转换：归一化和展平（展平需在数据增强后，避免破坏空间结构）
#     base_transform = [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 三通道均值/标准差均设为0.5
#     ]
    
#     # 训练集数据增强：仅在训练阶段使用，测试阶段禁用
#     if use_augmentation:
#         train_transform = transforms.Compose([
#             # 随机裁剪：先将图像放大到36x36，再随机裁剪回32x32（补充边缘信息）
#             transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
#             # 随机水平翻转：50%概率翻转（符合自然图像对称性，不改变语义）
#             transforms.RandomHorizontalFlip(p=0.5),
#             # 加入微小亮度/对比度扰动（增强鲁棒性，可选）
#             transforms.ColorJitter(brightness=0.1, contrast=0.1),
#             *base_transform,  # 拼接基础转换（转Tensor + 归一化）
#             transforms.Lambda(lambda x: x.view(-1))  # 展平为向量（适配全连接网络输入）
#         ])
#     else:
#         # 无增强的训练集转换（用于对比实验或调试）
#         train_transform = transforms.Compose([
#             transforms.Resize(32),  # 确保尺寸统一
#             *base_transform,
#             transforms.Lambda(lambda x: x.view(-1))
#         ])
    
#     # 测试集转换：禁用任何随机操作，仅保留确定性处理
#     test_transform = transforms.Compose([
#         transforms.Resize(32),
#         *base_transform,
#         transforms.Lambda(lambda x: x.view(-1))
#     ])
    
#     # 加载数据集（训练集用增强转换，测试集用原始转换）
#     trainset = datasets.CIFAR10(
#         root='./data', 
#         train=True, 
#         download=True, 
#         transform=train_transform
#     )
#     testset = datasets.CIFAR10(
#         root='./data', 
#         train=False, 
#         download=True, 
#         transform=test_transform
#     )
    
#     # 构建数据加载器
#     train_loader = torch.utils.data.DataLoader(
#         trainset, 
#         batch_size=100, 
#         shuffle=True,  # 训练集打乱顺序
#         drop_last=True  # 丢弃最后一个不足批次的数据（避免批次大小不一致影响BN）
#     )
#     test_loader = torch.utils.data.DataLoader(
#         testset, 
#         batch_size=100, 
#         shuffle=False  # 测试集无需打乱
#     )
    
#     return train_loader, test_loader

# 训练函数
def train_model(model:NeuralNet, train_loader, epochs=10):
    def one_hot(labels, num_classes=10):
        # 确保标签和设备一致
        device = model.device  # 获取模型的设备
        return torch.eye(num_classes, device=device)[labels.to(device)]
    
    model.TrainMode(True)
    total_start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        batch_count = 0
        
        epoch_start_time = time.time()
        for batch_idx, (data, labels) in enumerate(train_loader):
            batch_start_time = time.time()

            # 转换标签为one-hot编码
            y_true = one_hot(labels)
            
            # 训练一个批次
            loss, acc = model.train_step(data, y_true, epoch)
            
            total_loss += loss
            total_acc += acc
            batch_count += 1
            
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {(batch_idx+1)}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        
        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count
        epoch_time = time.time() - epoch_start_time

        print(f'Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Time: {epoch_time:.2f}s')
    
    total_time = time.time() - total_start_time
    print(f'Total training time: {total_time:.2f}s')

# 测试函数
def test_model(model:NeuralNet, test_loader):
    model.TrainMode(False)
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(model.device)  # 移动数据到GPU
            labels = labels.to(model.device)  # 移动标签到GPU
            predictions = model.predict(data)
            total_correct += torch.sum(predictions == labels).item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    return accuracy

# 主程序
if __name__ == "__main__":
    
    # 加载数据
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = Loader()
    
    # 创建模型
    model = NeuralNet(Layer0=3*32*32, Layer1=512, Layer2=256, Layer3=10, lr=0.005, alpha=2e-5, dropout_rate=0.2)
    
    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, epochs=10)
    
    # 测试模型
    print("Testing model...")
    test_accuracy = test_model(model, test_loader)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")