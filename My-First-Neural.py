import numpy as np
import torch

# 神经网络类
class NeuralNet:
    def __init__(self, Layer0, Layer1, Layer2, Layer3, lr, alpha, dropout_rate, momentum):
        # 初始化参数
        self.w1 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer0)), (Layer0, Layer1), requires_grad=True, dtype=torch.float32)
        self.b1 = torch.zeros((1, Layer1), requires_grad=True, dtype=torch.float32)
        self.w2 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer1)), (Layer1, Layer2), requires_grad=True, dtype=torch.float32)
        self.b2 = torch.zeros((1, Layer2), requires_grad=True, dtype=torch.float32)
        self.w3 = torch.normal(0, torch.sqrt(torch.tensor(2.0 / Layer2)), (Layer2, Layer3), requires_grad=True, dtype=torch.float32)
        self.b3 = torch.zeros((1, Layer3), requires_grad=True, dtype=torch.float32)

        # Batch Normalization 参数
        self.gamma1 = torch.ones((1, Layer1), requires_grad=True, dtype=torch.float32)
        self.beta1 = torch.zeros((1, Layer1), requires_grad=True, dtype=torch.float32)
        self.gamma2 = torch.ones((1, Layer2), requires_grad=True, dtype=torch.float32)
        self.beta2 = torch.zeros((1, Layer2), requires_grad=True, dtype=torch.float32)
        
        # 缓存 BN 的均值和方差
        self.running_mean1 = torch.zeros((1, Layer1), dtype=torch.float32)
        self.running_var1 = torch.ones((1, Layer1), dtype=torch.float32)
        self.running_mean2 = torch.zeros((1, Layer2), dtype=torch.float32)
        self.running_var2 = torch.ones((1, Layer2), dtype=torch.float32)
        self.momentum = momentum
        
        self.lr = lr
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
                       self.gamma1, self.beta1, self.gamma2, self.beta2]
        self.iftraining = True

    def TrainMode(self, iftraining: bool):
        self.iftraining = iftraining

    def forward(self, x: torch.Tensor):
        def FC(x, w, b):
            return torch.matmul(x, w) + b
        def ReLU(x):
            return torch.where(x > 0, x, 0.01 * x)
        def BN(x, running_mean, running_var, gamma, beta):
            if self.iftraining:
                mean = torch.mean(x, dim=0, keepdim=True) # 对所有样本的同一维度 进行归一化
                var = torch.var(x, dim=0, keepdim=True)
                running_mean = self.momentum * running_mean + (1 - self.momentum) * mean
                running_var = self.momentum * running_var + (1 - self.momentum) * var
            else:
                mean = running_mean
                var = running_var
            x_norm = (x - mean) / (torch.sqrt(var) + 1e-5)
            x_norm = gamma * x_norm + beta
            return x_norm, running_mean, running_var
        def Dropout(a):
            mask = torch.rand_like(a) > self.dropout_rate
            a = a * mask / (1 - self.dropout_rate)
            return a
        
        a1 = FC(x, self.w1, self.b1)
        a1, self.running_mean1, self.running_var1 = BN(a1, self.running_mean1, self.running_var1, self.gamma1, self.beta1)
        a1 = ReLU(a1)
        a1 = Dropout(a1) if self.iftraining else a1

        a2 = FC(a1, self.w2, self.b2)
        a2, self.running_mean2, self.running_var2 = BN(a2, self.running_mean2, self.running_var2, self.gamma2, self.beta2)
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
        def update_params():
            with torch.no_grad():
                current_lr = self.lr * (0.95 ** epoch)  # 指数衰减
                for param in self.params:
                    param -= current_lr * param.grad
        # 前向传播
        outputs = self.forward(x_batch)
        
        # 反向传播
        L2_loss = self.alpha * (self.w1.pow(2).sum() + self.w2.pow(2).sum() + self.w3.pow(2).sum())
        loss = self.H_loss(outputs, y_batch) + L2_loss
        self.zero_grad()
        loss.backward()
        
        # 更新参数
        update_params()
        
        # 计算准确率
        loss_value = loss.item()
        acc = self.ACC(outputs, y_batch)
        
        return loss_value, acc

    def predict(self, x):
        with torch.no_grad():
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
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader



# 训练函数
def train_model(model:NeuralNet, train_loader, epochs=10):
    def one_hot(labels, num_classes=10):
        return torch.eye(num_classes, device=labels.device)[labels]
    
    model.TrainMode(True)
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        batch_count = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # 转换标签为one-hot编码
            y_true = one_hot(labels)
            
            # 训练一个批次
            loss, acc = model.train_step(data, y_true, epoch)
            
            total_loss += loss
            total_acc += acc
            batch_count += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        
        avg_loss = total_loss / batch_count
        avg_acc = total_acc / batch_count
        print(f'Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}')

# 测试函数
def test_model(model:NeuralNet, test_loader):
    model.TrainMode(False)
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            predictions = model.predict(data)
            total_correct += torch.sum(predictions == labels).item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# 主程序
if __name__ == "__main__":
    
    # 加载数据
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = Loader()
    
    # 创建模型
    model = NeuralNet(Layer0=3*32*32, Layer1=512, Layer2=256, Layer3=10, lr=0.005, alpha=0, dropout_rate=0.2, momentum=0.9)
    
    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, epochs=20)
    
    # 测试模型
    print("Testing model...")
    test_accuracy = test_model(model, test_loader)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")