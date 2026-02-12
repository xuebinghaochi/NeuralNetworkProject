import numpy as np
import mnist
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# --- 1. 加载您的数据 ---
train_images = mnist.train_images(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
train_labels = mnist.train_labels(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
test_images = mnist.test_images(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")
test_labels = mnist.test_labels(target_dir="/Users/qs/PyCharmMiscProject/KerasForNeuralNetwork/data/mnist/MNIST/raw")

print(f"原始训练数据形状: {train_images.shape}")
print(f"原始测试数据形状: {test_images.shape}")

# --- 2. 数据预处理 ---
# 归一化到 [-0.5, 0.5] 并展平
train_images = (train_images / 255.0) - 0.5
test_images = (test_images / 255.0) - 0.5

train_images = train_images.reshape(-1, 784).astype(np.float32)
test_images = test_images.reshape(-1, 784).astype(np.float32)

# --- 3. 转换为 PyTorch 张量 ---
train_images_tensor = torch.tensor(train_images)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = torch.tensor(test_images)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# --- 4. 创建 Dataset 和 DataLoader ---
train_dataset = torch.utils.data.TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = torch.utils.data.TensorDataset(test_images_tensor, test_labels_tensor)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 5. 设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 6. 定义模型 ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入 784 维，输出 128 维
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)   # 输出 10 个类别的分数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP().to(device)

# --- 7. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# --- 8. 训练模型 ---
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 将数据移动到设备
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# --- 9. 评估模型 ---
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'\n✅ 测试准确率: {accuracy:.2f}%')