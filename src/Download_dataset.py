from torchvision import datasets, transforms

# 保存路径
data_dir = "../data"

# 简单的数据变换（只用于下载，不做训练）
transform = transforms.ToTensor()

# 下载训练集和测试集
train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

print("✅ CIFAR-10 下载完成！")
print(f"训练集大小: {len(train_set)}，测试集大小: {len(test_set)}")
