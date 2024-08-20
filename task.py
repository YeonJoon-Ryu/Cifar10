import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


root_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
mean = root_dataset.data.mean(axis=(0,1,2))/255
std = root_dataset.data.std(axis=(0,1,2))/255 #255로 나누는 이유는 픽셀 값을 [0,1]로 변환하기 위함. RGB는 0~255사이 정수이기 때문

print(mean,std)
transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((mean), (std))
])

class Custom_Cifar(Dataset):
    def __init__(self, transform = None):
        self.cifar10 = root_dataset
        self.transform = transform

    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        image, label = self.cifar10[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
custom_dataset = Custom_Cifar(transform= transform)

loader = DataLoader(dataset=custom_dataset, batch_size=8, shuffle=True)
data_iter = iter(loader)
images, labels = next(data_iter)

# images = images/2 + (1 - torch.tensor(mean).view(1, 3, 1, 1)) #정규화 해야 사용
# images = torch.clamp(images, 0, 1)

#위 코드로 하려다가 아무래도 망해서 다시 검색 시작....

def denormalize(imgs, mean, std): #역정규화를 정의
    mean = torch.tensor(mean).view(1, 3, 1, 1) #view는 텐서의 차원에 따라 브로드 캐스팅이 불가능 할 수 있기에 모양을 바꿔줌.
    std = torch.tensor(std).view(1, 3, 1, 1)
    return imgs * std + mean

images = denormalize(images, mean, std).clamp(0, 1) #특정 픽셀의 값이 유효범위인 [0,1]을 벗어날 수 있기에 강제함

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

class_names = custom_dataset.cifar10.classes

for idx in range(8):
    ax = axes[idx//4, idx%4]
    ax.imshow(np.transpose(images[idx].numpy(), (1,2,0)))
    ax.set_title(class_names[labels[idx]])
    ax.axis('off')
plt.show()
