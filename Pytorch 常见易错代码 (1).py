import numpy as np
import torch
import torchvision.transforms as transforms

# 在torch训练的时候,图片的格式tensor: CxBxHxW
Channels = 3
Height = 256
Width = 256

# 创建numpy array
array = np.random.randn(Channels,Height,Width)
print(array.shape)

# 从numpy array 到 tensor
Tensor_array = torch.from_numpy(array)
print(Tensor_array.shape)

# 用torchvision中的transforms来转换numpy array 到 tensor
Transform = transforms.ToTensor()
Transformed_Tensor = Transform(array)
print(Transformed_Tensor.shape)

# 由于transforms的ToTensor 会改变numpy的dimension的位置, 因此, 在调用ToTensor之前要先把numpy array 转换成 HxWxC 的格式