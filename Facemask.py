

!pip install kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download omkargurav/face-mask-dataset

from zipfile import ZipFile
dataset='/content/face-mask-dataset.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import random
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

with_mask_files=os.listdir('/content/data/with_mask')
without_mask_files=os.listdir('/content/data/without_mask')

print(with_mask_files[0:5])
print(with_mask_files[-5:])

print(without_mask_files[0:5])
print(without_mask_files[-5:])

print('number of mask images:',len(with_mask_files))
print('number of non-mask images',len(without_mask_files))

with_mask_labels=[1]*len(with_mask_files)
without_mask_labels=[0]*len(without_mask_files)

print(with_mask_labels[0:5])
print(without_mask_labels[0:5])

print(len(with_mask_labels))
print(len(without_mask_labels))

labels=with_mask_labels+without_mask_labels
print(len(labels))
print(labels[0:5])
print(labels[-5:])

img=mpimg.imread('/content/data/with_mask/with_mask_1545.jpg')
imgplot=plt.imshow(img)
plt.show()

#convert the images to numpy arrays
with_mask_path='/content/data/with_mask/'
without_mask_path='/content/data/without_mask/'
data=[]

for img_file in with_mask_files:

  image=Image.open(with_mask_path+img_file)
  image=image.resize((128,128))
  image=image.convert('RGB')
  image=np.array(image)
  data.append(image)

  # image=cv2.imread(os.path.join(with_mask_path,img_file))
for img_file in without_mask_files:

  image=Image.open(without_mask_path+img_file)
  image=image.resize((128,128))
  image=image.convert('RGB')
  image=np.array(image)
  data.append(image)

type(data)

#converting images and labels to the numpy array
X=np.array(data)
Y=np.array(labels)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# %matplotlib inline

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

#scaling  the data
X_train_scaled=X_train/255
X_test_scaled=X_test/255

train_dataset = TensorDataset(X_train_scaled, Y_train)
test_dataset = TensorDataset(X_test_scaled, Y_test)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

train_dataset[12]

class FaceMask(nn.Module):
  def __init__(self):
    super().__init__()
    #input is[1,3,128,128]
    self.conv1 = nn.Conv2d(3, 6, 3, 1) #output [1,6,126,126] and when passed through the pooling layer of 2 by 2 ,it will output (1,6,63,63)
    self.conv2 = nn.Conv2d(6,16,3,1)  #output [1,16,61,61] and when passed through the pooling layer of 2 by 2 ,it will output (1,16,30,30)
    self.fc1=nn.Linear(16*30*30,1200)
    self.fc2=nn.Linear(1200,400)
    self.fc3=nn.Linear(400,150)
    self.fc4=nn.Linear(150,2)

  def forward(self,x):
    x = x.permute(0, 3, 1, 2)
    x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
    x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
    x = x.reshape(-1, 16 * 30 * 30)
    x=F.relu(self.fc1(x))
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc3(x))
    x=self.fc4(x)
    return x

torch.manual_seed(41)
model=FaceMask()
model

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

import time
start_time=time.time()

train_losses=[]
test_losses=[]
train_correct=[]
test_correct=[]

epochs=5
log_interval=100
for epoch in range(epochs):
  trn_corr=0
  tst_corr=0

  for b,(x_train,y_train) in enumerate(train_loader):
    b+=1
    y_pred=model(x_train)
    loss=criterion(y_pred,y_train)

    predicted=torch.max(y_pred.data,1)[1]
    batch_corr=(predicted==y_train).sum()
    trn_corr+=batch_corr

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if b%log_interval==0:
      print(f'epoch:{epoch} batch_id:{b} loss:{loss.item()}')

  train_losses.append(loss)
  train_correct.append(trn_corr)

  with torch.no_grad():
    for b,(x_test,y_test) in enumerate(test_loader):
      y_val=model(x_test)
      predicted=torch.max(y_val.data,1)[1]
      tst_corr+=(predicted==y_test).sum()

  loss=criterion(y_val,y_test)
  test_losses.append(loss)
  test_correct.append(tst_corr.item())

current_time=time.time()
total = current_time-start_time
print(f'Training took:{total/60} minutes')

train_losses = [tl.item() if isinstance(tl, torch.Tensor) else tl for tl in train_losses]
test_losses = [tl.item() if isinstance(tl, torch.Tensor) else tl for tl in test_losses]

plt.plot(train_losses,label='Training Loss')
plt.plot(test_losses,label='Testing Loss')
plt.title('Training and Testing Loss')
plt.legend()



