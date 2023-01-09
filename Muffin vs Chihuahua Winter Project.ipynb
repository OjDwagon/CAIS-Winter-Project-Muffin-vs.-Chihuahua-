import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os # For working with file directories
from torchvision.io import read_image # For converting .jpg to tensors
from torchvision import models
import torchvision.transforms as T # For image preprocessing

# use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if(torch.cuda.is_available()):
  print("GPU is in use.")
else:
  print("GPU is NOT in use.")

# Mount the drive
# Giving Colab access to the Drive
from google.colab import drive
drive.mount('/content/drive/')

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html used as a reference
# https://sparrow.dev/torchvision-transforms/ used as a reference for image preprocessing
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html referenced for the testModel function

# Make a dataset class to store the data
# Expects a data directory containing two folders, one named "chihuahua" and the other "muffin"
# Labels chihuahuas as 0 and muffins as 1
class myDataset(Dataset):
  def __init__(self, imageDirectory, transform = None, targetTransform = None):
    self._imageDirectory = imageDirectory
    self._numChihuahuas = 0

    # Count how many chihuahua pictures there are (referenced: https://pynative.com/python-count-number-of-files-in-a-directory/)
    for path in os.listdir(self._imageDirectory + "/chihuahua"):
      self._numChihuahuas += 1
    
    # Add the number of muffins for the total number of pics in this dataset
    self._len = self._numChihuahuas
    for path in os.listdir(self._imageDirectory + "/muffin"):
      self._len += 1
    
    self._paths = [0] * self._len # Store all the file paths so that they are indexed for easy retrieval
    for index, path in enumerate(os.listdir(self._imageDirectory + "/chihuahua")): # Store all the chihuahuas
      self._paths[index] = self._imageDirectory + "/chihuahua/" + path
    for index, path in enumerate(os.listdir(self._imageDirectory + "/muffin")):
      self._paths[index + self._numChihuahuas] = self._imageDirectory + "/muffin/" + path
    
    
  def __len__(self):
    return self._len

  def __getitem__(self, idx): # Return the preprocessed image (as a tensor) and its label
    if(idx < self._numChihuahuas):
      label = 0
    else:
      label = 1
    label = torch.tensor([label]).to(device) # Move the labels to the device so everything is on the same device
    image = read_image(self._paths[idx])
    image = image.to(device) # Move the tensor to the right device (hopefully GPU)
    
    # Deal with grayscale images
    if(image.shape[0] == 1):
      newImage = torch.ones([3, image.shape[1], image.shape[2]], dtype=torch.float32, device=device)
      newImage[0] = image
      newImage[1] = image
      newImage[2] = image
      image = newImage
      
    preprocess = T.Compose([T.Resize((224, 224)), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = preprocess(image.type(torch.float32))

    return image, label
  
  def __getpath__(self, idx):
    return self._paths[idx]

# -----------------------------------------------------------------------------------------------------------------------

# Code to train the model
def trainModel(model, optimzer, criterion, EPOCHS):
  for e in range(EPOCHS):
    print(f"Beginning epoch {e}/{EPOCHS}")
    for i, data in enumerate(trainLoader):
      inputs, labels = data
      labels = torch.reshape(labels, (labels.shape[0],)) # To remove a dimension
      optimizer.zero_grad() # Zero the gradient before calculations

      outputs = model(inputs) # Feed forward
      loss = criterion(outputs, labels) # Check answers
      loss.backward() # Update the gradients of each tensor
      optimizer.step() # Update each weight

      if(i % 5 == 0):
        print(f"Just finished mini-batch {i} in epoch {e}.")

  print("Finished!!!")

# -----------------------------------------------------------------------------------------------------------------------

# Code to test the model
def testModel(model, dataLoader, lossFunction):
  with torch.no_grad(): # no_grad because we aren't training
    testLoss, correct = 0, 0

    for data in dataLoader:
      images, labels = data
      predictions = model(images)

      for i in range(predictions.shape[0]): # Go through all predictions
        prediction = -1
        if(predictions[i][0] > predictions[i][1]): # Calculate the prediction as 0 or 1 (whichever is higher)
          prediction = 1
        else:
          prediction = 0
        if(prediction == labels[i][0]): # Compare the prediction to determine if it was correct
          correct += 1

  accuracy = correct / len(dataLoader)
  print(f"Accuracy of the model is : {accuracy}%")

# --------------------------------------------------------------------------------------------------------------------------------------------------

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16

# Set up data in train and test loaders
trainDataset = myDataset("/content/drive/My Drive/CAIS++/Winter Project/train")
testDataset = myDataset("/content/drive/My Drive/CAIS++/Winter Project/test")
trainLoader = torch.utils.data.DataLoader(trainDataset, BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, BATCH_SIZE, shuffle = True)

# Set up the model (using transfer learning from RESNET18), referenced CAIS++ notebook
# Also set up the optimizer and loss function (criterion)

model = models.resnet18(weights=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Freeze every all layers (except fully connected) to use them as feature extractors
# Referenced https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6#:~:text=In%20PyTorch%20we%20can%20freeze,to%20apply%20a%20pretrained%20model.
# 'fc.weight', 'fc.bias' layers are the ones that shouldn't be frozen
for name, param in model.named_parameters():
  if("fc" in name):
    param.requires_grad = True
  else:
    param.requires_grad = False

# Save model to device (the GPU)
model = model.to(device)

criterion = nn.CrossEntropyLoss() # USe binary cross entropy because it is either a muffin or chihuahua
optimizer = optim.Adam(model.parameters()) # Use Adam for decaying momentum and give it the model's parameters so they will be updated

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Taken out of a CAIS notebook, decays learning rate in the optimizer

# -------------------------------------------------------------------------------------------------------------------------------------------------

# Trains and saves the model
# Referenced https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
trainModel(model, optimizer, criterion, EPOCHS)
torch.save(model, 'myModel.pth')

# -----------------------------------------------------------------------------------------------------------------------

# Loads and tests the model
studentModel = torch.load('myModel.pth')
testModel(studentModel, testLoader, criterion)
        
