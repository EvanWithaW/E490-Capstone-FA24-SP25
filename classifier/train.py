import os
import glob
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from CharacterModel import CharacterModel
from CharacterDataset import CharacterDataset


images_folder = ""
image_paths = glob.glob(os.path.join(images_folder, "*"))
image_size = 32
transforms = A.Compose([
    # A.Affine(shear={"x":15}, p=1.0),
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    ),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255,
    ),
    ToTensorV2()
])

lr = 0.001
batch_size = 256
num_workers = 64
num_epochs = 75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_dataset = CharacterDataset(image_paths, transforms=transforms)
train_dataloader = DataLoader(char_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
model = CharacterModel()
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
# lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(num_epochs):  # loop over the dataset multiple times
    if epoch % 10 == 0:
        PATH = f"./weights/train-{epoch}.pth"
        torch.save(model.state_dict(), PATH)

    running_loss = 0.0
    for i, batch in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'[{epoch + 1}, {i+1}] loss: {running_loss / 100:.5f}')
            running_loss = 0.0

print('Finished Training')
print()

PATH = './weights/train-last.pth'
torch.save(model.state_dict(), PATH)




