import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

class DepthEstimationDataset(Dataset):
    def __init__(self, data_dir):
        self.front_view_dir = os.path.join(data_dir, 'front-view')
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.front_images = os.listdir(self.front_view_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.front_images)

    def __getitem__(self, idx):
        front_img_name = self.front_images[idx]
        depth_img_name = front_img_name 

        front_img_path = os.path.join(self.front_view_dir, front_img_name)
        depth_img_path = os.path.join(self.depth_dir, depth_img_name)

        front_img = Image.open(front_img_path).convert('RGB')
        depth_img = Image.open(depth_img_path).convert('L')

        front_img = self.transform(front_img)
        depth_img = self.transform(depth_img)

        return front_img, depth_img

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x
    

if __name__ == "__main__":

    train_data = DepthEstimationDataset('path/to/train_data_folder')
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    val_data = DepthEstimationDataset('path/to/validation_data_folder')
    val_loader = DataLoader(val_data, batch_size=32)

    model = DepthEstimationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ### Train ###
    num_epochs = 10 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    ### Validation ###
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item() * inputs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Saving the model weights
    torch.save(model.state_dict(), 'depth_estimation_model.pth')
