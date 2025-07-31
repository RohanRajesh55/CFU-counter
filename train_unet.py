import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# --- Configuration ---
DATASET_DIR = 'dataset'
# Assuming simple text files listing image names
TRAIN_LIST = os.path.join(DATASET_DIR, 'train.txt')
VAL_LIST = os.path.join(DATASET_DIR, 'val.txt')
MODEL_SAVE_PATH = 'unet_model.pth'
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 4
IMAGE_SIZE = (128, 128)
# For binary segmentation (background/foreground)
NUM_CLASSES = 1
# ---------------------

## 🛠️ U-Net Building Blocks

class ConvBlock(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then a ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then a ConvBlock"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Use bilinear upsampling or a transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the upsampling path, x2 is the skip connection from the downsampling path
        x1 = self.up(x1)
        # Pad x1 to match the size of the skip connection tensor x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to map to the desired number of output classes"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

---

## 🏗️ The U-Net Model

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder (Downsampling path)
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (Upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Follow the U-shape
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

---

## 📦 Dataset and DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list_path, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        with open(file_list_path, 'r') as f:
            self.file_names = [line.strip() for line in f]
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Assumes mask has the same filename

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale for a single-channel mask

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

---

## 🚀 Training and Execution

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs):
    print("🚀 Starting training...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")

    print("✅ Training finished.")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    # Utility to create a dummy dataset for demonstration purposes
    def setup_dummy_data():
        print("Setting up dummy dataset for demonstration...")
        image_dir = os.path.join(DATASET_DIR, 'images')
        mask_dir = os.path.join(DATASET_DIR, 'masks')
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(10):  # Create 10 dummy images/masks
            img = Image.new('RGB', IMAGE_SIZE, color=(i * 20, 100, 255 - i * 20))
            mask = Image.new('L', IMAGE_SIZE, color=(255 if i % 2 == 0 else 0))
            img_name = f'sample_{i}.png'
            img.save(os.path.join(image_dir, img_name))
            mask.save(os.path.join(mask_dir, img_name))

        with open(TRAIN_LIST, 'w') as f:
            f.writelines([f'sample_{i}.png\n' for i in range(8)])
        with open(VAL_LIST, 'w') as f:
            f.writelines([f'sample_{i}.png\n' for i in range(8, 10)])
        print("Dummy data created successfully.")

    setup_dummy_data()

    # Define transforms for images and masks
    image_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    # Create Datasets and DataLoaders
    train_dataset = SegmentationDataset(
        image_dir=os.path.join(DATASET_DIR, 'images'),
        mask_dir=os.path.join(DATASET_DIR, 'masks'),
        file_list_path=TRAIN_LIST,
        transform=image_transform,
        mask_transform=mask_transform
    )
    val_dataset = SegmentationDataset(
        image_dir=os.path.join(DATASET_DIR, 'images'),
        mask_dir=os.path.join(DATASET_DIR, 'masks'),
        file_list_path=VAL_LIST,
        transform=image_transform,
        mask_transform=mask_transform
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True).to(device)
    
    # This loss combines a Sigmoid layer and the BCELoss in one single class.
    # It is more numerically stable than using a plain Sigmoid followed by a BCELoss.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the training process
    train_model(model, device, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)