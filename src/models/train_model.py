import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.dataset import SpokenDigitDataset
from src.models.model import SimpleCNN

def train(epochs=30, batch_size=32, learning_rate=0.001, data_dir='data/processed', model_save_path='models/best_model.pth'):
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load All Data Paths first
    # We use a temporary dataset instance just to crawl the directory
    temp_dataset = SpokenDigitDataset(data_dir)
    all_files = temp_dataset.file_list
    print(f"Total samples found: {len(all_files)}")
    
    if len(all_files) == 0:
        print("No data found! Check data/processed/")
        return
    
    # Split files into Train and Test
    # This allows us to apply augmentation ONLY on train_files
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42, stratify=[y for x, y in all_files])
    
    # Create Datasets
    train_dataset = SpokenDigitDataset(file_list=train_files, train=True)
    test_dataset = SpokenDigitDataset(file_list=test_files, train=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Step Scheduler
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with acc: {best_acc:.2f}%")
            
    print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, data_dir=args.data_dir)
