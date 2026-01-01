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
from src.models.model import SimpleCNN, DeeperCNN
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import numpy as np

def train(epochs=30, batch_size=32, learning_rate=0.001, data_dir='data/processed', model_type='simple', model_save_path='models/best_model.pth', time_mask=30, freq_mask=15):
    
    # Start MLflow run
    mlflow.set_experiment("Spoken Digit Recognition")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("time_mask", time_mask)
        mlflow.log_param("freq_mask", freq_mask)
        
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
        # Pass augmentation params
        train_dataset = SpokenDigitDataset(file_list=train_files, train=True, time_mask_param=time_mask, freq_mask_param=freq_mask)
        test_dataset = SpokenDigitDataset(file_list=test_files, train=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Model Selection
        if model_type == 'deeper':
            model = DeeperCNN(num_classes=10).to(device)
            print("Using DeeperCNN architecture.")
        else:
            model = SimpleCNN(num_classes=10).to(device)
            print("Using SimpleCNN architecture.")
        
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        best_f1 = 0.0 # optimizing for F1 now as per request for comparable metrics
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
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
                loop.set_postfix(loss=loss.item())
                
            avg_train_loss = train_loss/len(train_loader)
            
            # Validation & Metrics
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate Sklearn Metrics
            val_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            # Step Scheduler
            scheduler.step(val_acc) # Scheduler usually monitors accuracy or loss, keeping acc is fine
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric("val_precision", precision, step=epoch)
            mlflow.log_metric("val_recall", recall, step=epoch)
            mlflow.log_metric("val_f1", f1, step=epoch)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.2f}% - F1: {f1:.4f}")
            
            # Checkpointing
            # Save Best Model (using F1 as primary metric for improvements)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved best model with F1: {best_f1:.4f}")
                
                # Log model to MLflow
                # Log model to MLflow
                # Create input example and signature
                example_input = images[:1].cpu().numpy()
                signature = infer_signature(example_input, model(images[:1]).detach().cpu().numpy())
                
                mlflow.pytorch.log_model(
                    model, 
                    "model", 
                    signature=signature, 
                    input_example=example_input
                )
        
        print("Training finished.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'deeper'])
    parser.add_argument('--time_mask', type=int, default=30, help='Time masking param for SpecAugment')
    parser.add_argument('--freq_mask', type=int, default=15, help='Frequency masking param for SpecAugment')
    args = parser.parse_args()
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, data_dir=args.data_dir, model_type=args.model_type, time_mask=args.time_mask, freq_mask=args.freq_mask)
