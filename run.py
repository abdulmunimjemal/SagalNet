import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Spoken Digit Recognition CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'deeper'], help='Model architecture')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict a digit from an audio file')
    predict_parser.add_argument('file_path', type=str, help='Path to audio file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from src.models.train_model import train
        train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, model_type=args.model_type)
        
    elif args.command == 'predict':
        from src.models.predict_model import predict
        predict(args.file_path)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
