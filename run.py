import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Spoken Digit Recognition CLI")
    subparsers = parser.add_subparsers(
        dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--model_type', type=str, default='simple',
                              choices=['simple', 'deeper'], help='Model architecture')
    train_parser.add_argument('--time_mask', type=int,
                              default=30, help='Time masking param')
    train_parser.add_argument('--freq_mask', type=int,
                              default=15, help='Frequency masking param')

    # Predict command
    predict_parser = subparsers.add_parser(
        'predict', help='Predict a digit from an audio file')
    predict_parser.add_argument(
        'file_path', type=str, help='Path to audio file')

    # Preprocess features command
    prep_parser = subparsers.add_parser(
        'preprocess_features', help='Precompute Mel-spectrogram features to disk')
    prep_parser.add_argument(
        '--input_dir', type=str, default='data/processed', help='Input labeled audio directory')
    prep_parser.add_argument(
        '--output_dir', type=str, default='data/interim/features', help='Output directory for .pt features')
    prep_parser.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing feature files')
    prep_parser.add_argument('--sample_rate', type=int, default=16000)
    prep_parser.add_argument('--n_mels', type=int, default=64)
    prep_parser.add_argument('--max_duration', type=float, default=1.0)
    prep_parser.add_argument('--n_fft', type=int, default=1024)
    prep_parser.add_argument('--hop_length', type=int, default=512)

    args = parser.parse_args()

    if args.command == 'train':
        from src.models.train_model import train
        train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
              model_type=args.model_type, time_mask=args.time_mask, freq_mask=args.freq_mask)

    elif args.command == 'predict':
        from src.models.predict_model import predict
        predict(args.file_path)

    elif args.command == 'preprocess_features':
        from src.data.preprocess_features import preprocess_directory
        from src.data.audio_features import AudioFeatureConfig

        cfg = AudioFeatureConfig(
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            max_duration=args.max_duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        count = preprocess_directory(
            args.input_dir, args.output_dir, cfg=cfg, overwrite=args.overwrite)
        print(f"Wrote {count} feature files to {args.output_dir}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
