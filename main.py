"""LamSonOCR - Main CLI entry point."""
import argparse
from pathlib import Path
import os
import sys

# Ensure the project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set fallback for MPS unsupported ops like CTC loss
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def cmd_generate(args):
    """Generate synthetic training data."""
    from scripts.generate_synthetic import generate_synthetic_dataset
    from config.charset import Charset

    cs = Charset()
    output_dir = Path(args.output)
    generate_synthetic_dataset(
        output_dir=output_dir,
        num_samples=args.num_samples,
        charset=cs,
        fonts_dir=args.fonts_dir,
    )
    print(f"✅ Generated {args.num_samples} samples in {output_dir}")


def cmd_train(args):
    """Train the CRNN model."""
    from config.charset import Charset
    from config.settings import Settings
    from data.dataset import OCRDataset
    from data.etl_binary_dataset import ETLBinaryDataset
    from data.augmentation import OCRAugmentation
    from training.train import Trainer
    from torch.utils.data import ConcatDataset

    cs = Charset()
    settings = Settings(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    if args.checkpoint_dir:
        settings.model_dir = Path(args.checkpoint_dir)

    train_ds = OCRDataset(
        root_dir=Path(args.train_data),
        charset=cs,
        transform=OCRAugmentation() if not args.no_augment else None,
    )
    
    if args.etl_dir:
        etl_ds = ETLBinaryDataset(
            etl_dir=args.etl_dir,
            charset=cs,
            target_height=settings.img_height,
            transform=OCRAugmentation() if not args.no_augment else None,
        )
        if len(etl_ds) > 0:
            train_ds = ConcatDataset([train_ds, etl_ds])

    val_ds = None
    if args.val_data:
        val_ds = OCRDataset(root_dir=Path(args.val_data), charset=cs)

    print(f"📊 Training samples: {len(train_ds)}")
    if val_ds:
        print(f"📊 Validation samples: {len(val_ds)}")
    print(f"🏋️ Device: {settings.device}")
    print(f"🔤 Character classes: {cs.num_classes}")

    trainer = Trainer(charset=cs, settings=settings)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    history = trainer.train(train_ds, val_ds)

    print(f"\n✅ Training complete!")
    print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
    if history.get('val_loss'):
        print(f"   Final val loss: {history['val_loss'][-1]:.4f}")


def cmd_evaluate(args):
    """Evaluate model on a dataset."""
    import torch
    from config.charset import Charset
    from data.dataset import OCRDataset
    from model.crnn import CRNN
    from benchmark.evaluate import Evaluator

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vocab = checkpoint.get("vocab")
    cs = Charset(vocab=vocab)

    model = CRNN(num_classes=checkpoint.get("charset_size", cs.num_classes))
    
    # Strip _orig_mod. prefix from checkpoint if present (caused by torch.compile on RunPod)
    state_dict = checkpoint["model_state_dict"]
    has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if has_prefix:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)

    dataset = OCRDataset(root_dir=Path(args.data), charset=cs)

    evaluator = Evaluator(model=model, charset=cs, device=args.device)
    results = evaluator.evaluate(dataset)

    print(f"\n📋 Benchmark Results")
    print(f"   Samples: {results['num_samples']}")
    print(f"   CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"   WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"   Seq Accuracy: {results['sequence_accuracy']:.4f} ({results['sequence_accuracy']*100:.2f}%)")

    if args.report:
        evaluator.save_report(results, Path(args.report))


def cmd_predict(args):
    """Predict text from images."""
    from inference.predict import OCRPredictor
    from pathlib import Path

    predictor = OCRPredictor(
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
    )

    images = args.images
    for img_path in images:
        result = predictor.predict(img_path)
        print(f"📝 {img_path}: '{result['text']}' (conf: {result['confidence']:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="LamSonOCR - CRNN-based text line OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate
    gen = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen.add_argument("--output", type=str, default="data/train", help="Output directory")
    gen.add_argument("--num-samples", type=int, default=50000, help="Number of samples")
    gen.add_argument("--fonts-dir", type=str, default=None, help="Fonts directory")

    # Train
    trn = subparsers.add_parser("train", help="Train the model")
    trn.add_argument("--train-data", type=str, required=True, help="Training data dir")
    trn.add_argument("--val-data", type=str, default=None, help="Validation data dir")
    trn.add_argument("--batch-size", type=int, default=64)
    trn.add_argument("--epochs", type=int, default=50)
    trn.add_argument("--lr", type=float, default=0.001)
    trn.add_argument("--checkpoint-dir", type=str, default=None)
    trn.add_argument("--etl-dir", type=str, default=None, help="Directory containing raw ETL binary files")
    trn.add_argument("--resume", type=str, default=None, help="Path to checkpoint file (.pt) to resume training from")
    trn.add_argument("--no-augment", action="store_true")

    # Evaluate
    evl = subparsers.add_parser("evaluate", help="Evaluate model")
    evl.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    evl.add_argument("--data", type=str, required=True, help="Test data dir")
    evl.add_argument("--device", type=str, default="cpu")
    evl.add_argument("--report", type=str, default=None, help="Save report to file")

    # Predict
    prd = subparsers.add_parser("predict", help="Predict text from images")
    prd.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    prd.add_argument("--device", type=str, default="cpu")
    prd.add_argument("images", nargs="+", help="Image files to predict")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
