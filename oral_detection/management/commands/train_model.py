"""
Management command:
    python manage.py train_model --dataset /path/to/dataset

Options:
    --dataset PATH    Root folder containing train/ and val/ subfolders
    --epochs  N       Max epochs (default 50)
    --batch   N       Batch size (default 32)
    --no-transfer     Use custom CNN instead of EfficientNetB3
    --output  PATH    Override save location for .h5 file
"""
import os
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings


class Command(BaseCommand):
    help = 'Train the ScanMyMouth AI oral cancer CNN model'

    def add_arguments(self, parser):
        parser.add_argument('--dataset',     required=True,  help='Dataset root path')
        parser.add_argument('--epochs',      type=int, default=50)
        parser.add_argument('--batch',       type=int, default=32)
        parser.add_argument('--no-transfer', action='store_true',
                            help='Use custom CNN instead of EfficientNetB3')
        parser.add_argument('--output',      default=None,
                            help='Override model save path (.h5)')

    def handle(self, *args, **opts):
        dataset_path  = opts['dataset']
        epochs        = opts['epochs']
        batch_size    = opts['batch']
        use_transfer  = not opts['no_transfer']
        save_path     = opts['output'] or str(settings.AI_MODEL_PATH)

        # Validate structure
        for split in ('train', 'val'):
            p = os.path.join(dataset_path, split)
            if not os.path.isdir(p):
                raise CommandError(
                    f"Missing directory: {p}\n"
                    "Expected:\n"
                    "  dataset/\n"
                    "    train/cancerous/  train/normal/\n"
                    "    val/cancerous/    val/normal/"
                )

        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("  ScanMyMouth AI — Model Training"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(f"  Dataset      : {dataset_path}")
        self.stdout.write(f"  Save path    : {save_path}")
        self.stdout.write(f"  Epochs       : {epochs}")
        self.stdout.write(f"  Batch size   : {batch_size}")
        self.stdout.write(f"  Architecture : {'EfficientNetB3 (transfer)' if use_transfer else 'Custom CNN'}")
        self.stdout.write(self.style.SUCCESS("=" * 60))

        try:
            from oral_detection.ai_engine import train_model
            history = train_model(
                dataset_path  = dataset_path,
                save_path     = save_path,
                epochs        = epochs,
                batch_size    = batch_size,
                use_transfer  = use_transfer,
            )

            val_acc = history.history.get('val_accuracy', [0])[-1]
            val_auc = history.history.get('val_auc',      [0])[-1]
            n_epochs = len(history.history['accuracy'])

            self.stdout.write(self.style.SUCCESS("\n✅  Training complete!"))
            self.stdout.write(f"   Epochs trained : {n_epochs}")
            self.stdout.write(f"   Val Accuracy   : {val_acc:.4f}")
            self.stdout.write(f"   Val AUC        : {val_auc:.4f}")
            self.stdout.write(f"   Model saved to : {save_path}")

        except ImportError:
            raise CommandError(
                "TensorFlow not installed.\n"
                "Run: pip install tensorflow opencv-python-headless pillow scikit-learn"
            )
        except Exception as exc:
            raise CommandError(f"Training failed: {exc}")
