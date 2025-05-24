from Train import Trainer
import kagglehub
from pathlib import Path

Home_dir = Path(__file__).parent.absolute()
# Download latest version "pranavsingaraju/facemask-detection-dataset-20000-images"
dataset_path = Home_dir / "archive"

trainer = Trainer(path=dataset_path)
trainer.train_all()

