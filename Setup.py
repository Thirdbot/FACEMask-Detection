from Train import Trainer
import kagglehub
from pathlib import Path

from backend.server import run_server

Home_dir = Path(__file__).parent.absolute()
# Download latest version "pranavsingaraju/facemask-detection-dataset-20000-images"
dataset_path = Home_dir / "cleaned_dataset" / "data"

trainer = Trainer(path=dataset_path)
save_best = trainer.train_all()
print("save_best",save_best)

