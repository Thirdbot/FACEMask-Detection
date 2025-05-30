from Train import Trainer
import kagglehub
from pathlib import Path

Home_dir = Path(__file__).parent.absolute()
# Download latest version "pranavsingaraju/facemask-detection-dataset-20000-images"
dataset_path = Home_dir / "cleaned_dataset" / "data"

trainer = Trainer(path=dataset_path)

###model list 
model_list = ["DeepLearning","DecisionClass","KNNClass","RFC"]
model_name = "DeepLearning"

#using with single train
trainer.create_model(model_name)
save_best = trainer.train(model_name)
#using without declare creating model
# save_best = trainer.train_all()

print("save_best",save_best)

