from Train import Trainer
import kagglehub
from pathlib import Path

def run_train():
    Home_dir = Path(__file__).parent.absolute()
    # Download latest version "pranavsingaraju/facemask-detection-dataset-20000-images"
    dataset_path = Home_dir / "cleaned_dataset" / "data"

    trainer = Trainer(path=dataset_path)
    
    #กำหนดอบเทรน config (sweep model)
    trainer.runtime = 3

    ###model list  ["DeepLearning","DecisionClass","KNNClass","RFC"]
    model_name = "DeepLearning"

    #เทรนโมเดล 1 ตัว ให้สร้างโมเดลเเละเทรน
    trainer.create_model(model_name)
    save_best = trainer.train(model_name)
    
    #เทรนโมเดลทั้งหมด ไม่ต้องสร้างโมเดล
    # save_best = trainer.train_all()

    #ดูผลการเทรน
    print("save_best",save_best)

if __name__ == "__main__":
    run_train()


