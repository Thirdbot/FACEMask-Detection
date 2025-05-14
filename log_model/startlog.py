import wandb

config = dict(project="mask_detection", 
              name="mask_detection_v1",
                epochs=10,
                batch_size=32,
                validation_split=0.2,
                shuffle=True,
                # callbacks=[]
             )

# wandb.init(config=config)



class LogModel:
    def __init__(self,dataset_config):
        self._login()
        self.model_config = None
        self.dataset_config = dataset_config
        self.raw_dataset = None
        self.preprocessed_dataset = None
    def _login(self):
        wandb.login()
    
    def raw_data_and_log(self,raw_dataset):
        self.raw_dataset = raw_dataset
        #set up artifact
        with wandb.init(project="dataset_artifact",job_type="load-data") as run:
            names = self.dataset_config
            
            #create artifact placeholder
            raw_data = wandb.Artifact(name="raw_dataset",type="dataset",
                                    description="dataset get from local directory.",
                                    metadata={
                                        "source":"local directory",
                                        "size":[len(dataset) for dataset in raw_dataset]
                                    })
            #write data to artifact
            run.log_artifact(raw_data)
    def preprocessed_data_and_log(self,preprocessed_dataset,step):
        self.preprocessed_dataset = preprocessed_dataset
        #set up artifact
        with wandb.init(project="dataset_artifact",job_type="preprocess-data") as run:
            names = self.dataset_config
            processed_data = wandb.Artifact(name="preprocessed_dataset",type="dataset",
                                  description="preprocessed dataset get from local directory.",
                                  metadata=step)
            #declare use raw artifact
            run.use_artifact('raw_dataset:latest')
            #write data to artifact
            run.log_artifact(processed_data)
            
            








