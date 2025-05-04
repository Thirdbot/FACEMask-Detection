from kagglehub import dataset_download

class DatasetLoader:
    def __init__(self, datasetName):
        self.datasetName = datasetName

    def load_data(self):
        datasets = dataset_download(self.datasetName)
        return datasets
