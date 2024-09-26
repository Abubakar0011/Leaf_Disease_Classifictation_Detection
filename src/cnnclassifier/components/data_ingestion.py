import os
import zipfile
import gdown
from cnnclassifier import logger
# from cnnclassifier.utils.common import get_size
from cnnclassifier.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def downloading_file(self) -> str:
        '''it will fetch data from the url'''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artfacts/data_ingestion", exist_ok=True)
            logger.info(
                f"Download data from {dataset_url} into {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)

        except Exception as e:
            raise e
        
    def extratct_zip_file(self):
        '''Extracting the zip file into the data directory'''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
