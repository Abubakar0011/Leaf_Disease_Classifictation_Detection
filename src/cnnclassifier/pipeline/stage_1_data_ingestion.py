from cnnclassifier.config.configuration import ConfigurationManager
from cnnclassifier.components.data_ingestion import DataIngestion
from cnnclassifier import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.downloading_file()
        data_ingestion.extratct_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>> stage {STAGE_NAME} started <<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<< \n\n x=====x")

    except Exception as e:
        logger.exception(e)
