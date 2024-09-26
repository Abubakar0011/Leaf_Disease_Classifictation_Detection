from src.cnnclassifier import logger
from cnnclassifier.pipeline.stage_1_data_ingestion import (
    DataIngestionTrainingPipeline
)


STAGE_NAME = "Data Ingestion Stage"

if __name__ == '__main__':
    try:
        logger.info(f">>> stage {STAGE_NAME} started <<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<< \n\n x=====x")

    except Exception as e:
        logger.exception(e)
