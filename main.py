from src.cnnclassifier import logger
from cnnclassifier.pipeline.stage_1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.cnnclassifier.pipeline.stage_2_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
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

STAGE_NAME = "Prepare Base Model Stage"

if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e
