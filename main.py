from src.cnnclassifier import logger
from cnnclassifier.pipeline.stage_1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.cnnclassifier.pipeline.stage_2_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from cnnclassifier.pipeline.stage_3_model_training import (
    ModelTrainingPipeline
)
from cnnclassifier.pipeline.stage_4_model_evaluation import (
    EvaluationPipeline
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

STAGE_NAME = "Training"

if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Evaluation stage"

if __name__ == '__main__':
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_evalution = EvaluationPipeline()
        model_evalution.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")

    except Exception as e:
        logger.exception(e)
        raise e
