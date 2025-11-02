from e2eMLOpsDSMLFlow.logger import get_logger
from e2eMLOpsDSMLFlow.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from e2eMLOpsDSMLFlow.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from e2eMLOpsDSMLFlow.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from e2eMLOpsDSMLFlow.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from e2eMLOpsDSMLFlow.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

logger = get_logger(__name__)

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    ingestion_pipeline = DataIngestionTrainingPipeline()
    ingestion_pipeline.main()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 2: Data Validation
STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    validation_pipeline = DataValidationTrainingPipeline()
    validation_pipeline.main()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 3: Data Transformation
STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    transformation_pipeline = DataTransformationTrainingPipeline()
    transformation_pipeline.main()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 4: Model Training
STAGE_NAME = "Model Trainer Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    trainer_pipeline = ModelTrainerTrainingPipeline()
    trainer_pipeline.main()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Stage 5: Model Evaluation
STAGE_NAME = "Model Evaluation Stage"
try:
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    evaluation_pipeline = ModelEvaluationTrainingPipeline()
    evaluation_pipeline.main()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
