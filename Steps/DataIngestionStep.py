import pandas as pd
import os
import logging

from Src.IngestData import DataIngestorFactory
from zenml import step

# Configure logging
logging.basicConfig(
    filename="Steps/step.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """Ingest data from a file using the appropriate DataIngestor."""
    try:
        logging.info(f"DataIngestionStep: Starting ingestion for file: {file_path}")
        file_extension = os.path.splitext(file_path)[1]
        logging.info(f"DataIngestionStep: File extension detected: {file_extension}")
        data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
        df = data_ingestor.ingest(file_path)
        if isinstance(df, list):
            if not df:
                logging.error("DataIngestionStep: No DataFrames found in the provided file.")
                raise ValueError("No DataFrames found in the provided file.")
            df = df[0]
        if df is None:
            logging.error("DataIngestionStep: Data ingestion failed: no DataFrame returned.")
            raise ValueError("Data ingestion failed: no DataFrame returned.")
        logging.info("DataIngestionStep: Data ingestion successful.")
        return df
    except Exception as e:
        logging.error(f"DataIngestionStep: Error during ingestion: {e}")
        raise
