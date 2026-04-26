import os
import zipfile
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str):
        """Abstract method to ingest data from a given file."""
        pass


# Implement a concrete class for ZIP Ingestion
class ZipDataIngestor(DataIngestor):
    def __init__(self, output_dir: str = "Dataset/Extracted"):
        """Initialize the ZipDataIngestor with an output directory."""
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def ingest(self, file_path: str) -> List[pd.DataFrame]:
        """
        Extracts CSV files from a .zip file and returns a list of DataFrames.
        Only extracts CSV files to the output directory.
        """
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        dataframes = []
        
        # Extract only CSV files from the zip
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            # Get list of all files in the zip
            file_list = zip_ref.namelist()
            
            # Filter for CSV files
            csv_files = [f for f in file_list if f.lower().endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the zip archive.")
            
            # Extract and read each CSV file
            for csv_file in csv_files:
                # Get the base filename without path
                base_name = os.path.basename(csv_file)
                output_path = os.path.join(self.output_dir, base_name)
                
                # Extract the file
                with zip_ref.open(csv_file) as source, open(output_path, 'wb') as target:
                    target.write(source.read())
                
                # Read the CSV into a DataFrame
                df = pd.read_csv(output_path)
                dataframes.append(df)
                
                print(f"Extracted and processed: {base_name}")

        return dataframes


# Implement a concrete class for CSV Ingestion
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".csv"):
            raise ValueError("The provided file is not a .csv file.")
        return pd.read_csv(file_path)


# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension == ".csv":
            return CSVDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")


# Example usage:
if __name__ == "__main__":
    # Specify the file path
    file_path = "Dataset/Provided/1a513221-3-dataset.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a list of DataFrames
    dataframes = data_ingestor.ingest(file_path)

    # Process each DataFrame
    for i, df in enumerate(dataframes if isinstance(dataframes, list) else [dataframes]):
        print(f"\nDataFrame {i+1}:")
        print(df.head())  # Display the first few rows of each DataFrame
    pass
