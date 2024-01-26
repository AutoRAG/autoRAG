"""
Microsoft Azure AI Document Intelligence Output Processor
Process all the azure-preanalyzed files from data directory.
"""

from ..utils.json import JsonFileLoader
from .paragraph import AzureParagraphProcessor

class AzureOutputProcessor:
    """
    Initializes the AzureOutputProcessor with a specified data directory.

    This constructor loads all files from the given directory using JsonFileLoader
    and prepares to process them into document objects.

    :param data_dir: The directory containing JSON files to be processed.
    """

    def __init__(self, data_dir: str = None) -> None:
        # Load all files from the specified directory
        self.all_files = JsonFileLoader(data_dir).load()
        # Process the loaded files into documents
        # self.documents = self.all_files
        self.documents = self.get_documents()

    def get_documents(self) -> list:
        documents = []
        # Process the loaded files and return the count
        for file_name, file_content in self.all_files.items():
            paragraphs_list = file_content.get("paragraphs", [])
            table_content_list = file_content.get("tables", [])

            if paragraphs_list:
                paragraphs_documents = AzureParagraphProcessor(
                    paragraphs_list, file_name).documents
                print("paragraphs_documents", len(paragraphs_documents))
                documents += paragraphs_documents

            if table_content_list:
                table_content_documents = []
                documents += table_content_documents

        return documents
    

