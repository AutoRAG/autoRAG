"""
Microsoft Azure AI Document Intelligence Output Processor
Process all the azure-preanalyzed files from data directory.
"""

from ..utils.json import JsonFileLoader
from .paragraph import AzureParagraphProcessor
from .table import AzureTablesProcessor


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
        self.nodes = self.get_nodes()

    def get_nodes(self) -> list:
        nodes = []

        # Process the loaded files and return the count
        for file_name, file_content in self.all_files.items():
            paragraphs_list = file_content.get("paragraphs", [])
            tables_list = file_content.get("tables", [])

            if paragraphs_list:
                paragraphs_nodes = AzureParagraphProcessor(
                    paragraphs_list, file_name
                ).nodes
                nodes += paragraphs_nodes

            if tables_list:
                table_content_nodes = AzureTablesProcessor(tables_list, file_name).nodes
                nodes += table_content_nodes

        return nodes
