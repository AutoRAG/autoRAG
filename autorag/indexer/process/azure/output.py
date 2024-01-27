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

    This processor loads all files from the given directory using JsonFileLoader
    and process them into TextNode objects.

    :param data_dir: The directory containing JSON files to be processed.
    """

    def __init__(self, data_dir: str = None) -> None:
        # Load all files from the specified directory
        self.all_files = JsonFileLoader(data_dir).load()
        # Process the loaded files into nodes
        self.nodes = self.get_nodes()

    def get_nodes(self) -> list:
        nodes = []

        # Process the loaded files
        for file_name, file_content in self.all_files.items():
            paragraphs_list = file_content.get("paragraphs", [])
            tables_list = file_content.get("tables", [])
            # Process paragraph data
            if paragraphs_list:
                paragraphs_nodes = AzureParagraphProcessor(
                    paragraphs_list, file_name
                ).nodes
                nodes += paragraphs_nodes
            # Process table data
            if tables_list:
                table_content_nodes = AzureTablesProcessor(tables_list, file_name).nodes
                nodes += table_content_nodes

        return nodes
