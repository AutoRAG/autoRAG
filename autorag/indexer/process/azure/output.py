"""
Microsoft Azure AI Document Intelligence Output Processor
Process all the azure-preanalyzed files from data directory.
"""

from ..utils.json import JsonFileLoader
from .paragraph import AzureParagraphProcessor, AzurePolygonParagraphProcessor
from .table import AzureTablesProcessor


class AzureOutputProcessor:
    """
    Initializes the AzureOutputProcessor with a specified data directory.

    This processor loads all files from the given directory using JsonFileLoader
    and process them into TextNode objects.

    :param data_dir: The directory containing JSON files to be processed.
    :param file_type: The type of the processed files.
    :param paragraph_process_cfg: Configuration for paragraph processing.
                                If polygon_group=True, uses polygon tracking with chunk_size.
                                Otherwise uses sentence splitter with sentence_splitter_cfg.
    :param table_process_cfg: The configuration for processing tables.
    """

    def __init__(
        self,
        data_dir: str = None,
        file_type: str = None,
        paragraph_process_cfg: dict = {},
        table_process_cfg: dict = {},
        sentence_splitter_cfg: dict = {},
    ) -> None:
        # Load all files from the specified directory
        self.all_files = JsonFileLoader(data_dir).load()
        self.file_type = file_type

        # Paragraph processing configuration
        self.polygon_group = paragraph_process_cfg.get("polygon_group", False)
        self.sentence_splitter_cfg = sentence_splitter_cfg
        # Table processing configuration
        self.include_table = table_process_cfg.get("include_table", False)
        self.by_token = table_process_cfg.get("by_token", False)
        self.token_limit = table_process_cfg.get("token_limit", None)

        self.nodes = self.get_nodes()

    def get_nodes(self) -> list:
        nodes = []

        # Process the loaded files
        for file_name, file_content in self.all_files.items():

            tables_list = file_content.get("tables", [])

            # Process paragraph data
            if self.polygon_group:
                pages_list = file_content.get("pages", [])
                paragraphs_nodes = AzurePolygonParagraphProcessor(
                    pages_list,
                    file_name,
                    self.file_type,
                    self.sentence_splitter_cfg,
                ).nodes
            else:
                paragraphs_list = file_content.get("paragraphs", [])
                paragraphs_nodes = AzureParagraphProcessor(
                    paragraphs_list,
                    file_name,
                    self.file_type,
                    self.sentence_splitter_cfg,
                ).nodes
            nodes += paragraphs_nodes

            # Process table data
            if self.include_table and tables_list:
                table_content_nodes = AzureTablesProcessor(
                    tables_list,
                    file_name,
                    self.file_type,
                    self.by_token,
                    self.token_limit,
                ).nodes
                nodes += table_content_nodes

        return nodes
