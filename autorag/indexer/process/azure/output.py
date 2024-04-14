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
    :param file_type: The type of the processed files.
    :param sentence_splitter_args: Arguments for the SentenceSplitter function.
                                   This can include arguments like chunk_size,
                                   chunk_overlap, or any other arguments that
                                   SentenceSplitter expects.
    :param table_cfg: The configuration for processing tables.
    # :param include_table: whether to include the tables from the files
    # :param table_by_token: whether to process tables by row or by table df
    # :param token_limit: the maximum number of tokens to process in a table
    """

    def __init__(
        self,
        data_dir: str = None,
        file_type: str = None,
        sentence_splitter_args: dict = {},
        table_process_cfg: dict = {},
    ) -> None:
        # Load all files from the specified directory
        self.all_files = JsonFileLoader(data_dir).load()
        self.file_type = file_type
        # Process the loaded files into nodes
        self.sentence_splitter_args = sentence_splitter_args
        # Table processing configuration
        self.include_table = table_process_cfg.include_table
        self.by_token = table_process_cfg.by_token
        self.token_limit = table_process_cfg.token_limit
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
                    paragraphs_list,
                    file_name,
                    self.file_type,
                    self.sentence_splitter_args,
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
