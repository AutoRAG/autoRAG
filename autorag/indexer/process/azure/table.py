from typing import List, Dict, Any, Optional
from llama_index.core.schema import TextNode
from nltk.tokenize import word_tokenize


UNWANTED_CONTENT: dict[str, Any] = {
    "+\n:selected:": "+",
    "-\n:unselected:": "-",
    "+\n:unselected:": "+",
    "-\n:unselected: :unselected:": "-",
}


class AzureTablesProcessor:
    """
    Iterates over a list of table data.

    :param azure_tables_list: The list of tables returned from Azure.
    :param file_name: The name of the file.
    :param file_type: The type of the file.
    :param by_token: Whether to process tables by token or by table row. if False, by row.
    :param token_limit: The maximum number of tokens in a single TextNode (usually limited by model)
    """

    def __init__(
        self,
        azure_tables_list: list[dict[str, Any]] = [],
        file_name: str = None,
        file_type: str = None,
        by_token: bool = True,
        token_limit: int = 3000,
    ) -> None:

        # Initialize the AzureTablesProcessor class.
        self.azure_tables_list = azure_tables_list
        self.file_name = file_name
        self.file_type = file_type
        self.by_token = by_token
        self.token_limit = token_limit
        self.table_dataframes, self.table_pages = self.get_table_dataframes()
        self.nodes = self.get_table_nodes()

    def get_table_dataframes(self) -> List[Dict[str, Any]]:
        """
        Processes a list of tables.

        :param table_list: A list of tables, each a dictionary.
        :return: A list of processed table dataframe.
        """
        table_dataframes = []
        table_pages = []
        last_table_headers = None
        last_table_name = ""
        last_table_col_count = 0

        for table in self.azure_tables_list:
            # Get the table data
            table_name = table.get("caption", {}).get("content", "Table")
            table_cells = table.get("cells", [])
            has_headers = any(
                cell.get("kind") == "columnHeader" for cell in table_cells
            )
            col_count = table.get("columnCount", 0)
            table_page = self.get_table_page_number(table)

            # Skip tables that are content or directory listings
            if not has_headers and not table_name:
                continue

            # If the current table lacks headers and matches the column count of the
            # previous table, assume it is a continuation of the previous table split
            # across pages.
            if (
                not has_headers
                and last_table_headers
                and col_count == last_table_col_count
            ):
                # Construct the table using last table headers and name
                single_table = SingleTableProcessor(
                    table_cells, last_table_name, last_table_headers
                )
                table_dataframes.append(single_table.table_dataframe)
                table_pages.append(table_page)
                continue

            # Process normally if the table has headers
            if has_headers:
                if (
                    not table_name
                    and last_table_name
                    and col_count == last_table_col_count
                ):
                    table_name = last_table_name

                single_table = SingleTableProcessor(table_cells, table_name)
                table_dataframes.append(single_table.table_dataframe)
                table_pages.append(table_page)
                # save the table headers, table name, and table column count
                last_table_headers = single_table.headers
                last_table_name = single_table.table_name
                last_table_col_count = len(last_table_headers[0])

        return table_dataframes, table_pages

    def get_table_nodes(self) -> list[TextNode]:
        """
        Depending on the value of self.by_token, creates a list of TextNode objects from
        the table dataframes stored in self.table_dataframes.

        If self.by_token is True, the method treats each entire table
        as a single unit, spliting table rows text by token limit.

        If self.by_token is False, the method iterates over each
        row in every table, creating a TextNode for each row with the row's data
        as its text.

        :return: A list of TextNode objects, each representing either a single row or an
        entire table from the processed table dataframes, depending on the
        configuration of the processor.
        """
        nodes = []
        if self.by_token:
            for idx, table_df in enumerate(self.table_dataframes):
                table_page = self.table_pages[idx]
                # split the table dataframe based on token limit.
                groups_of_rows, table_title = (
                    self.split_table_into_groups_by_token_limit(table_df)
                )

                # Create a TextNode for each group of rows.
                for group in groups_of_rows:
                    new_node = TextNode(
                        text=str(group),
                        metadata={
                            "from_table": table_title,
                            "page_number": table_page,
                            "document_name": self.file_name,
                            "document_type": self.file_type,
                        },
                    )
                    nodes.append(new_node)
        else:
            for idx, table_df in enumerate(self.table_dataframes):
                table_page = self.table_pages[idx]
                for table_row_text in table_df:
                    # Create a new document for each page
                    new_node = TextNode(
                        text=str(table_row_text),
                        metadata={
                            "page_number": table_page,
                            "document_name": self.file_name,
                            "document_type": self.file_type,
                        },
                    )
                    nodes.append(new_node)
        return nodes

    def split_table_into_groups_by_token_limit(self, table_df):
        """
        Splits a table dataframe into groups of rows based on a token length limit.

        :param table_df (list): The table dataframe, a list of row dictionaries.
        :param char_limit (int): The maximum character length for each group.

        :return groups_of_rows (list): A list of groups, where each group is a list of modified rows.
        :return table_title (str): The 'from' value extracted from the rows, if any.
        """
        groups_of_rows = []
        current_group = []
        current_group_token_count = 0
        table_title = None

        for row in table_df:
            # Capture the 'from' value if not already done.
            if "from" in row and table_title is None:
                table_title = row["from"]

            # Create a modified row without the 'from' key
            modified_row = {key: value for key, value in row.items() if key != "from"}
            # Convert the modified row to a string
            modified_row_str = str(modified_row)
            # Count tokens by splitting the string on whitespace
            modified_row_tokens = word_tokenize(modified_row_str)
            modified_row_token_count = len(modified_row_tokens)

            # Check if adding the row exceeds the token limit
            if current_group_token_count + modified_row_token_count > self.token_limit:
                # Start a new group if the current one exceeds the limit
                groups_of_rows.append(current_group)
                current_group = [modified_row]
                current_group_token_count = modified_row_token_count
            else:
                # Otherwise, add the row to the current group and update the token count
                current_group.append(modified_row)
                current_group_token_count += modified_row_token_count

        # Add the last group if it has any rows
        if current_group:
            groups_of_rows.append(current_group)

        return groups_of_rows, table_title

    def get_table_page_number(self, data: Dict[str, Any]) -> int:
        """
        Extracts the page number from table data.

        :param data: Dictionary containing table and metadata.
        :return: Page number of the table or 0 if not found.
        """
        bounding_regions = data.get("boundingRegions", [])
        return bounding_regions[0].get("pageNumber", 0) if bounding_regions else 0


class SingleTableProcessor:
    """
    Processes a single table data from a list of tables.
    Steps:
        1. Initialize with cell data, optional table name, and headers.
        2. Build table rows from cell data.
        3. Bild table dataframe, mapping cell content to headers.
    """

    def __init__(
        self,
        table_cells: List[Dict[str, Any]],
        table_name: Optional[str] = None,
        table_headers: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """
        Initializes a new instance of the SingleTableProcessor class.

        :param table_cells: List of dictionaries, each containing info about a cell.
        :param table_name: Optional name of the table.
        :param table_headers: Optional list of headers for the table.
        """
        self.cell_data = table_cells
        self.table_name = table_name
        self.headers = table_headers

        self.table_rows: Dict[int, Dict[str, Any]] = self.build_table_rows()
        self.table_dataframe: List[Dict[str, Any]] = self.build_table_df()

    def build_table_rows(self) -> Dict[int, Dict[str, Any]]:
        """
        Constructs the table structure from cell data.
        :return: Structured representation of the table.

        Example:
        {
            row_index: {                 # int - Row index (e.g., 0, 1, 2, ...)
                'kind': str,             # Type of row ('columnHeader' or 'data')
                'content': {
                    col_index: str       # Column index mapped to cell data
                    ...
                },
            },
            ...
        }
        """
        table_rows = {}
        for cell in self.cell_data:
            row_idx, col_idx = cell.get("rowIndex", 0), cell.get("columnIndex", 0)
            row_span, col_span = cell.get("rowSpan", 1), cell.get("columnSpan", 1)
            content = cell.get("content", "")
            # The 'kind' attribute for each row is either set to 'columnHeader' or,
            # if unspecified, defaults to 'data'.
            cell_kind = cell.get("kind", "data")

            if col_idx == 0:
                row_kind = cell_kind

            for r in range(row_span):
                for c in range(col_span):
                    if row_idx + r not in table_rows:
                        table_rows[row_idx + r] = {"kind": row_kind, "content": {}}
                    table_rows[row_idx + r]["content"][col_idx + c] = content
        return table_rows

    def build_table_df(self) -> List[Dict[str, Any]]:
        """
        Processes the structured table data.
        :return: Processed data as a list of dictionaries.

        Example:
        [
            {
                from: str,               # Table name
                Header_1: Content_1,     # Mapping header and content
                Header_2: Content_2,     # Mapping header and content
                ...
            },
            ...
        ]
        """
        table_dataframe = []
        self.get_table_headers()

        for _, row_data in self.table_rows.items():
            if row_data["kind"] == "columnHeader":
                continue

            row_dict = {"from": self.table_name}
            for col_idx, content in row_data["content"].items():
                content = UNWANTED_CONTENT.get(content, content)
                header = self.find_cell_header(col_idx)
                row_dict[header] = content
            table_dataframe.append(row_dict)

        return table_dataframe

    def get_table_headers(self) -> None:
        """
        Extracts headers from the table if not set.
        """
        if self.headers is None:
            self.headers = [
                row["content"]
                for row in self.table_rows.values()
                if row["kind"] == "columnHeader"
            ]

    def find_cell_header(self, col_idx: int) -> str:
        """
        Finds the header for a given column index.
        :param col_idx: Column index.
        :return: Header name for the column. Header name for the column.
                If empty string, returns a default header name.
        """
        header_name = ""
        for header in self.headers:
            if col_idx in header:
                current_header = header[col_idx]
                if current_header == "":
                    # Set default header name if empty
                    current_header = f"column_{col_idx}"
                # Set header_name if it's the first header found, and
                # Concatenate if more than one header found.
                if header_name == "":
                    header_name = current_header
                else:
                    header_name += " - " + current_header

        return header_name
