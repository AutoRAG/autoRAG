from llama_index import Document
from llama_index.schema import TextNode
from llama_index.node_parser import SentenceSplitter

# Define roles to be excluded
DEFAULT_EXCLUDED_ROLES: list[str] = ["pageHeader", "pageNumber"]

# Define contents to be excluded
DEFAULT_EXCLUDED_CONTENTS: list[str] = ["Contains Nonbinding Recommendations"]


class AzureParagraphProcessor:
    """
    Process azure paragraphs list using sentence splitting.

    :param azure_paragraphs_list: The list of paragraphs returned from Azure.
    :param file_name: The name of the file.
    :param file_type: The type of the file.
    :param sentence_splitter_args: Arguments for the SentenceSplitter function.
                                   This can include arguments like chunk_size,
                                   chunk_overlap, or any other arguments that
                                   SentenceSplitter expects.
    """

    def __init__(
        self,
        azure_paragraphs_list: list[dict] = None,
        file_name: str = None,
        file_type: str = None,
        sentence_splitter_args: dict = {},
    ) -> None:

        # Initialize the AzureParagraphProcessor class.
        self.azure_paragraphs_list = azure_paragraphs_list
        self.file_name = file_name
        self.file_type = file_type

        # Filter content and obtain documents and nodes
        filtered_paragraphs = self._filter_content()
        combined_page_content = self.combined_page_content(filtered_paragraphs)
        self.documents = self.get_documents(combined_page_content)
        self.nodes = self.get_nodes(sentence_splitter_args)

    def _filter_content(self) -> list[dict]:
        """
        Filters out unwanted content.

        :return: A list of filtered content.
        """
        filtered_content = []
        for paragraph in self.azure_paragraphs_list:
            # Retrieve the role and content of each paragraph
            role = paragraph.get("role", "")
            content = paragraph.get("content", "")

            # Skip paragraphs with excluded roles or contents
            if role in DEFAULT_EXCLUDED_ROLES or content in DEFAULT_EXCLUDED_CONTENTS:
                continue
            else:
                filtered_content.append(paragraph)

        return filtered_content

    def combined_page_content(self, filtered_content: list[dict]) -> dict[int, str]:
        """
        Combines filtered content by page.

        :param filtered_content: Filtered paragraph content.
        :return: A dictionary where each key is a page number (int) and each
             value is a string combining all the text content from that page.
        """
        combined_page_content = {}
        for content in filtered_content:
            # Assuming there's always one bounding region per paragraph
            content_page = content.get("boundingRegions", [])[0].get("pageNumber", 0)
            content_text = content.get("content", "").strip()

            # Combine content from the same page
            if content_page not in combined_page_content:
                combined_page_content[content_page] = content_text
            else:
                combined_page_content[content_page] += " " + content_text
        return combined_page_content

    def get_documents(self, combined_page_content: dict[int, str]) -> list[Document]:
        """
        Creates a list of Document objects from the combined page content.

        :param combined_page_content: [{page(int):content(str)},].
        :return: A list of Document objects, each representing a page from the
                original content, along with its metadata.
        """
        documents = []
        for content_page, content_text in combined_page_content.items():
            # Create a new document for each page
            new_document = Document(
                text=content_text,
                metadata={
                    "page_number": content_page,
                    "document_name": self.file_name,
                    "document_type": self.file_type,
                },
            )
            documents.append(new_document)
        return documents

    def get_nodes(self, sentence_splitter_args: dict) -> list[TextNode]:
        """
        Get nodes from documents.

        :return: A list of splitted text nodes with the given config.
        """
        splitter = SentenceSplitter(**sentence_splitter_args)
        # Get nodes using the SentenceSplitter
        nodes = splitter.get_nodes_from_documents(self.documents, show_progress=True)
        return nodes


class AzurePolygonParagraphProcessor:
    """
    Process azure paragraphs list with polygon tracking and chunk size control.

    :param azure_paragraphs_list: The list of paragraphs returned from Azure.
    :param file_name: The name of the file.
    :param file_type: The type of the file.
    :param chunk_size: Maximum number of words per node (default: 512).
    """

    def __init__(
        self,
        azure_paragraphs_list: list[dict] = None,
        file_name: str = None,
        file_type: str = None,
        chunk_size: int = 512,
    ) -> None:
        self.azure_paragraphs_list = azure_paragraphs_list
        self.file_name = file_name
        self.file_type = file_type
        self.chunk_size = chunk_size

        # Process paragraphs into nodes
        filtered_paragraphs = self._filter_content()
        self.nodes = self._create_nodes(filtered_paragraphs)

    def _filter_content(self) -> list[dict]:
        """
        Filters out unwanted content.

        :return: A list of filtered content.
        """
        filtered_content = []
        for paragraph in self.azure_paragraphs_list:
            role = paragraph.get("role", "")
            content = paragraph.get("content", "")

            if role in DEFAULT_EXCLUDED_ROLES or content in DEFAULT_EXCLUDED_CONTENTS:
                continue
            else:
                filtered_content.append(paragraph)

        return filtered_content

    def _create_nodes(self, filtered_paragraphs: list[dict]) -> list[TextNode]:
        """
        Creates nodes by combining paragraphs and tracking polygon boundaries.
        Ensures each node's content doesn't exceed the chunk_size (in words).

        :param filtered_paragraphs: List of filtered paragraph dictionaries.
        :return: List of TextNode objects.
        """
        nodes = []
        current_text = []
        current_word_count = 0
        current_page = None
        first_polygon = None
        last_polygon = None

        for paragraph in filtered_paragraphs:
            content = paragraph.get("content", "").strip()
            page = paragraph["boundingRegions"][0]["pageNumber"]
            polygon = paragraph["boundingRegions"][0]["polygon"]

            # Count words in current paragraph
            word_count = len(content.split())

            # Start new node if:
            # 1. Page number changes
            # 2. Adding this paragraph would exceed chunk_size
            if (current_page is not None and page != current_page) or (
                current_word_count + word_count >= self.chunk_size
            ):
                if current_text:
                    node = self._create_single_node(
                        " ".join(current_text),
                        current_page,
                        first_polygon,
                        last_polygon,
                    )
                    nodes.append(node)
                    current_text = []
                    current_word_count = 0
                    first_polygon = None

            # Track first and last polygons
            if not first_polygon:
                first_polygon = polygon
            last_polygon = polygon

            current_text.append(content)
            current_word_count += word_count
            current_page = page

        # Create final node if there's remaining text
        if current_text:
            node = self._create_single_node(
                " ".join(current_text), current_page, first_polygon, last_polygon
            )
            nodes.append(node)

        return nodes

    def _create_single_node(
        self, text: str, page: int, first_polygon: list, last_polygon: list
    ) -> TextNode:
        """
        Creates a single TextNode with metadata.

        :param text: Combined text content.
        :param page: Page number.
        :param first_polygon: Polygon coordinates of first paragraph.
        :param last_polygon: Polygon coordinates of last paragraph.
        :return: TextNode object.
        """
        metadata = {
            "page_number": page,
            "document_name": self.file_name,
            "document_type": self.file_type,
            "first_polygon": first_polygon,
            "last_polygon": last_polygon,
        }

        return TextNode(text=text, metadata=metadata, text_template="{content}")
