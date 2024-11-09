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
        sentence_splitter_cfg: dict = {},
    ) -> None:    
        self.azure_paragraphs_list = azure_paragraphs_list
        self.file_name = file_name
        self.file_type = file_type
        self.chunk_size = sentence_splitter_cfg.get("chunk_size", 512)
        self.chunk_overlap = sentence_splitter_cfg.get("chunk_overlap", 32)
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size)

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

    def _create_nodes(self, filtered_pages: list[dict]) -> list[TextNode]:
        nodes = []
        all_words = []
        
        # First, collect all words with their metadata
        for page in filtered_pages:
            words = page.get("words", [])
            for word in words:
                all_words.append({
                    'content': word.get("content", "").strip(),
                    'page_number': page.get("pageNumber", 0),
                    'polygon': word.get("polygon", [])
                })

        # Process words with overlap
        i = 0
        while i < len(all_words):
            current_words = []
            first_page = all_words[i]['page_number']
            first_polygon = all_words[i]['polygon']
            
            # Add words until we reach chunk_size
            end_idx = min(i + self.chunk_size, len(all_words))
            current_words = [all_words[j]['content'] for j in range(i, end_idx)]
            current_page = all_words[end_idx - 1]['page_number']
            last_polygon = all_words[end_idx - 1]['polygon']

            # Create node
            if current_words:
                node = self._create_single_node(
                    " ".join(current_words),
                    first_page,
                    current_page,
                    first_polygon,
                    last_polygon
                )
                nodes.append(node)

            # Move forward by (chunk_size - overlap) or at least 1 to avoid infinite loop
            step_size = max(self.chunk_size - self.chunk_overlap, 1)
            i += step_size

        return nodes

    def _create_single_node(
        self,
        text: str,
        first_page: int,
        last_page: int,
        first_polygon: list,
        last_polygon: list
    ) -> TextNode:
        """
        Creates a single TextNode with metadata.

        :param text: Combined text content.
        :param first_page: First page number of the content.
        :param last_page: Last page number of the content.
        :param first_polygon: Polygon coordinates of first paragraph.
        :param last_polygon: Polygon coordinates of last paragraph.
        :return: TextNode object.
        """
        metadata = {
            "first_page_number": first_page,
            "last_page_number": last_page,
            "document_name": self.file_name,
            "document_type": self.file_type,
            "first_polygon": first_polygon,
            "last_polygon": last_polygon,
        }

        return TextNode(text=text, metadata=metadata, text_template="{content}")
