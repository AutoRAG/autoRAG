from llama_index import Document
from llama_index.schema import TextNode
from llama_index.node_parser import SentenceSplitter

# Define roles to be excluded
DEFAULT_EXCLUDED_ROLES: list[str] = ["pageHeader", "pageNumber"]

# Define contents to be excluded
DEFAULT_EXCLUDED_CONTENTS: list[str] = ["Contains Nonbinding Recommendations"]


class AzureParagraphProcessor:
    """
    Process azure paragraphs list

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
