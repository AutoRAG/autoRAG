import os
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.base.base_retriever import BaseRetriever
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from typing import List
import re


class GoogleRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        api_key: str = None,
        cse_id: str = None,
        topk: int = 10,
    ) -> None:
        """Init params."""
        api_key = api_key or os.environ["GOOGLE_SEARCH_API_KEY"]
        cse_id = cse_id or os.environ["GOOGLE_SEARCH_CSE_ID"]
        self.service = build("customsearch", "v1", developerKey=api_key).cse()
        self.cse_id = cse_id
        self.topk = topk
        super().__init__()

    @staticmethod
    def fetch_page_text(url):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"\n+", "\n", text)
            return text
        except requests.RequestException as e:
            print(f"Failed to retrieve {url}: {e}")
            return None

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # query_bundle.query_str is the original query
        res = self.service.list(
            q=query_bundle.query_str, cx=self.cse_id, num=self.topk
        ).execute()
        items = res.get("items", [])

        nodes_with_score = []
        total_items = len(items)
        for rank, item in enumerate(items):
            title = item["title"]
            link = item["link"]
            if link.endswith("download") or link.endswith(".pdf"):
                continue
            text = GoogleRetriever.fetch_page_text(link)
            if text is None:
                continue
            metadata = {
                "page_number": None,
                "document_name": title,
                "document_type": "webpage",
                "url": link,
            }

            node = TextNode(text=text, metadata=metadata)
            node_with_score = NodeWithScore(
                node=node, score=(total_items - rank) / total_items
            )
            nodes_with_score.append(node_with_score)
        return nodes_with_score


class GoogleAndVectorRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Google search."""

    def __init__(
        self, vector_retriever: VectorIndexRetriever, google_retriever: GoogleRetriever
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._google_retriever = google_retriever
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        google_nodes = self._google_retriever.retrieve(query_bundle)

        return vector_nodes + google_nodes
