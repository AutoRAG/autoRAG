import os
from llama_index.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.base_retriever import BaseRetriever
import requests
from bs4 import BeautifulSoup
from typing import List
import re
from autorag.data_builder.pdf_to_txt import parse_single_pdf
from requests import Session
from typing import Generator, Union
from llama_index.llms import OpenAI
from llama_index.prompts.base import PromptTemplate

QUERY2KEYWORD_PROMPT_TEMPLATE = PromptTemplate(template='Given a natural language question or a conversion, rewrite it into a short keyword-based query. \n\n<Original question>\n{question}\n\n<keyword-based query>\n')

class SemanticScholarRetriever(BaseRetriever):
    """Custom retriever that performs semantic search for papers"""

    def __init__(
        self,
        directory: str = "papers",
        api_key: str = None,
        topk: int = 10,
        openai_model_name: str = 'gpt-4-1106-preview',
    ) -> None:
        """Init params."""
        self.directory = directory
        self.api_key = api_key or os.environ["S2_API_KEY"]
        self.topk = topk
        self.llm = OpenAI(model=openai_model_name, temperature=0)        
        super().__init__()

    def download_pdf(self, session: Session, url: str, path: str, user_agent: str = 'requests/2.0.0'):
        # this method is not used for now
        # send a user-agent to avoid server error
        headers = {
            'user-agent': user_agent,
        }

        # stream the response to avoid downloading the entire file into memory
        with session.get(url, headers=headers, stream=True, verify=False) as response:
            # check if the request was successful
            response.raise_for_status()

            if response.headers['content-type'] != 'application/pdf':
                raise Exception('The response is not a pdf')

            with open(path, 'wb') as f:
                # write the response to the file, chunk_size bytes at a time
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    def get_paper_text(self, paper_id, paper_url):
        # this method is not used for now
        try:
            with Session() as session:
                pdf_path = os.path.join(self.directory, f'{paper_id}.pdf')
                # create the directory if it doesn't exist
                os.makedirs(self.directory, exist_ok=True)
                self.download_pdf(session, paper_url, pdf_path)
            
            return parse_single_pdf(pdf_path)
        except:
            return None
            
    def search(self, query, topk):
        # Define the API endpoint URL
        url = 'https://api.semanticscholar.org/graph/v1/paper/search'
        # More specific query parameter
        query_params = {'query': query, 'limit': topk, 'fields': 'paperId,title,abstract,openAccessPdf,url'}

        # Define headers with API key
        headers = {'x-api-key': self.api_key}

        print(url)
        print(query_params)
        print(headers)
        # Send the API request
        response = requests.get(url, params=query_params, headers=headers)

        # Check response status
        if response.status_code == 200:
           response_data = response.json()
           # Process and print the response data as needed
           return response_data
        else:
           print(f"Request failed with status code {response.status_code}: {response.text}")

    def query_to_keywords(self, query):
        return self.llm.predict(
            QUERY2KEYWORD_PROMPT_TEMPLATE, question=query
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        topk = self.topk
        keywords = self.query_to_keywords(query_bundle.query_str)
        print(f'rewritten keywords for semantic scholar: {keywords}')
        res = self.search(keywords, topk)

        items = res.get("data", [])
        nodes_with_score = []
        total_items = len(items)
        for rank, item in enumerate(items):
            title = item["title"]
            paper_id = item["paperId"]
            paper_url = item['url']            
            text = item['abstract']
            #text = self.get_paper_text(paper_id, paper_url)
            if text is None:
                print(f'paper: {title}, paper_id: {paper_id}, link: {paper_url} not found')
                continue
            metadata = {
                "page_number": None,
                "document_name": title,
                "document_type": "paper",
                "paper_url": paper_url,
            }

            node = TextNode(text=text, metadata=metadata)
            node_with_score = NodeWithScore(
                node=node, score=(total_items - rank) / total_items
            )
            nodes_with_score.append(node_with_score)
        return nodes_with_score
