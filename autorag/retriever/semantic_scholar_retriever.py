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

QUERY2KEYWORD_PROMPT_TEMPLATE = PromptTemplate(
    template="Given a natural language question or a conversion, rewrite it into a short keyword-based query. \n\n<Original question>\n{question}\n\n<keyword-based query>\n"
)

RELEVANCE_CHECK_PROMPT = PromptTemplate(
    template="""Given the following question and paper information, determine the relevance of the paper on a scale of 1 to 5, where:

1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Very relevant
5 = Extremely relevant

Question: {question}

Paper Title: {title}
Abstract: {abstract}

Please respond with only a single number between 1 and 5, representing the Relevance Score.

Example output:
4

Relevance Score (1-5):"""
)

KEYWORD_IMPROVEMENT_PROMPT = PromptTemplate(
    template="""The current keywords '{keywords}' did not yield sufficiently relevant results for the question: '{question}'. 
    Please suggest {num_keywords} improved keywords that will be used to search papers powered by a traditional keyword based search engine.
    Please respond with a list of keywords separated by newlines.

    Example output:
    - keyword phrase 1
    - keyword phrase 2
    - keyword phrase 3

    List of keywords:"""
)


class SemanticScholarRetriever(BaseRetriever):
    """Custom retriever that performs semantic search for papers"""

    def __init__(
        self,
        directory: str = "papers",
        api_key: str = None,
        topk: int = 10,
        openai_model_name: str = "gpt-4-1106-preview",
    ) -> None:
        """Init params."""
        self.directory = directory
        self.api_key = api_key or os.environ["S2_API_KEY"]
        self.topk = topk
        self.llm = OpenAI(model=openai_model_name, temperature=0)
        super().__init__()

    def download_pdf(
        self, session: Session, url: str, path: str, user_agent: str = "requests/2.0.0"
    ):
        # this method is not used for now
        # send a user-agent to avoid server error
        headers = {
            "user-agent": user_agent,
        }

        # stream the response to avoid downloading the entire file into memory
        with session.get(url, headers=headers, stream=True, verify=False) as response:
            # check if the request was successful
            response.raise_for_status()

            if response.headers["content-type"] != "application/pdf":
                raise Exception("The response is not a pdf")

            with open(path, "wb") as f:
                # write the response to the file, chunk_size bytes at a time
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    def get_paper_text(self, paper_id, paper_url):
        # this method is not used for now
        try:
            with Session() as session:
                pdf_path = os.path.join(self.directory, f"{paper_id}.pdf")
                # create the directory if it doesn't exist
                os.makedirs(self.directory, exist_ok=True)
                self.download_pdf(session, paper_url, pdf_path)

            return parse_single_pdf(pdf_path)
        except:
            return None

    def search(self, query, topk):
        # Define the API endpoint URL
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # More specific query parameter
        query_params = {
            "query": query,
            "limit": topk,
            "fields": "paperId,title,abstract,openAccessPdf,url",
        }

        # Define headers with API key
        headers = {"x-api-key": self.api_key}

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
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

    def query_to_keywords(self, query):
        return self.llm.predict(QUERY2KEYWORD_PROMPT_TEMPLATE, question=query)

    def _retrieve_with_iterative_improvement(
        self,
        query_bundle: QueryBundle,
        max_iterations: int = 3,
        min_highly_relevant: int = 10,
        relevance_score_threshold: int = 4,
    ) -> List[NodeWithScore]:
        """Retrieve nodes with iterative keyword improvement."""
        question = query_bundle.query_str
        keywords = self.query_to_keywords(question)
        highly_relevant_count = 0
        highly_relevant_nodes = []
        list_of_keywords = [keywords]
        set_of_paper_ids = set()
        for iteration in range(max_iterations):
            if len(list_of_keywords) <= iteration:
                break
            cur_keywords = list_of_keywords[iteration]
            print(f"Iteration {iteration}, keywords: {cur_keywords}")
            res = self.search(cur_keywords, self.topk)
            items = res.get("data", [])

            for item in items:
                title = item["title"]
                abstract = item["abstract"]
                if item["paperId"] in set_of_paper_ids:
                    continue
                relevance_score = self.llm.predict(
                    RELEVANCE_CHECK_PROMPT,
                    question=question,
                    title=title,
                    abstract=abstract,
                )

                try:
                    relevance_score = float(relevance_score.strip())
                except ValueError:
                    print(
                        f"Invalid relevance score: {relevance_score}. Skipping this item."
                    )
                    continue

                if relevance_score >= relevance_score_threshold:
                    if item["abstract"] is None:
                        print(f"Skipping {title} because it has no abstract")
                        continue
                    node = self._create_node_from_item(item, relevance_score)
                    highly_relevant_nodes.append(node)
                    set_of_paper_ids.add(item["paperId"])

            print(
                f"So far found {len(highly_relevant_nodes)} highly relevant papers (score >= {relevance_score_threshold})"
            )

            if len(highly_relevant_nodes) >= min_highly_relevant:
                print(
                    f"Reached the target of {min_highly_relevant} highly relevant papers. Stopping iteration."
                )
                print(
                    f"number of highly relevant nodes found: {len(highly_relevant_nodes)}"
                )
                break

            if iteration == 0:
                list_of_keywords_str = self.llm.predict(
                    KEYWORD_IMPROVEMENT_PROMPT,
                    keywords=cur_keywords,
                    question=question,
                    num_keywords=max_iterations - 1,
                )
                try:
                    list_of_keywords += [
                        kws.strip("- \n")
                        for kws in list_of_keywords_str.split("\n")
                        if kws.strip("- \n") != ""
                    ]
                    print(list_of_keywords)
                except Exception as e:
                    print(f"failed to parse {list_of_keywords_str}. error: {str(e)}")
                    break

        # Sort the relevant_nodes by score in descending order
        highly_relevant_nodes.sort(key=lambda x: x.score, reverse=True)
        return highly_relevant_nodes

    def _create_node_from_item(self, item, relevance_score=None):
        title = item["title"]
        paper_id = item["paperId"]
        paper_url = item["url"]
        text = item["abstract"]

        metadata = {
            "page_number": None,
            "document_name": title,
            "document_type": "paper",
            "url": paper_url,
        }

        node = TextNode(text=text, metadata=metadata)

        return NodeWithScore(
            node=node, score=relevance_score
        )  # We could adjust the score based on relevance if needed

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query using the iterative improvement method."""
        return self._retrieve_with_iterative_improvement(query_bundle)
