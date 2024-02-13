from collections import defaultdict
import os
from typing import Any, List, Optional
from llama_index.bridge.pydantic import Field
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import (
    TextNode,
    QueryBundle,
    NodeWithScore,
    NodeRelationship,
    RelatedNodeInfo,
)
from llama_index.storage.docstore import SimpleDocumentStore

ORIGINAL_NODE_DIR_BASENAME = "original_docstore.json"
EXPANDED_NODE_DIR_BASENAME = "parent_docstore.json"


class NodeExpander(BaseNodePostprocessor):
    """For nodes created from PDF files, NodeExpander will expand the original retrieved node to all the nodes in the same PDF file."""

    all_original_nodes: dict = Field(description="All the original nodes")
    all_parent_nodes: dict = Field(
        description="All the parent nodes. For example, one parent is the whole PDF"
    )

    def __init__(
        self,
        all_original_nodes: dict = None,
        all_parent_nodes: dict = None,
    ):
        super().__init__(
            all_original_nodes=all_original_nodes,
            all_parent_nodes=all_parent_nodes,
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[TextNode]:
        parent_node_scores = {}
        """The original retrieved node from a PDF file will be expanded to all the nodes in the same PDF file. In other words, 
        if a node in a file is retrieved, all the nodes from this file will be returned."""
        for node in nodes:
            parent_node_info = self.all_original_nodes[node.node.node_id].relationships[
                NodeRelationship.PARENT
            ]
            if parent_node_info.node_id not in parent_node_scores:
                parent_node_scores[parent_node_info.node_id] = node.score
        num_parent_nodes = len(parent_node_scores)
        expanded_nodes = []
        for parent_idx, (e_node_id, e_node_score) in enumerate(
            parent_node_scores.items()
        ):
            parent_node = self.all_parent_nodes[e_node_id]
            child_nodes_info = parent_node.relationships[NodeRelationship.CHILD]
            assert isinstance(child_nodes_info, list), "child_nodes_info is not a list"

            for child_idx, child_node_info in enumerate(child_nodes_info):
                new_score = (num_parent_nodes - parent_idx) + (
                    len(child_nodes_info) - child_idx
                ) * 1.0 / len(child_nodes_info)
                new_node = NodeWithScore(
                    node=self.all_original_nodes[child_node_info.node_id],
                    score=new_score,
                )
                expanded_nodes.append(new_node)
        return expanded_nodes

    @classmethod
    def load(cls, persist_dir):
        ori_node_path = os.path.join(persist_dir, ORIGINAL_NODE_DIR_BASENAME)
        ori_nodes_list = SimpleDocumentStore.from_persist_path(ori_node_path)
        all_original_nodes = ori_nodes_list.docs

        exp_node_path = os.path.join(persist_dir, EXPANDED_NODE_DIR_BASENAME)
        exp_nodes_list = SimpleDocumentStore.from_persist_path(exp_node_path)
        all_parent_nodes = exp_nodes_list.docs
        # {node.node_id: node for node in exp_nodes_list}

        return cls(all_original_nodes, all_parent_nodes)

    def persist(self, persist_dir):
        ori_node_path = os.path.join(persist_dir, ORIGINAL_NODE_DIR_BASENAME)
        ori_docstore = SimpleDocumentStore()
        ori_docstore.add_documents(self.all_original_nodes.values())
        ori_docstore.persist(persist_path=ori_node_path)

        exp_node_path = os.path.join(persist_dir, EXPANDED_NODE_DIR_BASENAME)
        exp_docstore = SimpleDocumentStore()
        exp_docstore.add_documents(self.all_parent_nodes.values())
        exp_docstore.persist(persist_path=exp_node_path)

    @classmethod
    def build(cls, index, parent_metadata_field="document_name", sep=" "):
        parent2original_mapping = defaultdict(list)
        all_original_nodes = index.docstore.docs
        for node_id, node in all_original_nodes.items():
            # assuming nodes in index.docstore.docs are ordered in the preferred way
            # For example, when parent_metadata_field is document_name, the nodes are ordered in the doc order
            parent2original_mapping[node.metadata[parent_metadata_field]] += [node_id]
        all_parent_nodes = {}
        for parent_metadata, child_nodes in parent2original_mapping.items():
            parent_text = sep.join(all_original_nodes[n].text for n in child_nodes)
            child_nodes_info = [RelatedNodeInfo(node_id=n_id) for n_id in child_nodes]
            parent_node = TextNode(
                text=parent_text,
                metadata={parent_metadata_field: parent_metadata},
                relationships={NodeRelationship.CHILD: child_nodes_info},
            )
            all_parent_nodes[parent_node.node_id] = parent_node
            for child_node_id in child_nodes:
                related_parent_info = RelatedNodeInfo(node_id=parent_node.node_id)
                all_original_nodes[child_node_id].relationships = {
                    NodeRelationship.PARENT: related_parent_info
                }

        return cls(all_original_nodes, all_parent_nodes)
