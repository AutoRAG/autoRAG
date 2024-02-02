from collections import defaultdict
import os
from typing import Any, List, Optional
from llama_index.bridge.pydantic import Field
from llama_index.postprocessor.types import BaseNodePostprocessor
from llama_index.schema import TextNode, QueryBundle, NodeWithScore


class NodeExpander(BaseNodePostprocessor):
    """For nodes created from PDF files, NodeExpander will expand the original retrieved node to all the nodes in the same PDF file."""

    docstore: dict = Field(description="All the existing nodes")
    node2parent_mapping: dict = Field(
        description="mapping from original node to parent pseudo node"
    )
    parent2node_mapping: dict = Field(
        description="mapping from parent pseudo node to a list of its child nodes"
    )

    def __init__(
        self,
        docstore: dict = None,
        node2parent_mapping: dict = None,
        parent2node_mapping: dict = None,
    ):
        super().__init__(
            docstore=docstore,
            node2parent_mapping=node2parent_mapping,
            parent2node_mapping=parent2node_mapping,
        )

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[TextNode]:
        parent_node_ids = {}

        for node in nodes:
            parent_node_id = self.node2parent_mapping[node.node.id_]
            if parent_node_id not in parent_node_ids:
                parent_node_ids[parent_node_id] = node.score
        parent_size = len(parent_node_ids)
        expanded_nodes = []
        for parent_idx, (p_node_id, p_node_score) in enumerate(parent_node_ids.items()):
            child_size = len(self.parent2node_mapping[p_node_id])
            for child_idx, node_id in enumerate(self.parent2node_mapping[p_node_id]):
                new_score = (parent_size - parent_idx) + (
                    child_size - child_idx
                ) * 1.0 / child_size
                new_node = NodeWithScore(
                    node=self.docstore[node_id],
                    score=new_score,
                )
                expanded_nodes.append(new_node)
        return expanded_nodes

    @classmethod
    def build(cls, index, parent_metadata_field="document_name"):
        parent2node_mapping = defaultdict(list)
        node2parent_mapping = {}
        for node_id, node in index.docstore.docs.items():
            # assuming nodes in index.docstore.docs are ordered in the preferred way
            # For example, when parent_metadata_field is document_name, the nodes are ordered in the doc order
            parent2node_mapping[node.metadata[parent_metadata_field]] += [node_id]
            node2parent_mapping[node_id] = node.metadata[parent_metadata_field]
        docstore = index.docstore.docs
        return cls(docstore, node2parent_mapping, parent2node_mapping)
