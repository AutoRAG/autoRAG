from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext

from .process.azure.output import AzureOutputProcessor

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):

    # Extracting specific configuration values from the loaded configuration.
    cur_cfg = cfg.indexer.build
    data_dir = cur_cfg.data_dir
    index_dir = cur_cfg.index_dir
    data_processor = cur_cfg.data_processor

    # Processing documents based on the specified data_processor type.
    if data_processor == "azure":
        # Initialize a SentenceSplitter with the given arguments
        sentence_splitter_args = cur_cfg.node_parser.args.sentence_splitter
        file_type = cur_cfg.file_type
        include_table = cur_cfg.include_table

        nodes = AzureOutputProcessor(
            data_dir, file_type, sentence_splitter_args, include_table
        ).nodes
        index = VectorStoreIndex(nodes)
    else:
        documents = SimpleDirectoryReader(data_dir).load_data()
        service_context = ServiceContext.from_defaults()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

    index.storage_context.persist(persist_dir=index_dir)


if __name__ == "__main__":
    main()
