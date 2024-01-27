from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext

from .process.azure.output import AzureOutputProcessor
from .config_singleton import ConfigSingleton

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Setting the cfg in the singleton instance to the one loaded by Hydra.
    config_singleton = ConfigSingleton.get_instance()
    config_singleton.cfg = cfg

    # Extracting specific configuration values from the loaded configuration.
    data_dir = cfg.indexer.build.data_dir
    index_dir = cfg.indexer.build.index_dir
    data_processor = cfg.indexer.build.data_processor

    # Processing documents based on the specified data_processor type.
    if data_processor == "azure":
        nodes = AzureOutputProcessor(data_dir).nodes
        print(len(nodes))
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
