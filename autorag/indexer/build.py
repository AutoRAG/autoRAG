from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext

from .process.azure.output import AzureOutputProcessor

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    data_dir = cfg.indexer.build.data_dir
    index_dir = cfg.indexer.build.index_dir
    data_processor = cfg.indexer.build.data_processor

    if data_processor == "azure":
        documents = AzureOutputProcessor(data_dir).documents
        print(len(documents))
        print(documents[0])
    else:
        documents = SimpleDirectoryReader(data_dir).load_data()
        service_context = ServiceContext.from_defaults()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist(persist_dir=index_dir)
 
    

if __name__ == "__main__":
    main()
    
