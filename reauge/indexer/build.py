from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext

from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    data_dir = cfg.indexer.build.data_dir
    index_dir = cfg.indexer.build.index_dir
    service_context = ServiceContext.from_defaults()
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=index_dir)
