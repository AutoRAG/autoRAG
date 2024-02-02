from autorag.indexer.expanded_indexer import ExpandedIndexer
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Extracting specific configuration values from the loaded configuration.
    cur_cfg = cfg.indexer.build

    expanded_indexer = ExpandedIndexer.build(
        cur_cfg.data_dir,
        cur_cfg.pre_processor_cfg,
        cur_cfg.post_processor_cfg,
    )
    expanded_indexer.persist(cur_cfg.index_dir)


if __name__ == "__main__":
    main()
