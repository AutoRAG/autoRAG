# reauge
Easy development for retrieval augmented generation system

## Usage
This repo uses [Hydra](https://hydra.cc/docs/intro/) to manage the configs. Please check conf/config.yaml for the default values and use Hydra arguments to modify the default values. 

### Prepare a test dataset
Given some annotated data (question, reference) pairs, you can use the following command to prepare test data.
```
python -m reauge.data_builder.build_from_annotated_retrieval_data 
```
### Build an index
```
python -m reauge.indexer.build
```
### Evaluate the retriever
```
python -m reauge.retriever.evaluate
```
### Run a chatbot app
Note that you need to be in the entry directory of this repo to run the chatbot. 
```
streamlit run reauge/synthesizer/render.py
```
