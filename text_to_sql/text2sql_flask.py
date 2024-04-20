from flask import Flask, request, jsonify
from llama_index.core import SQLDatabase
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine, MetaData
from llama_index.core.query_engine import NLSQLTableQueryEngine
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

engine = create_engine("sqlite:///airbnb_ny.db")
metadata_obj = MetaData()
metadata_obj.reflect(engine)

os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

sql_database = SQLDatabase(engine, include_tables=["airbnb_ny"])

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["airbnb_ny"],
    llm=llm
)

@app.route('/predict', methods=['POST'])
def process_query():
    prompt = request.json.get('prompt')
    response = query_engine.query(prompt)
    
    return jsonify({
        'response': str(response),
        'sql_query': response.metadata['sql_query'],
        'result': response.metadata['result']
    })
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)