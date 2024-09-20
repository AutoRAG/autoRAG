from setuptools import setup, find_packages

setup(
    name="autorag",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "setuptools==75.1.0",
        "thefuzz==0.20.0",
        "openpyxl==3.1.2",
        "pypdf==3.17.4",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "pandas==2.1.0",
        "llama_index==0.9.28.post2",
        "streamlit==1.29.0",
        "python-dotenv==1.0.1",
        "google-api-python-client==2.146.0",
    ],
)
