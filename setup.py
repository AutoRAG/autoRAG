from setuptools import setup, find_packages

setup(
    name="autorag",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "setuptools==75.8.0",
        "thefuzz==0.22.1",
        "openpyxl==3.1.5",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "pandas==2.2.3",
        "llama-index==0.12.15",
        "llama-index-core==0.12.15",
        "streamlit==1.41.1",
        "python-dotenv==1.0.1",
        "google-api-python-client==2.146.0",
        "flask==3.1.0",
        "flask_cors==5.0.0",
    ],
)
