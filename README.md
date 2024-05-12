
# TAI - An LLM Teaching Assistant
TAI is an LLM (Large Language Model) application designed to leverage Retrieval Augmented Generation (RAG) for answering student questions using course materials.  It’s meant to help teachers leverage AI in their classrooms in a way that they can control and teach their students how to use AI tools while promoting learning, not detracting from it. At the beginning of an academic year, educators can upload all relevant course materials—PowerPoint slides, documents, textbooks, and more—into TAI, which processes and stores this information in a vector database to serve as a foundation for answering student questions.

# API Keys
You'll need  API keys for both Cohere and OpenAI. They need to be set at environment variables as:

```OPENAI_API_KEY="..."```

```COHERE_API_KEY="..."```
# Setup Instructions

1. run ```poetry init``` to install all required packages for the project
    https://python-poetry.org

2. Store course materials in ```/data```

3. execute ```poetry run python load_data.py``` to convert course materials into vector embeddings in the chromaDB vector store

>In order to run ```load_data.py``` you must run the nlm-ingestor docker containter with port mapping 5010:5001
    >>from NLM-ingestor readme:
        https://github.com/nlmatics/nlm-ingestor
        A docker image is available via public github container registry.

        Pull the docker image

        docker pull ghcr.io/nlmatics/nlm-ingestor:latest
        Run the docker image mapping the port 5001 to port of your      choice.

        docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest-<version>

4. query the LLM by running ``` poetry run python query.py ```

# Improvements and Testing
The ```testing_playground.ipynb``` notebook can be used to test MRR, hit rate, relevance, faithfullness, and to experiment with different ingestion pipeline componentse and LLMs