import os.path
import shutil
import chromadb
import nest_asyncio
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.ingestion import IngestionPipeline
from sherpa_reader import LLMSherapaReader
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from document_sorter import DocumentSorter

nest_asyncio.apply()

PERSIST_DIR = "./chromadb"
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = chroma_client.get_or_create_collection("class_materials")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024)

# Load data
print("Loading Data")
documents = SimpleDirectoryReader("data", file_extractor={".pdf" : LLMSherapaReader()}).load_data()
info, questions, garbage, broken = DocumentSorter().sort(documents)
print("Data Loaded")

# Ingestion pipeline
pipeline = IngestionPipeline(
transformations=[
    SentenceWindowNodeParser.from_defaults(
        # how many sentences on either side to capture
        window_size=3,
        # the metadata key that holds the window of surrounding sentences
        window_metadata_key="window",
        # the metadata key that holds the original sentence
        original_text_metadata_key="original_sentence",
    ),
    OpenAIEmbedding(model_name="text-embedding-3-large")
],
vector_store=vector_store
)

nodes_post_pipe = pipeline.run(documents=info)

with open("nodes_post_pipe.txt", "w") as file:
    for node in nodes_post_pipe:
        file.write(node.text + "\n\n")

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=OpenAIEmbedding(model_name="text-embedding-3-large"), storage_context=storage_context)
