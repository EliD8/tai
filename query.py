import os.path
import nest_asyncio
import chromadb
import gradio as gr
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

PERSIST_DIR = "./chromadb"

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = chroma_client.get_collection("class_materials")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-large"),
    storage_context=storage_context
    )


llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1024)

window_post_processor = MetadataReplacementPostProcessor(target_metadata_key="window")
cohere_api_key = os.environ.get("COHERE_API_KEY")
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=3)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    llm=llm,
    node_postprocessors=[
        window_post_processor,
        cohere_rerank
    ],
    )


def query(input: str):
    return query_engine.query(input)


ui = gr.Interface(fn=query, inputs="text", outputs="text")

ui.launch()
