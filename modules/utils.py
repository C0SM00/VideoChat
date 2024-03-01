from typing import Optional, Tuple

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from modules.models.qwen import QwenModel


def get_splits(text: list[Document], chunk_size=2000, chunk_overlap=0):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(text)


def get_summery(splits: list[Document], model: QwenModel, flag: bool):
    return model.summarize(splits) if flag else ""


def get_vector_store(splits: list[Document], model_name: str = "moka-ai/m3e-base"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(splits, embeddings)
    gr.Info("Vector store created. Ready to chat.")
    return vector_store


def get_answer(vector_store, query: str, history: Optional[Tuple[str, str]], model: QwenModel):
    return model.predict(vector_store, query, history)

# todo: get_answer_streaming
