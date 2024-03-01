import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chains import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from modules.prompt_template import *


class QwenModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def summarize(self, splits: list[Document]):
        PROMPT = PromptTemplate(template=SUMMERY_TEMPLATE, input_variables=["text"])

        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        llm = Tongyi()
        chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True,
                                     map_prompt=PROMPT, combine_prompt=PROMPT)
        os.environ["DASHSCOPE_API_KEY"] = ""

        summery = chain.invoke({"input_documents": splits}, return_only_outputs=True)
        return summery["output_text"]

    def predict(self, vector_store, query: str, history: list[tuple[str, str]]):
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        PROMPT = PromptTemplate(template=QA_TEMPLATE, input_variables=["chat_history", "question"])
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=Tongyi(),
            retriever=vector_store.as_retriever(),
            memory=memory,
            condense_question_prompt=PROMPT,
        )
        os.environ["DASHSCOPE_API_KEY"] = ""
        return qa_chain.invoke({"question": query})["answer"].strip()
