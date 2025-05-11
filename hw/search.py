
# Load vector database that was persisted earlier and check collection count in it
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

persist_directory = 'C:/my_project/chroma_db'
os.environ["OPENAI_API_KEY"] = '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'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

question = "List all papers  by al-banna"


llm = ChatOpenAI( temperature=0)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

result = qa_chain({"query": question})
print(result["result"])