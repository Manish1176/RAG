#loading external data
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"C:\Users\MANISH\Downloads\yolov9_paper.pdf")

data=loader.load()

#Spliting data into smaller chunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000)
docs=text_splitter.split_documents(data)

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Using LLMs
from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

#Performing embeddings and soring in vector DB
embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore=FAISS.from_documents(documents=docs,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

#Creating retriever for fetching documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

retrieved_docs = retriever.invoke("What is new in yolov9?")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0.3, max_tokens=500)

#Creating retrieveral Chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#Invoking chian for gettiing response
response = rag_chain.invoke({"input": "what is new in YOLOv9?"})
print(response["answer"])
