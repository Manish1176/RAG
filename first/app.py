import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

pdf=PyPDFLoader(r"C:\Users\MANISH\Downloads\Behavior Styles Checklist.pdf")

pdfpages=pdf.load_and_split()

mybooks=pdf.load()

text_splitter=CharacterTextSplitter(chunk_size=1500,chunk_overlap=0)

split_text= text_splitter.split_documents(mybooks)

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore=FAISS.from_documents(split_text,embeddings)

vectorstore_retriever=vectorstore.as_retriever()

from langchain.agents.agent_toolkits import create_retriever_tool

tool=create_retriever_tool(vectorstore_retriever,"Behavior_Styles_Checklist","Retrieve detailed information on behaviour")

tools=[tool]

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')

myagent=create_conversational_retrieval_agent(llm,tools,verbose=True)

input="how many number of columns in the list"

result=myagent.invoke({"input":input})


