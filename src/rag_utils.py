import os
import shutil
import git
import stat
from dotenv import load_dotenv  

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

REPO_DIR = "repos"
FAISS_INDEX_DIR = "./faiss_index"

if not os.path.exists(REPO_DIR):
    os.makedirs(REPO_DIR)

retriever = None
chat_history = []   

def handle_remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_and_process_repo(repo_url: str) -> str:
    global retriever

    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(REPO_DIR, repo_name)

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=handle_remove_readonly)

    git.Repo.clone_from(repo_url, repo_path)

    loader = GitLoader(clone_url=repo_url, repo_path=repo_path, branch=None)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_DIR)

    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3
    retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3],
        search_kwargs={"k": 5}
    )

    return repo_name

def query_repo(query: str) -> str:
    global retriever, chat_history
    if retriever is None:
        return "⚠️ No repository loaded. Please load one first."

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        google_api_key=GOOGLE_API_KEY
    )

    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that explains the GitHub repository. "
                   "Only explain parts relevant to the user's query."),
        ("human", "{input}"),
        ("system", "Relevant context:\n{context}")
    ])

    build_messages = (
        RunnableLambda(lambda q: {"input": q, "docs": retriever.invoke(q)})
        | RunnableLambda(lambda x: {
            "input": x["input"],
            "context": "\n\n".join(doc.page_content for doc in x["docs"])
        })
        | RunnableLambda(lambda x: doc_prompt.invoke(x).messages)
    )

    rag_with_history = (
        build_messages
        | RunnableLambda(lambda prompt_messages: chat_history.copy() + prompt_messages)
    )

    messages = rag_with_history.invoke(query)

    llm_output = model.invoke(messages)
    answer = StrOutputParser().parse(llm_output.content)

    messages.append(AIMessage(content=answer))
    chat_history.clear()
    chat_history.extend(messages)

    return answer
