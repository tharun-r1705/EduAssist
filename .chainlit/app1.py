from chainlit.element import ElementBased
import chainlit as cl
from dotenv import load_dotenv
import os
from io import BytesIO
from langchain_community.vectorstores.faiss import FAISS
from langchain_ai21 import AI21Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langsmith import Client
import logging
from groq import Groq
import vertexai
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from google.cloud import storage
import PyPDF2
import io
from langchain.chains import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory

'''logger setup'''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''google credentials setup'''
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'angelic-archery-434703-n1-72fb5ce2bbca.json'

'''vectorization using vertex ai'''
PROJECT_ID = "angelic-archery-434703-n1"
REGION = "us-central1"
MODEL_ID = "textembedding-gecko@001"

# chat = ChatGroq(model = "gemma2-9b-it")

vertexai.init(project=PROJECT_ID, location=REGION)
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
embeddings = VertexAIEmbeddings(model=MODEL_ID)

'''project id setup'''
project_id = "angelic-archery-434703-n1"
vertexai.init(project=project_id, location="us-central1")

'''loading .env file'''
load_dotenv()

'''Groq Api Setup'''
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
client1 = Client(api_key='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')
client = Groq(api_key=GROQ_API_KEY)
os.environ["GROQ_API_KEY"] = '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'
chat = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")

'''setting up groq llm and ai21 embedding'''
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# llm = VertexAI(model_name="gemini-1.0-pro-001")

embed = AI21Embeddings(
    api_key='1ejh0c6LGZPhJ5kI80NsQXTzOc6tfY6v',
    batch_size=200,
)

'''Vectorization of pdf '''


def vectorization(docsv):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(docsv)
    vectorstore = FAISS.from_texts(texts=splits, embedding=embeddings)
    print("Vector Store is Ready!")
    return vectorstore


'''Fetching data from cloud and extracting the text'''


def fetch_and_print_pdf(bucket_name, source_blob_name):
    # Initialize Google Cloud Storage client
    storage_client = storage.Client.from_service_account_json(
        r"angelic-archery-434703-n1-72fb5ce2bbca.json"
    )

    # Get the bucket and blob (PDF file in the cloud)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the blob content into memory
    pdf_data = blob.download_as_bytes()

    # Read PDF content using PyPDF2 PdfReader (updated for PyPDF2 3.0.0)
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))

    docs = ""

    # Iterate through pages and print the content
    for page_num, page in enumerate(pdf_reader.pages):
        # print(f"Page {page_num + 1} content:")
        # print(page.extract_text())  # Extract and print the text from the page
        docs = docs + page.extract_text()
    return docs


bucket_name = 'titans-1'
source_blob_name = 'DATA.pdf'
doc = fetch_and_print_pdf(bucket_name, source_blob_name)

'''Passing the text file and using rag_chain method'''

vector = vectorization(doc)

retriever = vector.as_retriever()


# prompt = client1.pull_prompt("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


'''def invo(d):

    #a =rag_chain.invoke({"input":d})
    return a'''
'''Converting speech to text'''


async def speech_to_text(audio_file):
    response = client.audio.translations.create(
        file=audio_file,  # Required audio file
        model="whisper-large-v3",  # Required model to use for translation
        prompt="Specify context or spelling",  # Optional
        response_format="json",  # Optional
        temperature=0.0  # Adjust this based on Groq's API
    )
    return response.text


'''Memory Management'''
'''def invo(prompt1):

    return conversational_rag_chain'''

'''Chainlit Interface'''


@cl.on_chat_start
def initialize_resources():
    print("A new session has started!")


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    cl.user_session.get("audio_buffer").write(chunk.data)


store = {}


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    # audio_buffer = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(audio_buffer)
    await cl.Message(content=transcription).send()

    try:
        qa_system_prompt = (
            "system"
            "You are an expert extraction algorithm. "
            "Give the answer that satisfies the user query."
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            "\n\n"
            "{context}"
        )

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                        which might reference context in the chat history, formulate a standalone question \
                        which can be understood without the chat history. Do NOT answer the question, \
                        just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        a = conversational_rag_chain.invoke(
            {"input": transcription},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )
        response = a
        answer = response["answer"]

    except Exception as e:
        answer = f"Error retrieving or processing answer: {str(e)}"
        logger.error(f"Exception occurred: {str(e)}")

    # Send the answer back to the user
    await cl.Message(content=answer).send()


@cl.on_message
async def main(message):
    try:
        # response = handle_user_query(message.content)
        # response = r
        # conversational_rag_chain = invo(message.content)
        qa_system_prompt = (
            "system"
            "You are an expert extraction algorithm. "
            "Give the answer that satisfies the user query."
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            "\n\n"
            "{context}"
        )

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        a = conversational_rag_chain.invoke(
            {"input": message.content},
            config={
                "configurable": {"session_id": "abc123"}
            },  # constructs a key "abc123" in `store`.
        )
        response = a
        answer = response["answer"]
    except Exception as e:
        answer = f"Error retrieving or processing answer: {str(e)}"
        logger.error(f"Exception occurred: {str(e)}")

    # Send the answer back to the user
    await cl.Message(content=answer).send()