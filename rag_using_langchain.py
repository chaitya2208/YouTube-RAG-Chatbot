import os
from dotenv import load_dotenv
import yt_dlp
import whisper
import gradio as gr

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please add GOOGLE_API_KEY to your .env file")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


video_url = input("Enter YouTube video URL: ").strip()
text = ""


# Step 1: Try captions first
try:
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    docs = loader.load()
    if docs and docs[0].page_content.strip():
        text = docs[0].page_content
        print("✅ Transcript fetched from captions successfully!")
except Exception as e:
    print(f"⚠️ Captions not available or error: {e}")



# Step 2: Fallback to audio + Whisper
if not text.strip():
    print("🎧 Falling back to audio download and Whisper transcription...")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": "video_audio.%(ext)s",
        "quiet": True,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_file = ydl.prepare_filename(info)
        print("Audio downloaded:", audio_file)

    model = whisper.load_model("base")  # small/medium/large if GPU
    result = model.transcribe(audio_file, language="en", fp16=False)
    text = result["text"]
    print("✅ Transcript generated via Whisper")



# Validate transcript
if len(text.strip()) < 50:
    raise RuntimeError("❌ Transcript too short or invalid. Cannot continue.")

print(f"Transcript length: {len(text)} characters")
print("Transcript preview:", text[:300], "...\n")



# Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([text])
print(f"Total chunks created: {len(chunks)}")



# Create vector store & retriever
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})



# Setup Gemini LLM and prompt
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""",
    input_variables=['context', 'question']
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser



# Gradio interface
def answer_question(question):
    response = main_chain.invoke(question)
    return response

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask anything about the video..."),
    outputs="text",
    title="YouTube Video Chatbot (RAG + Gemini)",
    description="Enter your question about the video. The assistant answers based only on the transcript."
)

iface.launch()
