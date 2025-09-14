# import os
# import hashlib
# from dotenv import load_dotenv
# import yt_dlp
# import whisper
# import gradio as gr

# from langchain_community.document_loaders import YoutubeLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI


# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please add GOOGLE_API_KEY to your .env file")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# CACHE_DIR = "faiss_cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# video_url = None
# main_chain = None


# def get_video_id(url: str):
#     """Generate a unique ID for the video based on its URL."""
#     return hashlib.md5(url.encode()).hexdigest()


# def build_chain(video_url: str):
#     """Loads transcript (or Whisper fallback), creates or loads FAISS index, and returns a QA chain."""
#     video_id = get_video_id(video_url)
#     cache_path = os.path.join(CACHE_DIR, video_id)

#     # Use HuggingFace embeddings locally (no quota)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # If cache exists, load it
#     if os.path.exists(cache_path):
#         print(f"🔄 Loading cached FAISS index for video: {video_url}")
#         vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
#     else:
#         print(f"📥 Processing video: {video_url}")
#         text = ""

#         # Step 1: Try captions
#         try:
#             loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
#             docs = loader.load()
#             if docs and docs[0].page_content.strip():
#                 text = docs[0].page_content
#                 print("✅ Transcript fetched from captions successfully!")
#         except Exception as e:
#             print(f"⚠️ Captions not available or error: {e}")

#         # Step 2: Fallback to Whisper
#         if not text.strip():
#             print("🎧 Falling back to audio download and Whisper transcription...")
#             ydl_opts = {
#                 "format": "bestaudio/best",
#                 "noplaylist": True,
#                 "outtmpl": "video_audio.%(ext)s",
#                 "quiet": True,
#                 "overwrites": True,
#             }
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 info = ydl.extract_info(video_url, download=True)
#                 audio_file = ydl.prepare_filename(info)
#                 print("Audio downloaded:", audio_file)

#             model = whisper.load_model("base")
#             result = model.transcribe(audio_file, language="en", fp16=False)
#             text = result["text"]
#             print("✅ Transcript generated via Whisper")

#         if len(text.strip()) < 50:
#             raise RuntimeError("❌ Transcript too short or invalid. Cannot continue.")

#         print(f"Transcript length: {len(text)} characters")
#         print("Transcript preview:", text[:300], "...\n")

#         # Split into chunks
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = splitter.create_documents([text])
#         print(f"Total chunks created: {len(chunks)}")

#         # Build FAISS index and save
#         vector_store = FAISS.from_documents(chunks, embeddings)
#         vector_store.save_local(cache_path)
#         print(f"💾 Saved FAISS index to {cache_path}")

#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#     # Gemini LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

#     prompt = PromptTemplate(
#         template="""
# You are a helpful assistant.
# Answer ONLY from the provided transcript context.
# If the context is insufficient, just say you don't know.

# Context:
# {context}

# Question: {question}
# """,
#         input_variables=["context", "question"]
#     )

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     parallel_chain = RunnableParallel({
#         "context": retriever | RunnableLambda(format_docs),
#         "question": RunnablePassthrough()
#     })
#     parser = StrOutputParser()
#     chain = parallel_chain | prompt | llm | parser
#     return chain


# def load_video(url):
#     global video_url, main_chain
#     video_url = url
#     try:
#         main_chain = build_chain(video_url)
#         return "✅ Video loaded successfully! You can now ask questions."
#     except Exception as e:
#         return f"❌ Failed to load video: {str(e)}"


# def chat_with_bot(message, history):
#     if not main_chain:
#         return "⚠️ Please provide a YouTube video link first."
#     response = main_chain.invoke(message)
#     return response


# # Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("## 🎥 YouTube Video Chatbot (RAG + Gemini)")
#     gr.Markdown("Paste a YouTube link, load the transcript, and start chatting!")

#     with gr.Row():
#         url_box = gr.Textbox(label="YouTube Video URL", placeholder="Paste a YouTube video link...")
#         load_button = gr.Button("Load Video")
#         status = gr.Label(value="")

#     chatbot = gr.ChatInterface(
#         fn=chat_with_bot,
#         chatbot=gr.Chatbot(),
#         textbox=gr.Textbox(placeholder="Ask anything about the video...", container=False),
#         title="Chat",
#     )

#     load_button.click(fn=load_video, inputs=url_box, outputs=status)

# demo.launch()






import os
import json
import hashlib
from dotenv import load_dotenv
import yt_dlp
import whisper
import gradio as gr

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# === Setup ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please add GOOGLE_API_KEY to your .env file")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

CACHE_DIR = "faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

video_url = None
main_chain = None


def get_video_id(url: str):
    """Create a hash for caching files based on video URL."""
    return hashlib.md5(url.encode()).hexdigest()


def get_cache_paths(video_id: str):
    """Return FAISS and transcript cache file paths."""
    faiss_path = os.path.join(CACHE_DIR, video_id)
    transcript_path = os.path.join(CACHE_DIR, f"{video_id}_transcript.json")
    return faiss_path, transcript_path


def load_or_generate_transcript(video_url: str):
    """Load transcript from cache or generate it via captions/Whisper."""
    video_id = get_video_id(video_url)
    _, transcript_path = get_cache_paths(video_id)

    # Check transcript cache
    if os.path.exists(transcript_path):
        print(f"🔄 Loading cached transcript for {video_url}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)["transcript"]

    # Try captions
    text = ""
    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        if docs and docs[0].page_content.strip():
            text = docs[0].page_content
            print("✅ Transcript fetched from captions successfully!")
    except Exception as e:
        print(f"⚠️ Captions not available or error: {e}")

    # Fallback to Whisper
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

        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="en", fp16=False)
        text = result["text"]
        print("✅ Transcript generated via Whisper")

    if len(text.strip()) < 50:
        raise RuntimeError("❌ Transcript too short or invalid. Cannot continue.")

    # Save transcript to cache
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump({"transcript": text}, f, ensure_ascii=False, indent=2)
    print(f"💾 Transcript cached at {transcript_path}")

    return text


def build_chain(video_url: str):
    """Create or load FAISS index and return QA chain."""
    video_id = get_video_id(video_url)
    faiss_path, _ = get_cache_paths(video_id)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS if available
    if os.path.exists(faiss_path):
        print(f"🔄 Loading cached FAISS index for {video_url}")
        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Get transcript (cached or new)
        text = load_or_generate_transcript(video_url)
        print(f"Transcript length: {len(text)} characters")
        print("Transcript preview:", text[:300], "...\n")

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        print(f"Total chunks created: {len(chunks)}")

        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(faiss_path)
        print(f"💾 Saved FAISS index to {faiss_path}")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Setup Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.
            Provide a **clear and detailed answer** based ONLY on the transcript context.
            If the context is insufficient, explain that clearly.

            Context:
            {context}

            Question: {question}
            """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser
    return chain


def load_video(url):
    global video_url, main_chain
    video_url = url
    try:
        main_chain = build_chain(video_url)
        return "✅ Video loaded successfully! You can now ask questions."
    except Exception as e:
        return f"❌ Failed to load video: {str(e)}"


def chat_with_bot(message, history):
    if not main_chain:
        return "⚠️ Please provide a YouTube video link first."
    return main_chain.invoke(message)


# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 YouTube Video Chatbot (RAG + Gemini)")
    gr.Markdown("Paste a YouTube link, load the transcript, and start chatting!")

    with gr.Row():
        url_box = gr.Textbox(label="YouTube Video URL", placeholder="Paste a YouTube video link...")
        load_button = gr.Button("Load Video")
        status = gr.Label(value="")

    chatbot = gr.ChatInterface(
        fn=chat_with_bot,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask anything about the video...", container=False),
        title="Chat",
    )

    load_button.click(fn=load_video, inputs=url_box, outputs=status)

demo.launch()

