import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up environment
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "HF_TOKEN")

# Page configuration
st.set_page_config(page_title="YouTube Transcript Chatbot", layout="wide")
st.title("🎥 YouTube Transcript Chatbot")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Sidebar for video input
with st.sidebar:
    st.header("📝 Video Setup")
    video_id = st.text_input("Enter YouTube Video ID", placeholder="e.g., dQw4w9WgXcQ")
    
    if video_id and video_id != st.session_state.video_id:
        st.session_state.video_id = video_id
        st.session_state.chat_history = []  # Reset chat history for new video
        
        with st.spinner("Fetching transcript..."):
            try:
                # Fetch transcript
                fetched_transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
                transcript_list = fetched_transcript.to_raw_data()
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                
                # Split text into chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.create_documents([transcript])
                
                # Create embeddings and vector store
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vectorstore = FAISS.from_documents(chunks, st.session_state.embeddings)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity", search_kwargs={"k": 4}
                )
                
                # Load LLM
                model_id = "distilgpt2"
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=200,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    device=-1
                )
                st.session_state.llm = HuggingFacePipeline(pipeline=pipe)
                
                st.success(f"✅ Loaded {len(chunks)} chunks from transcript!")
                
            except TranscriptsDisabled:
                st.error("❌ Transcripts are disabled for this video.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Main chat interface
if st.session_state.vectorstore is not None:
    st.success(f"✅ Video loaded: {st.session_state.video_id}")
    
    # Chat history display
    st.subheader("💬 Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input for new question
    user_question = st.chat_input("Ask a question about the video transcript...")
    
    if user_question:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    retrieved_docs = st.session_state.retriever.invoke(user_question)
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    # Create prompt
                    prompt_template = PromptTemplate(
                        template="""Based on the context below, answer the question directly and concisely. If the question is out of context, say "I don't know".

Context: {context}

Question: {question}

Answer:""",
                        input_variables=["context", "question"]
                    )
                    
                    final_prompt = prompt_template.format(context=context_text, question=user_question)
                    
                    # Generate answer
                    answer = st.session_state.llm.invoke(final_prompt)
                    answer_only = answer.split("Question:")[-1].strip() if "Question:" in answer else answer
                    answer_only = answer_only.split("Answer:")[-1].strip() if "Answer:" in answer_only else answer_only
                    
                    st.write(answer_only)
                    
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer_only})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.info("👈 Enter a YouTube Video ID in the sidebar to get started!")


