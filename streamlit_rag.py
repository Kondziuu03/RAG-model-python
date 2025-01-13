import os
import shutil
import streamlit as st
import sqlite3
import requests
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
#from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader

DATABASE_PATH = "chat_history.db"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
auth = (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD"))

auth_kwargs = {
    'auth': auth,
    'verify': False, # Disable SSL verification
}

CHROMA_PATH = "chroma"

#RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#reranker = CrossEncoder(RERANKER_MODEL)

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_name TEXT,
                    query TEXT,
                    response TEXT,
                    sources TEXT,
                    model TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )''')
    conn.commit()
    conn.close()

def save_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_sessions")
    for session_name, history in st.session_state.chat_sessions.items():
        for entry in history:
            c.execute("INSERT INTO chat_sessions (session_name, query, response, sources, model) VALUES (?, ?, ?, ?, ?)",
                      (session_name, entry['query'], entry['response'], json.dumps(entry['sources']), entry['model']))
    conn.commit()
    conn.close()

def load_chat_history():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT session_name, query, response, sources, model FROM chat_sessions")
    rows = c.fetchall()
    chat_sessions = {}
    for row in rows:
        session_name, query, response, sources, model = row
        if session_name not in chat_sessions:
            chat_sessions[session_name] = []
        chat_sessions[session_name].append({
            "query": query,
            "response": response,
            "sources": json.loads(sources),
            "model": model
        })
    conn.close()
    return chat_sessions

def delete_session(session_name):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_sessions WHERE session_name = ?", (session_name,))
    conn.commit()
    conn.close()

    providers = ["OpenAI", "Ollama", "PG"]
    for provider in providers:
        try:
            collection_name = get_collection_name(provider, session_name)
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function(provider),
                collection_name=collection_name
            )
            db.delete_collection()
        except Exception as e:
            st.warning(f"Could not delete collection for {provider}: {str(e)}")

def get_embedding_function(provider):
    if provider == "OpenAI":
        return OpenAIEmbeddings(model="text-embedding-ada-002", 
                              openai_api_key=OPENAI_API_KEY)
    elif provider == "Ollama":
        return OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    elif provider == "PG":
        return OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

def clear_database(provider="OpenAI"):
    try:
        if not st.session_state.current_session:
            st.error("Please select or create a session first!")
            return
            
        collection_name = get_collection_name(provider, st.session_state.current_session)

        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(provider),
            collection_name=collection_name
        )
        db.delete_collection()
        
        st.success(f"Successfully cleared {collection_name} collection!")
    except Exception as e:
        st.error(f"Error clearing collection: {str(e)}")

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_installed_ollama_models():
    try:
        response = requests.get('http://ollama:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
        return []
    except:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
       
    return []

def get_collection_name(provider, session_name):
    return f"documents_{provider.lower()}_{session_name}"

def add_to_chroma(chunks: list[Document], provider="OpenAI"):
    try:
        if not chunks:
            st.warning("No documents to add to the database.")
            return
        
        if not st.session_state.current_session:
            st.error("Please select or create a session first!")
            return
        
        embedding_function = get_embedding_function(provider)
        collection_name = get_collection_name(provider, st.session_state.current_session)
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function,
            collection_name=collection_name
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        
        db.add_documents(chunks_with_ids, ids=chunk_ids)
        
        collection_size = len(db.get()['ids'])
        st.info(f"Dodano {collection_size} dokumentów do kolekcji {collection_name}.")
        
    except Exception as e:
        st.error(f"Error adding documents to database: {str(e)}")
        raise e

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def get_data_path(provider, session_name):
    return os.path.join("data", provider.lower(), session_name)

def populate_database(files, provider="OpenAI"):
    if not st.session_state.current_session:
        st.error("Please select or create a session first!")
        return
        
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    
    data_path = get_data_path(provider, st.session_state.current_session)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    documents = []

    for uploaded_file in files:
        file_path = os.path.join(data_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

    chunks = split_documents(documents)
    add_to_chroma(chunks, provider)
    
    st.success(f"Database populated successfully! Documents saved in {data_path}")

def get_reranked_documents(query: str, provider="OpenAI"):
    initial_results = get_similar_documents(query, provider);
    
    sorted_results = sorted(initial_results, key=lambda x: x[1])
    
    #for doc, score in sorted_results:
        #st.write(f"Similarity score: {score:.4f} - {doc.page_content[:100]}...")

    return sorted_results
    
    """
    query_doc_pairs = [(query, doc.page_content) for doc, _ in initial_results]
    scores = reranker.predict(query_doc_pairs)

    reranked_results = sorted(zip([doc for doc, _ in initial_results], scores),
                              key=lambda x: x[1], reverse=True)
    
    return [(doc, score) for doc, score in reranked_results]
    """
    
def get_similar_documents(query: str, provider="OpenAI"):
    if not st.session_state.current_session:
        st.error("Please select or create a session first!")
        return []
        
    collection_name = get_collection_name(provider, st.session_state.current_session)
    
    db = Chroma(
         persist_directory=CHROMA_PATH,
         embedding_function=get_embedding_function(provider),
         collection_name=collection_name
    )
    
    k = st.session_state.get('chroma_k', 10)
    return db.similarity_search_with_score(query, k=k)

def get_settings():
    settings = load_settings()
    return {
        'OPENAI_API_KEY': settings.get('openai_api_key', os.getenv("OPENAI_API_KEY")),
        'OLLAMA_URL': settings.get('ollama_url', 'http://ollama:11434'),
        'PG_URL': settings.get('pg_url', 'https://153.19.239.239'),
        'PG_USERNAME': settings.get('pg_username', os.getenv("LLM_USERNAME")),
        'PG_PASSWORD': settings.get('pg_password', os.getenv("LLM_PASSWORD"))
    }

def PG(prompt):
    settings = get_settings()
    base_url = settings['PG_URL']
    api_endpoint = f"{base_url}/api/llm/prompt/chat"
    auth = (settings['PG_USERNAME'], settings['PG_PASSWORD'])
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_length": 2000,
        "temperature": 0.7
    }

    response = requests.put(
        api_endpoint,
        json=data,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        **auth_kwargs,
    )
    response.raise_for_status()

    response_json = response.json()
    return response_json['response']

def query_rag(query, provider="OpenAI", model="GPT-4o"):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        
    #results = db.similarity_search_with_score(query, k=5)
    #results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    
    reranked_results = get_reranked_documents(query, provider)

    k = st.session_state.get('rerank_k', 5)
    top_results = reranked_results[:k]

    prompt_template = ChatPromptTemplate.from_template(
        """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Provide a right answer with a short explanation.
        """
    )
    
    prompt_template_pl = ChatPromptTemplate.from_template(
        """
        Użyj poniższych informacji, aby odpowiedzieć na pytanie użytkownika.
        Jeśli nie znasz odpowiedzi, po prostu powiedz, że nie wiesz, nie próbuj wymyślać odpowiedzi.

        Kontekst: {context}
        Pytanie: {question}

        Podaj prawidłową odpowiedź wraz z krótkim wyjaśnieniem.
        """
    )

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
    prompt = prompt_template.format(context=context, question=query)

    client = None
    response = None

    if provider == "OpenAI":
        settings = get_settings()
        client = OpenAI(api_key=settings['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content
    elif provider == "Ollama":
        res = requests.get(f"{settings['OLLAMA_URL']}/api/tags")
        
        client = OllamaLLM(model=model, base_url=settings['OLLAMA_URL'])
        response = client.invoke(prompt)
    elif provider == "PG":
        response = PG(prompt_template_pl.format(context=context, question=query))

    return response, [doc.metadata for doc, _ in top_results]


def pull_ollama_model(model_name):
    try:
        settings = get_settings()
        
        response = requests.post(f"{settings['OLLAMA_URL']}/api/pull", json={'name': model_name})
        
        if response.status_code == 200:
            return True, "Model pobrany prawidłowo!"
        else:
            return False, f"Nie udało się pobrać modelu: {response.text}"
    except Exception as e:
        return False, f"Błąd podczas pobierania modelu: {str(e)}"
    
def save_settings(settings_dict):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    for key, value in settings_dict.items():
        c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                  (key, value))
    conn.commit()
    conn.close()

def load_settings():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute("SELECT key, value FROM settings")
    settings = dict(c.fetchall())
    conn.close()
    return settings

st.title("RAG - przeszukiwanie dokumentów")

if "chat_sessions" not in st.session_state:
    init_db()
    st.session_state.chat_sessions = load_chat_history()
    st.session_state.current_session = None

def manage_sessions():
    st.header("Zarządzanie sesjami")
    
    new_session_name = st.text_input("Nazwa nowej sesji")
    if st.button("Utwórz sesję"):
        if new_session_name and new_session_name not in st.session_state.chat_sessions:
            st.session_state.chat_sessions[new_session_name] = []
            st.session_state.current_session = new_session_name
            st.success(f"Sesja '{new_session_name}' utworzona i aktywowana!")
        elif new_session_name in st.session_state.chat_sessions:
            st.warning("Sesja już istnieje!")
        else:
            st.warning("Proszę podać nazwę sesji.")

    if st.session_state.chat_sessions:
        session_selector = st.selectbox("Wybierz sesję", st.session_state.chat_sessions.keys())
        if st.button("Przełącz na wybraną sesję"):
            st.session_state.current_session = session_selector
            st.success(f"Przełączono na sesję '{session_selector}'!")

    if st.session_state.chat_sessions:
        session_to_delete = st.selectbox("Usuń sesję", st.session_state.chat_sessions.keys())
        
        if st.button("Usuń wybraną sesję"):
            delete_session(session_to_delete)
            
            del st.session_state.chat_sessions[session_to_delete]
            
            if st.session_state.current_session == session_to_delete:
                st.session_state.current_session = None
                
            st.success(f"Sesja '{session_to_delete}' usunięta!")
            st.rerun()

    if st.session_state.current_session:
        st.write(f"### Aktywna sesja: {st.session_state.current_session}")
    else:
        st.write("Brak aktywnej sesji.")

    if st.session_state.current_session:
        save_chat_history()

def upload_files():
    st.header("Dokumenty")

    if not st.session_state.current_session:
        st.warning("Brak aktywnej sesji. Proszę utworzyć lub wybrać sesję w 'Zarządzanie sesjami'.")
    else:
        st.write(f"### Aktywna sesja: {st.session_state.current_session}")
        
        uploaded_files = st.file_uploader("Prześlij pliki PDF", type="pdf", accept_multiple_files=True, help="Maksymalny rozmiar pliku: 200MB")
        provider = st.selectbox("Dostawca", ["OpenAI", "Ollama", "PG"]) 
        
        if st.button("Resetuj bazę dokumentów"):
            clear_database(provider)

        if st.button("Załaduj wybrane dokumenty") and uploaded_files:
            with st.spinner("Przetwarzanie dokumentów..."):
                populate_database(uploaded_files, provider)

def query_database():
    st.header("Chat")

    if not st.session_state.current_session:
        st.warning("Brak aktywnej sesji. Proszę utworzyć lub wybrać sesję w 'Zarządzanie sesjami'.")
    else:
        st.write(f"### Aktywna sesja: {st.session_state.current_session}")
         
        session_name = st.session_state.current_session

        query_text = st.text_input("Pytanie:")
        
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox("Dostawca", ["OpenAI", "Ollama", "PG"])
        with col2:
            if provider == "OpenAI":
                model = st.selectbox("Model", ["OpenAI GPT-4o"])
            elif provider == "Ollama":
                installed_models = get_installed_ollama_models()
                if not installed_models:
                    st.error("Brak modeli Ollama. Upewnij się, że Ollama działa i ma zainstalowane modele.")
                    return
                model = st.selectbox("Model", installed_models)
            elif provider == "PG":
                model = st.selectbox("Model", ["Bielik-11B-v2.2-Instruct model"])

        col3, col4 = st.columns(2)
        with col3:
            chroma_k = st.number_input(
                "Dokumenty pobrane z ChromaDB",
                min_value=1,
                max_value=20,
                value=10
            )
        with col4:
            rerank_k = st.number_input(
                "Dokumenty pobrane z rerankingu",
                min_value=1,
                max_value=chroma_k,
                value=min(5, chroma_k)
            )

        if st.button("Wyślij") and query_text:
            with st.spinner("Pobieranie informacji..."):
                st.session_state.chroma_k = chroma_k
                st.session_state.rerank_k = rerank_k
                response, sources = query_rag(query_text, provider, model)
            modelInfo = f"Dostawca: {provider}, Model: {model}"
            
            st.session_state.chat_sessions[session_name].append({
                "query": query_text,
                "response": response,
                "sources": sources,
                "model": modelInfo
            })
            
            save_chat_history()

        st.subheader(f"Historia chatu dla sesji: {session_name}")
        chat_history = st.session_state.chat_sessions[session_name]
        if chat_history:
            for idx, entry in enumerate(chat_history):
                st.write(f"**Q{idx+1}:** {entry['query']}")
                st.write(f"**A{idx+1} ({entry['model']}):** {entry['response']}")
                
                with st.expander(f"Źródła #{idx+1}"):
                    for source in entry['sources']:
                        st.write(source)
                    
                st.markdown("---")
        else:
            st.write("Brak historii chatu dla tej sesji.")

def ollama_management():
    st.header("Zarządzanie modelami Ollama")
    
    model_name = st.text_input("Nazwa modelu do pobrania (np. llama3.1, mistral)")
    
    if st.button("Pobierz model"):
        if not model_name:
            st.warning("Proszę podać nazwę modelu")
        else:
            with st.spinner(f"Pobieranie modelu {model_name}..."):
                success, message = pull_ollama_model(model_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    st.subheader("Zainstalowane modele")
    installed_models = get_installed_ollama_models()
    if installed_models:
        for model in installed_models:
            st.write(f"- {model}")
    else:
        st.info("Brak zainstalowanych modeli")

def settings_page():
    st.header("Ustawienia")

    current_settings = load_settings()

    openai_key = st.text_input(
        "OpenAI - Klucz API",
        value=current_settings.get('openai_api_key', ''),
        type="password"
    )
    
    ollama_url = st.text_input(
        "Ollama - URL",
        value=current_settings.get('ollama_url', 'http://ollama:11434')
    )
    
    pg_url = st.text_input(
        "PG Bielik - URL",
        value=current_settings.get('pg_url', 'https://153.19.239.239')
    )
    
    pg_username = st.text_input(
        "PG Bielik - Nazwa użytkownika",
        value=current_settings.get('pg_username', ''),
        type="password"
    )
    
    pg_password = st.text_input(
        "PG Bielik - Hasło",
        value=current_settings.get('pg_password', ''),
        type="password"
    )
    
    if st.button("Zapisz Ustawenia"):
        settings = {
            'openai_api_key': openai_key,
            'ollama_url': ollama_url,
            'pg_url': pg_url,
            'pg_username': pg_username,
            'pg_password': pg_password
        }
        save_settings(settings)
        st.success("Ustawienia zostały zapisane!")

pg = st.navigation([
    st.Page(settings_page, title="Ustawienia"),
    st.Page(ollama_management, title="Ollama"),
    st.Page(manage_sessions, title="Sesje"), 
    st.Page(upload_files, title="Dokumenty"), 
    st.Page(query_database, title="Chat"),
])

pg.run()