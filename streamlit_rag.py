import os
import shutil
import streamlit as st
import sqlite3
import requests
import json
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_ollama import OllamaLLM
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader

DATABASE_PATH = "./data/chat_history.sqlite3"
CHROMA_PATH = "chroma"

#RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#reranker = CrossEncoder(RERANKER_MODEL)

def get_available_providers():
    settings = get_settings()
    providers = ["PG", "Ollama"]
    if settings.get('OPENAI_API_KEY'):
        providers.append("OpenAI")
    return providers

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

    for provider in get_available_providers():
        clear_database(provider)

def get_embedding_function(provider):
    settings = get_settings()
    if provider == "OpenAI":
        api_key = settings['OPENAI_API_KEY']
        if not api_key:
            raise ValueError("OpenAI API key not configured. Please set it in Settings.")
        return OpenAIEmbeddings(model="text-embedding-ada-002", 
                              openai_api_key=api_key)
    elif provider == "Ollama":
        return OllamaEmbeddings(model="nomic-embed-text", base_url=settings['OLLAMA_URL'])
    elif provider == "PG":
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def clear_database(provider):
    try:
        if not st.session_state.current_session:
            st.error("Najpierw wybierz lub utwórz sesję!")
            return
            
        available_docs = get_available_collections(provider, st.session_state.current_session)
        
        # Delete base collection first
        base_collection_name = get_collection_name(provider, st.session_state.current_session)
        try:
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function(provider),
                collection_name=base_collection_name,
                collection_metadata={"hnsw:space": "cosine"}
            )
            db.delete_collection()
        except Exception as e:
            st.warning(f"Could not delete base collection {base_collection_name}: {str(e)}")
            
        # Delete document-specific collections
        if available_docs:
            for doc_name, _ in available_docs:
                collection_name = get_collection_name(provider, st.session_state.current_session, doc_name)
                try:
                    db = Chroma(
                        persist_directory=CHROMA_PATH,
                        embedding_function=get_embedding_function(provider),
                        collection_name=collection_name,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                    db.delete_collection()
                except Exception as e:
                    st.warning(f"Could not delete collection {collection_name}: {str(e)}")
        
        # Reset the loaded state for this provider
        st.session_state.loaded[provider] = False
        
        st.success(f"Usunięto załadnowane dokumenty dla {provider}!")
    except Exception as e:
        st.error(f"Wystąpił błąd podczas usuwania: {str(e)}")

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
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

def get_collection_name(provider, session_name, document_name=None):
    # Clean the session name first
    clean_session = ''.join(c if c.isalnum() else '_' for c in session_name)
    clean_session = clean_session.strip('_')
    
    base_name = f"docs_{provider.lower()}_{clean_session}"
    
    if not document_name or not document_name.strip():
        return base_name
        
    clean_name = os.path.splitext(document_name)[0]
    clean_name = ''.join(c if c.isalnum() else '_' for c in clean_name)
    clean_name = clean_name.strip('_')
    
    full_name = f"{base_name}_{clean_name}"
    
    # Ensure the name meets Chroma's requirements
    if len(full_name) > 63:
        full_name = full_name[:63]
    if not full_name[0].isalnum():
        full_name = 'c' + full_name[1:]
    if not full_name[-1].isalnum():
        full_name = full_name[:-1] + 'c'
        
    return full_name

def add_to_chroma(chunks: list[Document], provider, document_name=None):
    try:
        if not chunks:
            st.warning("Brak fragmentów do dodania.")
            return
        
        if not st.session_state.current_session:
            st.error("Najpierw wybierz lub utwórz sesję!")
            return
        
        embedding_function = get_embedding_function(provider)
        collection_name = get_collection_name(provider, st.session_state.current_session, document_name)
        
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function,
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
        
        db.add_documents(chunks_with_ids, ids=chunk_ids)
        
        collection_size = len(db.get()['ids'])
        st.info(f"Dodano {collection_size} fragmentów do kolekcji {collection_name}.")
        
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

def populate_database(files, provider):
    if not st.session_state.current_session:
        st.error("Najpierw wybierz lub utwórz sesję!")
        return
        
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    
    data_path = get_data_path(provider, st.session_state.current_session)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    
    for uploaded_file in files:
        documents = []

        file_path = os.path.join(data_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

        chunks = split_documents(documents)
        add_to_chroma(chunks, provider, uploaded_file.name)

def get_reranked_documents(query: str, provider, selected_docs=None):
    initial_results = get_similar_documents(query, provider, selected_docs)
    sorted_results = sorted(initial_results, key=lambda x: x[1])  # ascending
    return sorted_results
    
    """
    query_doc_pairs = [(query, doc.page_content) for doc, _ in initial_results]
    scores = reranker.predict(query_doc_pairs)

    reranked_results = sorted(zip([doc for doc, _ in initial_results], scores),
                              key=lambda x: x[1], reverse=True)
    
    return [(doc, score) for doc, score in reranked_results]
    """
    
def get_similar_documents(query: str, provider, selected_docs=None):
    if not st.session_state.current_session:
        st.error("Najpierw wybierz lub utwórz sesję!")
        return []
    
    all_results = []
    k = st.session_state.get('chroma_k', 30)
    
    collection_name = get_collection_name(provider, st.session_state.current_session, selected_docs)
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function(provider),
            collection_name=collection_name,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        results = db.similarity_search_with_score(query, k=k)
        all_results.extend(results)
    except Exception as e:
        st.warning(f"Nie można przeszukać kolekcji {collection_name}: {str(e)}")
    
    # Sort all results by score
    return sorted(all_results, key=lambda x: x[1])[:k]

def get_settings():
    settings = load_settings()
    return {
        'OPENAI_API_KEY': settings.get('openai_api_key', ''),
        'OLLAMA_URL': settings.get('ollama_url', 'http://ollama:11434'),
        'PG_URL': settings.get('pg_url', 'https://153.19.239.239'),
        'PG_USERNAME': settings.get('pg_username', ''),
        'PG_PASSWORD': settings.get('pg_password', '')
    }

def PG(prompt):
    settings = get_settings()
    base_url = settings['PG_URL']
    api_endpoint = f"{base_url}/api/llm/prompt/chat"
    auth = (settings['PG_USERNAME'], settings['PG_PASSWORD'])
    
    auth_kwargs = {
        'auth': auth,
        'verify': False, # Disable SSL verification
    }
    data = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_length": 2000,
        "temperature": 0.5
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

def query_rag(query, provider, model, lang, selected_docs=None, brawl=False):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
        
    reranked_results = get_reranked_documents(query, provider, selected_docs)

    k = st.session_state.get('rerank_k', 15)
    top_results = reranked_results[:k]

    # Create enhanced metadata with text snippets and scores
    enhanced_sources = []
    for doc, score in top_results:
        source_info = {
            'id': doc.metadata.get('id', 'N/A'),
            'page': doc.metadata.get('page', 'N/A'),
            'source': doc.metadata.get('source', 'N/A'),
            'score': f"{score:.4f}",
            'text_snippet': doc.page_content
        }
        enhanced_sources.append(source_info)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Jesteś pomocnym asystentem specjalizującym się w analizie dokumentów w języku polskim.
        Użyj poniższych informacji, aby odpowiedzieć na pytanie użytkownika.
        Jeśli nie znasz odpowiedzi lub brakuje ci potrzebnego kontekstu, po prostu powiedz, że nie wiesz, nie próbuj wymyślać odpowiedzi.\n\n
        
        ---------------------------\n\n
        Kontekst: {context}\n\n
        ---------------------------\n\n
        Pytanie: {question}\n\n
        ---------------------------\n\n
        Podaj prawidłową odpowiedź wraz z krótkim wyjaśnieniem. 
        Używaj poprawnej polskiej gramatyki i interpunkcji.
        Nie cytuj fragmentów tekstu.
        Odpowiedz krótko i konkretnie w maksymalnie dwóch zdaniach.
        """
    ) if lang == "Polski" else ChatPromptTemplate.from_template(
        """
        You are a helpful assistant specializing in document analysis in English.
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        ---------------------------
        Question: {question}
        ---------------------------

        Provide a right answer with a short explanation.
        If you quote a piece of text, mark it with quotation marks.
        """
    )

    if brawl:
        prompt_template = ChatPromptTemplate.from_template("""
            Jesteś ekspertem od wymyślania pytań do tekstu.
            Użyj poniższych informacji, aby stworzyć dokładnie tylko jedno pytanie na podstawie kontekstu.
            Pytanie to powinno dotyczyć treści tekstu, fabuly lub informacji zawartych w tekście.
            Nie pytaj o znaczenie słów, formy wyrazów czy też poprwaność gramatyczną.
            Nie proś o użycie słowa w zdaniu lub odmianę.
            Jeśli nie wiesz jak sformułować pytanie, po prostu powiedz, że nie wiesz.

            Kontekst: {context}
                                                           
            Nie odpowiadaj na swoje pytanie. Podaj po prostu pytanie w znaczniku <question>{{pytanie}}</question>.
        """) if lang == "Polski" else ChatPromptTemplate.from_template("""
            You are an expert in generating questions from text.
            Use the following information to create exactly just one question based on the context.
            The question should be about the content of the text, the plot, or the information contained in the text.
            Do not ask about the meaning of words, word forms, or grammatical correctness.
            Do not ask for the use of a word in a sentence or its inflection.
            If you don't know how to phrase the question, just say that you don't know.

            Context: {context}

            Do not answer your question. Just provide the question in the <question>{{question}}</question> tag.
        """)

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
    prompt = prompt_template.format(context=context, question=query)
    
    client = None
    response = None
    settings = get_settings()

    if provider == "OpenAI":
        if not settings['OPENAI_API_KEY']:
            raise ValueError("OpenAI API key not configured. Please set it in Settings.")
        client = OpenAI(api_key=settings['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        ).choices[0].message.content
    elif provider == "Ollama":
        client = OllamaLLM(model=model, base_url=settings['OLLAMA_URL'])
        response = client.invoke(prompt)
    elif provider == "PG":
        response = PG(prompt)

    return response, enhanced_sources


def pull_ollama_model(model_name):
    try:
        settings = get_settings()
        
        response = requests.post(f"{settings['OLLAMA_URL']}/api/pull", json={'name': model_name})
        
        if response.status_code == 200:
            return True, f"Model {model_name} pobrany prawidłowo!"
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

def get_available_collections(provider, session_name):
    if not os.path.exists(CHROMA_PATH):
        return []
    
    base_prefix = f"docs_{provider.lower()}_{session_name}"
    collections = []
    
    embedding_function = get_embedding_function(provider)

    client = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name="temp",
        collection_metadata={"hnsw:space": "cosine"}
    )._client
    
    all_collections = client.list_collections()
    
    for collection in all_collections:
        if collection.name.startswith(base_prefix):
            doc_name = collection.name[len(base_prefix):]
            if doc_name.startswith('_'):
                doc_name = doc_name[1:]
            
            # Check if collection has documents
            try:
                db = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=embedding_function,
                    collection_name=collection.name,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                if len(db.get()['ids']) > 0:  # Only add if collection has documents
                    collections.append([doc_name, len(db.get()['ids'])])
            except Exception as e:
                st.warning(f"Could not check collection {collection.name}: {str(e)}")
    
    return collections

def clear_all_chroma():
    try:
        if os.path.exists(CHROMA_PATH):
            # Get all collections and delete them
            client = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function("PG"),
                collection_name="temp"
            )
            
            for collection in client._client.list_collections():
                db = Chroma(
                    persist_directory=CHROMA_PATH,
                    embedding_function=get_embedding_function("PG"),
                    collection_name=collection.name
                )
                db.delete_collection()
            
            # Reset loaded state
            st.session_state.loaded = {p: False for p in get_available_providers()}
            
            return True, "Pomyślnie wyczyszczono ChromaDB!"
    except Exception as e:
        return False, f"Wystąpił błąd podzas czyszczenia ChromaDB: {str(e)}"

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
            update_loaded()
            st.success(f"Sesja '{new_session_name}' utworzona i aktywowana!")
        elif new_session_name in st.session_state.chat_sessions:
            st.warning("Sesja już istnieje!")
        else:
            st.warning("Proszę podać nazwę sesji.")

    if st.session_state.chat_sessions:
        session_selector = st.selectbox("Wybierz sesję", st.session_state.chat_sessions.keys())
        if st.button("Przełącz na wybraną sesję"):
            st.session_state.current_session = session_selector
            update_loaded()
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

def update_loaded():
    for provider in get_available_providers():
        # Get all collections for this provider
        available_docs = get_available_collections(provider, st.session_state.current_session)
        has_documents = False
        try:
            base_collection_name = get_collection_name(provider, st.session_state.current_session)
            db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function(provider),
                collection_name=base_collection_name,
                collection_metadata={"hnsw:space": "cosine"}
            )
            if len(db.get()['ids']) > 0:
                has_documents = True
        except:
            pass
        
        if not has_documents and available_docs:
            for doc_name, _ in available_docs:
                try:
                    collection_name = get_collection_name(provider, st.session_state.current_session, doc_name)
                    db = Chroma(
                        persist_directory=CHROMA_PATH,
                        embedding_function=get_embedding_function(provider),
                        collection_name=collection_name,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                    if len(db.get()['ids']) > 0:
                        has_documents = True
                        break
                except:
                    continue
        
        st.session_state.loaded[provider] = has_documents

def upload_files():
    st.header("Dokumenty")

    if not st.session_state.current_session:
        st.warning("Brak aktywnej sesji. Proszę utworzyć lub wybrać sesję w 'Zarządzanie sesjami'.")
    else:
        st.write(f"### Aktywna sesja: {st.session_state.current_session}")
        
        uploaded_files = st.file_uploader("Prześlij pliki PDF", type="pdf", accept_multiple_files=True, help="Maksymalny rozmiar pliku: 200MB")
        provider = st.selectbox("Dostawca", get_available_providers()) 
        
        if st.button("Resetuj bazę dokumentów"):
            clear_database(provider)

        if st.button("Załaduj wybrane dokumenty") and uploaded_files:
            with st.spinner("Przetwarzanie dokumentów..."):
                populate_database(uploaded_files, provider)

                update_loaded()

def query_database():
    st.header("Czat")

    if not st.session_state.current_session:
        st.warning("Brak aktywnej sesji. Proszę utworzyć lub wybrać sesję w 'Zarządzanie sesjami'.")
    else:
        st.write(f"### Aktywna sesja: {st.session_state.current_session}")
         
        session_name = st.session_state.current_session

        query_text = st.text_input("Pytanie:")

        response = None
        sources = None
        
        col1, col2 = st.columns(2)
        with col1:
            provider = st.selectbox("Dostawca", get_available_providers())
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
                model = st.selectbox("Model", ["speakleash/Bielik-11B-v2.2-Instruct"])

        # Add document selection
        with st.spinner("Pobieranie dostępnych dokumentów..."):
            available_docs = get_available_collections(provider, session_name)
                    
        docs = [doc_name for doc_name, _ in available_docs]
        counts = {}
        for doc_name, count in available_docs:
            counts[doc_name] = count
        selected_doc = st.selectbox("Wybierz dokument", docs)

        col3, col4 = st.columns(2)
        with col3:
            chroma_k = st.number_input(
                "Fragmenty pobrane z ChromaDB",
                min_value=1,
                max_value=40 if selected_doc is None else max(20, round(counts[selected_doc]/2)),
                value=20 if selected_doc is None else max(4, round(counts[selected_doc]/10))
            )
        with col4:
            rerank_k = st.number_input(
                "Fragmenty pobrane z rerankingu",
                min_value=1,
                max_value=round(chroma_k/2),
                value=min(15, round(chroma_k/4))
            )

        lang = st.radio("Język", ["Polski", "Angielski"], horizontal=True)

        if not st.session_state.loaded[provider]:
             st.warning("Żaden dokument nie został załadowany do bazy danych. Proszę załadować dokumenty w zakładce 'Dokumenty'.")

        if st.session_state.loaded[provider] and st.button("Wyślij") and query_text:
            with st.spinner(f"Generowanie odpowiedzi na pytanie \"{query_text}\"..."):
                st.session_state.chroma_k = chroma_k
                st.session_state.rerank_k = rerank_k
                response, sources = query_rag(query_text, provider, model, lang, selected_doc)

                if response:
                    modelInfo = f"Dostawca: {provider}, Model: {model}"
                    st.session_state.chat_sessions[session_name].append({
                        "query": query_text,
                        "response": response,
                        "sources": sources,
                        "model": modelInfo
                    })
                    save_chat_history()

        st.subheader(f"Historia czatu dla sesji: {session_name}")
        chat_history = st.session_state.chat_sessions[session_name]
        if chat_history:
            for idx, entry in enumerate(reversed(chat_history)):
                real_idx = len(chat_history) - idx  # Calculate the real index for display
                st.write(f"**Q{real_idx}:** {entry['query']}")
                st.write(f"**A{real_idx} ({entry['model']}):** {entry['response']}")
                
                with st.expander(f"Źródła #{real_idx}"):
                    if entry.get('sources') and isinstance(entry['sources'], list):
                        for source in entry['sources']:
                            st.write("---")
                            st.write(f"📄 **ID:** {source.get('id', 'N/A')}")
                            st.write(f"📑 **Strona:** {source.get('page', 'N/A')}")
                            st.write(f"📁 **Źródło:** {source.get('source', 'N/A')}")
                            st.write(f"🎯 **Scoring:** {source.get('score', 'N/A')}")
                            st.write(f"📝 **Fragment tekstu:** *Kliknij, aby rozwinąć*")
                            st.html(f"<style>#i{idx} summary{{display:inline;cursor:pointer}}#i{idx} summary::after{{content:'...'}}#i{idx}[open] summary::after{{content:''}}</style><details id=i{idx}><summary>{source.get('text_snippet', 'N/A')[:300]}</summary>{source.get('text_snippet', 'N/A')[300:]}</details>")
                    else:
                        st.write("Brak źródeł dla tej odpowiedzi.")
                st.markdown("---")
        else:
            st.write("Brak historii czatu dla tej sesji.")

def ollama_management():
    st.header("Zarządzanie modelami Ollama")
    
    model_name = st.text_input("Nazwa modelu do pobrania (np. llama3.1, SpeakLeash/bielik-11b-v2.3-instruct:Q4_K_M)")
    
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
    )
    
    pg_password = st.text_input(
        "PG Bielik - Hasło",
        value=current_settings.get('pg_password', ''),
        type="password"
    )
    
    if st.button("Zapisz Ustawienia"):
        settings = {
            'openai_api_key': openai_key,
            'ollama_url': ollama_url,
            'pg_url': pg_url,
            'pg_username': pg_username,
            'pg_password': pg_password
        }
        save_settings(settings)
        st.success("Ustawienia zostały zapisane!")

    st.divider()
    st.subheader("Zarządzanie bazą danych")
    
    if st.button("Wyczyść wszystkie załadowane dokumenty", type="primary"):
        if st.session_state.current_session:
            with st.spinner("Trwa czyszczenie ChromaDB..."):
                success, message = clear_all_chroma()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.error("Najpierw wybierz lub utwórz sesję!")
            
def brawl():
    col1, col2 = st.columns(2)
    with col1:
        provider1 = st.selectbox("Dostawca 1", get_available_providers())
        if provider1 == "OpenAI":
            model1 = st.selectbox("Model 1", ["OpenAI GPT-4o"])
        elif provider1 == "Ollama":
            installed_models = get_installed_ollama_models()
            if not installed_models:
                st.error("Brak modeli Ollama. Upewnij się, że Ollama działa i ma zainstalowane modele.")
                return
            model1 = st.selectbox("Model 1", installed_models)
        elif provider1 == "PG":
            model1 = st.selectbox("Model 1", ["speakleash/Bielik-11B-v2.2-Instruct"])
    with col2:
        provider2 = st.selectbox("Dostawca 2", get_available_providers())
        if provider2 == "OpenAI":
            model2 = st.selectbox("Model 2", ["OpenAI GPT-4o"])
        elif provider2 == "Ollama":
            installed_models = get_installed_ollama_models()
            if not installed_models:
                st.error("Brak modeli Ollama. Upewnij się, że Ollama działa i ma zainstalowane modele.")
                return
            model2 = st.selectbox("Model 2", installed_models)
        elif provider2 == "PG":
            model2 = st.selectbox("Model 2", ["speakleash/Bielik-11B-v2.2-Instruct"])
    query_text = st.text_input("Wprowadź pytanie")
    lang = st.radio("Język", ["Polski", "Angielski"], horizontal=True)
    question_limit = st.number_input("Limit pytań", min_value=1, max_value=10, value=1)
    with st.spinner("Pobieranie dostępnych dokumentów..."):
        available_docs1 = get_available_collections(provider1, st.session_state.current_session)
        available_docs2 = get_available_collections(provider2, st.session_state.current_session)
        intersection = [(doc_name, count) for doc_name, count in available_docs1 if doc_name in [doc_name for doc_name, _ in available_docs2]]
        docs = [doc_name for doc_name, _ in intersection]
        counts = {}
        for doc_name, count in intersection:
            counts[doc_name] = count
    selected_doc = st.selectbox("Wybierz dokument", docs)

    with col1:
        chroma_k = st.number_input(
            "Fragmenty pobrane z ChromaDB",
            min_value=1,
            max_value=20 if selected_doc is None else round(counts[selected_doc]/2),
            value=10 if selected_doc is None else round(counts[selected_doc]/10)
        )
    with col2:
        rerank_k = st.number_input(
            "Fragmenty pobrane z rerankingu",
            min_value=1,
            max_value=round(chroma_k/2),
            value=min(15, round(chroma_k/4))
        )

    if not st.session_state.loaded[provider1] or not st.session_state.loaded[provider2]:
        st.write("Żaden dokument nie został załadowany do bazy danych dla jednego z dostawców. Proszę załadować dokumenty w zakładce 'Dokumenty'.")
        return

    if st.button("Rozpocznij bójkę") and query_text:
        st.session_state.chroma_k = chroma_k
        st.session_state.rerank_k = rerank_k
        st.write(f"Rozpoczynamy bójkę między {model1} ({provider1}) a {model2} ({provider2})!")

        col1, col2 = st.columns(2)
        
        question = query_text
        for i in range(question_limit):
            idx = i + 1
            with col1:
                st.write(f"**Q{idx}:** {question}")
                with st.spinner(f"PGenerowanie odpowiedzi na pytanie {idx}. od {model2} ({provider2})..."):
                    response1, sources1 = query_rag(question, provider1, model1, lang, selected_doc)
                    st.write(f"**A{idx} ({provider1}/{model1}):** {response1}")
                with st.spinner(f"Generowanie pytania dla {model2} ({provider2})..."):
                    question1, sources1 = query_rag("null", provider1, model1, lang, selected_doc, brawl=True)
                    st.write(f"**F{idx}:** {question1}")
                    match = re.search(r"<question>(.*?)</question>", question1)
                    question = match.group(1) if match else question1
            with col2:
                st.write(f"**Q{idx}:** {question}")
                with st.spinner(f"PGenerowanie odpowiedzi na pytanie {idx}. od {model1} ({provider1})..."):
                    response2, sources2 = query_rag(question, provider2, model2, lang, selected_doc)
                    st.write(f"**A{idx} ({provider2}/{model2}):** {response2}")
                if i < question_limit - 1:
                    with st.spinner(f"Generowanie pytania dla {model1} ({provider1})..."):
                        question2, sources2 = query_rag("null", provider2, model2, lang, selected_doc, brawl=True)
                        st.write(f"**F{idx}:** {question2}")
                        match = re.search(r"<question>(.*?)</question>", question2)
                        question = match.group(1) if match else question2

pg = st.navigation([
    st.Page(settings_page, title="Ustawienia"),
    st.Page(ollama_management, title="Ollama"),
    st.Page(manage_sessions, title="Sesje"), 
    st.Page(upload_files, title="Dokumenty"), 
    st.Page(query_database, title="Czat"),
    st.Page(brawl, title="Arena")
])

pg.run()
if st.session_state.get('loaded') is None:
    st.session_state.loaded = {"PG": False, "Ollama": False, "OpenAI": False}