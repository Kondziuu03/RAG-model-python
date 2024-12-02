import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)

PROMPT_TEMPLATE = """
Use the following pieces of information to asnwer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Just provide a right answer without additional information.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query: str):
    reranked_results = get_reranked_documents(query)

    top_results = reranked_results[:10]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in top_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def get_reranked_documents(query: str):
    initial_results = get_similar_documents(query);
    
    query_doc_pairs = [(query, doc.page_content) for doc, _ in initial_results]
    scores = reranker.predict(query_doc_pairs)

    # Sort by the re-ranked scores (descending)
    reranked_results = sorted(zip([doc for doc, _ in initial_results], scores),
                              key=lambda x: x[1], reverse=True)
        
    return [(doc, score) for doc, score in reranked_results]

def get_similar_documents(query: str):
     db = Chroma(persist_directory=CHROMA_PATH,embedding_function=get_embedding_function())
     return db.similarity_search_with_score(query, k=100)

if __name__ == "__main__":
    main()
