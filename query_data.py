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

Answer the question and provide additional helpful information,
based on the pieces of information, if applicable. Be succinct.

Responses should be properly formatted to be easily read.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    initial_results = db.similarity_search_with_score(query_text, k=10)

    print("\n--- Initial Results ---")
    for i, (doc, score) in enumerate(initial_results):
        print(f"Result {i + 1}:")
        print(f"  Score: {score}")
        print(f"  Source: {doc.metadata.get("id", None)}...")
        print(f"  Document Content: {doc.page_content}...")
        print()

    reranked_results = rerank_documents(query_text, initial_results)
    #reranked_results = initial_results
    print("\n--- Re-ranked Results ---")
    for i, (doc, score) in enumerate(reranked_results[:3]):
        print(f"Re-ranked Result {i + 1}:")
        print(f"  Score: {score}")
        print(f"  Source: {doc.metadata.get("id", None)}...")
        print(f"  Document Content: {doc.page_content}...")
        print()

    top_results = reranked_results[:3]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in top_results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in top_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def rerank_documents(query: str, results):
    query_doc_pairs = [(query, doc.page_content) for doc, _ in results]
    scores = reranker.predict(query_doc_pairs)

    # Sort by the re-ranked scores (descending)
    reranked_results = sorted(zip([doc for doc, _ in results], scores), key=lambda x: x[1], reverse=True)
        
    return [(doc, score) for doc, score in reranked_results]

if __name__ == "__main__":
    main()
