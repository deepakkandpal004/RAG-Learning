import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def load_documents(docs_path="docs"):
    """Load all the text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")

    # check if the docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The Directory {docs_path} does not exist. Please create it and add your documents."
        )

    # load all .txt files from the docs directory
    loader = DirectoryLoader(path=docs_path, glob="*.txt", loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            f"No .txt files found in {docs_path}. Please add some .txt files."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i + 1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content Length: {len(doc.page_content)} characters")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split the documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("_" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating Embedding and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="embeddinggemma")

    # create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("--- Finishing creating Vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main Function")

    # 1. Loading the files
    documents = load_documents(docs_path="docs")

    # 2. Chunking the files
    chunks = split_documents(documents)

    # 3. Embedding and Storing in VectorDB
    create_vector_store(chunks)


if __name__ == "__main__":
    main()


# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]
