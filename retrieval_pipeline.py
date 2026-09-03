from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

persistent_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="embeddinggemma")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata=({"hnsw:space": "cosine"}),
)

# search for revelant documents
query = "What was the original name of Microsoft before it became Microsoft?"

# retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# combine the query and the revelant documents content
combine_input = f"""Based on the following documents, answer the question: {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear answer based on the above documents. If you can't find the answer in the documents, please respond with "I don't have enough information to answer that question."

"""

# create a ollama model
model = ChatOllama(model="llama3")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combine_input)
]

# Invoke the model with the combined input
result = model.invoke(messages)


# Display the full result and content only
print("\n ---Generated Response ---")
print("\n ---Content Only ---")
print(result.content)


# Synthetic Questions:

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
