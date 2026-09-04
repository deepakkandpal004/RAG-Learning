from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

# connect to your document database
persistent_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="embeddinggemma")
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# set up AI model
model = ChatOllama(model="llama3")

# store our conversation history
conversation_history = []

def ask_question(user_question):
    print(f"\n you asked: {user_question}")

    # step 1: Make the question clear using conversation history
    if conversation_history:
        # Ask AI to make the question standalone
        messages = (
            [
                SystemMessage(
                    content="Given the conversation history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                ),
            ]
            + conversation_history
            + [HumanMessage(content=f"New question: {user_question}")]
        )

        result = model.invoke(messages)
        search_question = str(result.content).strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # step 2: Find relevant documents
    relevant_docs = db.as_retriever(search_kwargs={"k": 3})
    docs = relevant_docs.invoke(search_question)

    print(f"Found {len(docs)} relevant documents")
    for i, doc in enumerate(docs):
        # show first two lines of each document
        lines = doc.page_content.split("\n")
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    # step 3: Create final prompt
    combined_output = f"""Based on the following documents, please answer the question: {user_question}\n\n

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """

    # step 4: Get the answer
    messages = (
        [
            SystemMessage(
                content="You are a helpful assistant that answers questions based on provided documents and conversation history."
            ),
        ]
        + conversation_history
        + [HumanMessage(content=combined_output)]
    )

    result = model.invoke(messages)
    answer = str(result.content)

    # step 5: Remember this conversation
    conversation_history.append(HumanMessage(content=user_question))
    conversation_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

# simple chat loop
def start_chat():
    print("Ask me questions! Type quit to exit.")

    while True:
        question = input("\n Your question:")

        if question.lower() =='quit':
         print("Goodbye!")
         break

        ask_question(question)

if __name__ == "__main__":
    start_chat()
