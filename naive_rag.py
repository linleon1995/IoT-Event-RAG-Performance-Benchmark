from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from events import generate_events, make_documents
from llama_config import load_llm


def main():
    print("🔧 Preparing event data and LLM ...")
    events = generate_events(30)
    try:
        events.sort(key=lambda x: datetime.strptime(x['time'], '%Y-%m-%d %H:%M'))
        print("✅ Events sorted by time.")
    except KeyError:
        print("⚠️ Warning: No 'time' key in event dictionary, cannot sort.")
    except ValueError as ve:
        print(f"⚠️ Warning: 'time' format incorrect, cannot sort. Error: {ve}")
    # --- End of sorting logic ---

    for i, e in enumerate(events):
        print(f"Event {i+1}: {e}") # Print after sorting to confirm order
    docs = make_documents(events)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)

    # llm = load_llm(model_path='gemma-2b/gemma-2b.gguf', framework='llama_cpp')
    llm, tokenizer = load_llm(model_path='gemma-3-4b-it', framework='safetensors')

    prompt_template = """
    As a professional surveillance system analyst, your task is to carefully analyze and synthesize the information from the event records provided below to answer the user's question.

    Your answer must be based on a logical combination of the facts in the records. If the necessary facts to form a conclusion are not present, then and only then should you state: "Based on the provided data, I cannot answer this question."

    [EVENT RECORDS]
    {context}

    [QUESTION]
    {question}

    [ANSWER]
    """
    # qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Pass the prompt when creating the QA Chain
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr", # or "similarity"
            search_kwargs={'k': 30, 'fetch_k': 40} # <--- Increase k and fetch_k
        ),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True # Recommended to enable for easier debugging
    )

    print("\n🧠 System started! Please enter a natural language question (type `exit` to quit)")

    while True:
        query = input("\n❓ Your question: ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            print("👋 Ending conversation, thank you for using!")
            break
        
        # answer = qa.run(query)
        # print(f"\n📣 Answer:\n{answer}")
        result = qa.invoke({"query": query})
        print(f"\n📣 Answer:\n{result['result']}")
        print(f"\n📚 Reference documents:\n{[doc.metadata for doc in result['source_documents']]}")


if __name__ == "__main__":
    main()
