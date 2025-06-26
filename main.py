from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from events import generate_events, make_documents
from llama_config import load_llm

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM = "gemma-3-4b-it"


def load_and_categorize_questions(filepath):
    """
    Load questions from a text file and organize them by category.
    Returns a dict: {category: [questions]}
    """
    categories = {}
    current_category = None
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---'):
                continue
            # Detect category headers (e.g., 'Reasoning & Inference')
            if not line[0].isdigit() and not line.startswith('List') and not line.endswith('?') and not line.endswith('.'):  # crude but works for this file
                current_category = line.replace(':', '').strip()
                if current_category:
                    categories[current_category] = []
                continue
            # Detect numbered questions
            if line and (line[0].isdigit() and '.' in line):
                if current_category:
                    categories[current_category].append(line)
                continue
            # Detect unnumbered questions (e.g., 'List all the instances ...')
            if current_category and (line.endswith('?') or line.endswith('.')):
                categories[current_category].append(line)
    return categories


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

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(docs, embedding)

    # llm = load_llm(model_path='gemma-2b/gemma-2b.gguf', framework='llama_cpp')
    llm, tokenizer = load_llm(model_path=LLM, framework='safetensors')

    prompt_template = """
    As a professional surveillance system analyst, your task is to carefully analyze and synthesize the information from the event records provided below to answer the user's question.

    Your answer must be based on a logical combination of the facts in the records. If the necessary facts to form a conclusion are not present, then and only then should you state: "Based on the provided data, I cannot answer this question."

    [EVENT RECORDS]
    {context}

    [QUESTION]
    {question}

    [ANSWER]
    """
    prompt_template = """
    You are an expert data assistant that specializes in generating SQL queries from natural language questions. 

    The data comes from IoT event logs with the following schema:

    - time (string in format "YYYY-MM-DD HH:MM")
    - location (string, e.g., "factory_1", "warehouse_3")
    - event_name (string, e.g., "fire_detected", "door_opened")
    - device_name (string, e.g., "camera_A", "sensor_5")
    - video (string, e.g., "video_42.mp4")

    Your job is to convert a user’s question into a valid SQL query using this schema. Do not include any explanation. Assume the table is called `iot_events`. Always format the SQL query using standard SQL syntax.
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



# Example usage:
if __name__ == "__main__":
    main()
    # question_categories = load_and_categorize_questions("llm_eval_questions.txt")
    # print("\nLoaded and categorized questions:")
    # for cat, qs in question_categories.items():
    #     print(f"\n[{cat}]")
    #     for q in qs:
    #         print(f"- {q}")
