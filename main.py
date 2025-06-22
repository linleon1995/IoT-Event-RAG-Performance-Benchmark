from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from events import generate_events
from llama_config import load_llm


def make_documents(events):
    docs = []
    for idx, e in enumerate(events):
        # Determine ordinal suffix for idx+1
        n = idx + 1
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        ordinal = f"{n}{suffix}"

        content = (
            f"This is the {ordinal} event. "
            f"At {e['time']}, the device '{e['device_name']}' located at '{e['location']}' "
            f"detected the event: '{e['event_name']}'. "
            f"Video evidence is available at: {e['video']}."
        )
        metadata = {
            "time": e['time'],
            "location": e['location'],
            "event_name": e['event_name'],
            "device_name": e['device_name'],
            "video_link": e['video'] # 鍵名可以取得更有意義
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def main():
    print("🔧 準備事件資料與 LLM...")
    events = generate_events(30)
    try:
        events.sort(key=lambda x: datetime.strptime(x['time'], '%Y-%m-%d %H:%M'))
        print("✅ 事件已按時間排序。")
    except KeyError:
        print("⚠️ 警告: 事件字典中沒有 'time' 鍵，無法排序。")
    except ValueError as ve:
        print(f"⚠️ 警告: 'time' 格式不正確，無法排序。錯誤: {ve}")
    # --- 排序邏輯結束 ---

    for i, e in enumerate(events):
        print(f"事件 {i+1}: {e}") # 排序後再印出，確認順序
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

    # 在建立 QA Chain 時傳入 prompt
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr", # 或者 "similarity"
            search_kwargs={'k': 30, 'fetch_k': 40} # <--- 增加 k 和 fetch_k
        ),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True # 建議開啟，方便除錯
    )

    print("\n🧠 啟動成功！請輸入自然語言問題（輸入 `exit` 結束）")

    while True:
        query = input("\n❓ 你的問題： ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            print("👋 結束對話，感謝使用！")
            break
        
        # answer = qa.run(query)
        # print(f"\n📣 回答：\n{answer}")
        result = qa.invoke({"query": query})
        print(f"\n📣 回答：\n{result['result']}")
        print(f"\n📚 參考資料：\n{[doc.metadata for doc in result['source_documents']]}")

if __name__ == "__main__":
    main()
