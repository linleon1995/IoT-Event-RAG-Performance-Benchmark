from datetime import datetime
import networkx as nx
from typing import List, Dict, Any

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

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
            "video_link": e['video'],
            "event_id": f"event_{idx}" # Add a unique ID for graph nodes
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


# --- NEW: Graph Construction Function ---
def build_event_graph(events: List[Dict[str, Any]]) -> nx.Graph:
    """
    Builds a NetworkX graph from a list of event dictionaries.
    Nodes represent events, devices, locations, and event types.
    Edges represent relationships between them.
    """
    G = nx.Graph()

    for idx, event in enumerate(events):
        event_id = f"event_{idx}"
        
        # Add the event node with its details
        G.add_node(event_id,
                   type="event",
                   time=event['time'],
                   event_name=event['event_name'],
                   video_link=event['video'],
                   device_name=event['device_name'], # Store directly for easier access
                   location=event['location'],       # Store directly for easier access
                   full_text=f"At {event['time']}, the device '{event['device_name']}' located at '{event['location']}' detected the event: '{event['event_name']}'. Video evidence: {event['video']}.",
                   metadata={
                        "time": event['time'],
                        "location": event['location'],
                        "event_name": event['event_name'],
                        "device_name": event['device_name'],
                        "video_link": event['video']
                   })

        # Add device node and its relationship to the event
        device_node_name = f"device_{event['device_name'].replace(' ', '_').lower()}"
        if not G.has_node(device_node_name):
            G.add_node(device_node_name, type="device", name=event['device_name'])
        G.add_edge(event_id, device_node_name, relation="detected_by")

        # Add location node and its relationship to the event
        location_node_name = f"location_{event['location'].replace(' ', '_').lower()}"
        if not G.has_node(location_node_name):
            G.add_node(location_node_name, type="location", name=event['location'])
        G.add_edge(event_id, location_node_name, relation="occurred_at")

        # Add event type node and its relationship to the event
        event_type_node_name = f"event_type_{event['event_name'].replace(' ', '_').lower()}"
        if not G.has_node(event_type_node_name):
            G.add_node(event_type_node_name, type="event_type", name=event['event_name'])
        G.add_edge(event_id, event_type_node_name, relation="is_type_of")
        
        # Add a relationship between device and location if not already present
        # This implies the device is generally located there
        if not G.has_edge(device_node_name, location_node_name):
             G.add_edge(device_node_name, location_node_name, relation="located_at")


    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- NEW: Custom Graph Retriever ---
class CustomGraphRetriever(BaseRetriever):
    """
    A LangChain retriever that uses a NetworkX graph for retrieval,
    with a FAISS vector store as a fallback or for hybrid search.
    """
    graph: nx.Graph
    faiss_vectorstore: FAISS
    
    # Optional: For more sophisticated entity extraction, you might pass an LLM or NER model here
    # llm_for_entity_extraction: Any = None 

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Retrieve documents by traversing the graph based on identified entities
        or falling back to vector search.
        """
        relevant_event_ids = set()
        retrieved_docs: List[Document] = []

        query_lower = query.lower()

        # --- Entity Extraction (Basic Keyword Matching) ---
        # In a real system, use an LLM or a dedicated NER model here.
        # For demonstration, we'll do simple keyword matching against known entities.
        
        found_entities = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type')
            node_name = data.get('name', '').lower()

            if node_type == 'device' and node_name in query_lower:
                found_entities.setdefault('device', []).append(node)
                print(f"Detected device in query: {data['name']}")
            elif node_type == 'location' and node_name in query_lower:
                found_entities.setdefault('location', []).append(node)
                print(f"Detected location in query: {data['name']}")
            elif node_type == 'event_type' and node_name in query_lower:
                found_entities.setdefault('event_type', []).append(node)
                print(f"Detected event type in query: {data['name']}")

        # --- Graph Traversal ---
        if found_entities:
            # Start with the directly found entities
            nodes_to_traverse = set()
            for entity_list in found_entities.values():
                nodes_to_traverse.update(entity_list)

            # Traverse neighbors to find connected events
            for start_node in list(nodes_to_traverse): # Iterate over a copy
                # Add immediate neighbors (depth 1)
                for neighbor in self.graph.neighbors(start_node):
                    if self.graph.nodes[neighbor].get('type') == 'event':
                        relevant_event_ids.add(neighbor)
                    nodes_to_traverse.add(neighbor) # Also consider neighbors for deeper traversal

            # For events, add their related device/location/event_type for context
            # And potentially more distant events that are "related" through shared entities
            # Let's do a 2-hop traversal from initial entities for broader context
            processed_nodes = set()
            queue = list(nodes_to_traverse) # Start with all found entities and their immediate neighbors

            while queue:
                current_node = queue.pop(0)
                if current_node in processed_nodes:
                    continue
                processed_nodes.add(current_node)

                node_data = self.graph.nodes[current_node]
                if node_data.get('type') == 'event':
                    relevant_event_ids.add(current_node)
                
                # Add neighbors to the queue for next iteration (up to 2-hops for this example)
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in processed_nodes:
                        queue.append(neighbor)
            
            print(f"Relevant event IDs identified through graph traversal: {len(relevant_event_ids)}")
            for event_id in relevant_event_ids:
                event_data = self.graph.nodes[event_id]
                # Ensure it's an event node and has full_text
                if event_data.get('type') == 'event' and 'full_text' in event_data:
                    doc_metadata = event_data.get('metadata', {}) # Use the stored metadata
                    retrieved_docs.append(Document(page_content=event_data['full_text'], metadata=doc_metadata))
            
            # If graph traversal found documents, prioritize them
            if retrieved_docs:
                print(f"Graph-based retrieval found {len(retrieved_docs)} documents.")
                return retrieved_docs
        
        # --- Fallback to FAISS Vector Search if no relevant entities or graph results ---
        print("Falling back to FAISS vector search.")
        # You can adjust 'k' here for the fallback search
        faiss_retriever = self.faiss_vectorstore.as_retriever(search_kwargs={'k': 5})
        return faiss_retriever._get_relevant_documents(query, run_manager=run_manager)

    async def _aget_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Implement async version if needed, or just call the sync version
        return self._get_relevant_documents(query, run_manager=run_manager)


def main():
    print("🔧 準備事件資料與 LLM...")
    events = generate_events(30) # Assuming generate_events provides events with time, device_name, location, event_name, video

    # Sort events by time
    try:
        events.sort(key=lambda x: datetime.strptime(x['time'], '%Y-%m-%d %H:%M'))
        print("✅ 事件已按時間排序。")
    except KeyError:
        print("⚠️ 警告: 事件字典中沒有 'time' 鍵，無法排序。")
    except ValueError as ve:
        print(f"⚠️ 警告: 'time' 格式不正確，無法排序。錯誤: {ve}")

    # Optional: print sorted events (uncomment if needed for debugging)
    for i, e in enumerate(events):
        print(f"事件 {i+1}: {e}") 

    docs = make_documents(events)
    
    # --- Graph Creation ---
    event_graph = build_event_graph(events)

    # --- Embedding and FAISS Vector Store ---
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding)

    # --- Load LLM ---
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
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # --- Instantiate Custom Graph Retriever ---
    graph_retriever = CustomGraphRetriever(
        graph=event_graph,
        faiss_vectorstore=vectorstore
    )

    # --- Set up RetrievalQA Chain with the Custom Retriever ---
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=graph_retriever, # Use the custom graph retriever here
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True 
    )

    print("\n🧠 啟動成功！請輸入自然語言問題（輸入 `exit` 結束）")

    while True:
        query = input("\n❓ 你的問題： ")
        if query.strip().lower() in ["exit", "quit", "q"]:
            print("👋 結束對話，感謝使用！")
            break
        
        result = qa.invoke({"query": query})
        print(f"\n📣 回答：\n{result['result']}")
        print(f"\n📚 參考資料：\n{[doc.metadata for doc in result['source_documents']]}")

if __name__ == "__main__":
    main()