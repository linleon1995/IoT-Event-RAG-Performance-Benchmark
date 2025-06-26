# IoT Event RAG System

## Introduction

This repository provides a framework for benchmarking Retrieval-Augmented Generation (RAG) system performance on Internet of Things (IoT) event data. We explore and compare two distinct RAG architectures: a traditional vector-similarity-based RAG and a more advanced Graph RAG approach. Our primary objective is to evaluate how effectively these systems leverage a Large Language Model (LLM) – specifically the **Gemma 3 (12B parameter model)** – to understand, reason, summarize, and retrieve relevant information from complex, time-series IoT event streams.

IoT environments generate vast amounts of event data, often characterized by intricate relationships between devices, locations, and event types. While traditional RAG excels at semantic similarity searches, it often struggles with multi-hop reasoning, temporal queries, and understanding inherent structural connections. Graph RAG addresses these limitations by explicitly modeling entities and their relationships as a knowledge graph, enabling more precise and contextually rich retrieval.

This project aims to demonstrate the strengths and weaknesses of each RAG paradigm in handling typical analytical queries against a simulated IoT event dataset, providing insights into their suitability for real-world IoT monitoring and intelligence applications.

## Installation

To set up and run this project, follow these steps:

### Prerequisites

  * Python 3.9+
  * `pip` (Python package installer)

### 1\. Clone the Repository

```bash
git clone https://github.com/YourGitHubUsername/iot-event-rag-benchmark.git
cd iot-event-rag-benchmark
```

### 2\. Install Dependencies

It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**`requirements.txt` content (create this file if it doesn't exist):**

```
langchain
langchain-community
langchain-core
networkx
pydantic>=2.0 # Ensure Pydantic is up-to-date for LangChain v0.1.x
transformers
torch # Or tensorflow, depending on your PyTorch/TF setup for Gemma
# If using llama_cpp for Gemma:
llama-cpp-python
# If using safetensors for Gemma:
safetensors
```

### 3\. Model Download and Configuration

This project uses the `gemma-3-4b-it` (instruction-tuned) model as specified. You will need to download the model weights and place them in the correct location or configure `llama_config.py` to point to their path.

**For `safetensors` (Hugging Face Transformers):**

The `load_llm` function in `llama_config.py` should handle downloading `gemma-3-4b-it` directly from Hugging Face if you configure it to use the `transformers` framework and specify the model name. Ensure your `llama_config.py` is correctly set up for this:

```python
# llama_config.py (example for transformers/safetensors)
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_path: str, framework: str):
    if framework == 'safetensors':
        # model_path should be the Hugging Face model ID, e.g., 'google/gemma-3-4b-it'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16 # or torch.float16, adjust based on GPU support
        )
        return model, tokenizer
    else:
        raise ValueError(f"Unsupported framework: {framework}")

```

**Note:** The original prompt mentions `gemma-3-12b` but your code uses `gemma-3-4b-it`. Ensure consistency. `gemma-3-12b-it` would require more VRAM. Adjust `model_path` in `main.py` accordingly if you intend to use the 12B version: `model_path='google/gemma-3-12b-it'`.

### 4\. Run the Application

Execute the main script:

```bash
python main.py
```

The application will initialize, load the LLM, build the RAG system (including the graph), and then prompt you for questions.

## Evaluation Questions

We will evaluate the RAG system's performance using a diverse set of questions designed to test various capabilities:

-----

### 🧠 **Reasoning & Inference**

1.  Which locations had both *Fire Alarm* and *Power Outage* events?
2.  Which device was involved in all three types of events: *Fire Alarm*, *Unauthorized Access*, and *Water Leak*?
3.  List all the instances where multiple different event types occurred in the same location on the same day.

-----

### 📅 **Temporal Reasoning**

4.  What was the first event recorded in *Building A*?
5.  What was the 4th event recorded in *Building B*?
6.  What event happened just before the *Fire Alarm* in the Warehouse at 2025-06-09 09:04?
7.  List all events that happened between **2025-06-06 00:00** and **2025-06-08 00:00**.

-----

### 📍 **Location-Based Queries**

8.  How many unique event types were recorded in the *Lobby*?
9.  List all devices that reported events in *Building B*.
10. Which location experienced the most *Unauthorized Access* events?

-----

### 🎥 **Video Metadata Use**

11. Find all video filenames where *Sensor\_A* detected a *Water Leak*.
12. Which event corresponds to `video_13.mp4`?
13. List all video files related to *Drone\_X*.

-----

### 📊 **Aggregation / Counting**

14. How many *Power Outage* events occurred overall?
15. Which device was involved in the most events?
16. Count how many events happened at each location.

-----

### 🔄 **Event Pattern Discovery**

17. Is there a pattern in the times when *Fire Alarm* events are detected (e.g., mostly morning, afternoon)?
18. List events that occurred more than once in the same location *by the same device*.

-----

### 🧪 **LLM Robustness Check (Ambiguity / Natural Language)**

19. “Tell me all times the warehouse had some kind of trouble.” *(Tests generalized filtering)*
20. “What was Drone\_X doing on June 9th?” *(Tests context resolution and device-time matching)*
21. “Show me incidents where Camera\_1 saw a problem in the middle of the night.” *(Tests interpretation of “middle of the night”)*

-----

## Evaluation Results

*(This section is a placeholder for where you will add your findings after running the evaluations. You'd typically add qualitative observations and/or quantitative metrics here.)*

To evaluate, you would systematically run each of the questions against both your traditional RAG setup and your Graph RAG setup (you would need to create a separate script or modify `main.py` to switch between retrievers for a direct comparison).

For each question, consider:

  * **Accuracy:** Was the answer correct and complete based on the provided data?
  * **Relevance:** Was the retrieved context truly relevant to the question?
  * **Completeness:** Did the answer include all necessary details?
  * **Conciseness:** Was the answer to the point without irrelevant information?
  * **Reasoning Capability:** For complex questions (e.g., "Which locations had both..."), did the system correctly infer the answer?
  * **Temporal Accuracy:** For time-based questions, was the ordering and filtering correct?
  * **Robustness to Ambiguity:** How well did it handle the "Robustness Check" questions?

You might present your findings in a table, like this:

| Question \# | Question Text                                              | Traditional RAG Accuracy | Graph RAG Accuracy | Observations                                                                                                                                                                                                                                                                                                    |
| :--------- | :--------------------------------------------------------- | :----------------------- | :----------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1          | Which locations had both *Fire Alarm* and *Power Outage* events? | Low (Missed connections)   | High               | Graph RAG excelled here due to its ability to traverse relationships between event types and locations. Traditional RAG often only retrieved documents for one event type at a time, failing to synthesize the intersection.                                                                                |
| 4          | What was the first event recorded in *Building A*?         | Medium                   | High               | Both might answer correctly, but Graph RAG could potentially leverage temporal ordering stored as graph properties or relations more reliably. Traditional RAG relies on the LLM to sort retrieved text chunks.                                                                                   |
| 19         | “Tell me all times the warehouse had some kind of trouble.” | Low (Too specific)         | Medium (Better, but needs more sophisticated entity extraction) | Traditional RAG struggled with the generalized "trouble," requiring very specific keyword matches. Graph RAG's ability to link "warehouse" to various "event types" (e.g., fire, water leak) improved performance, but an advanced entity extractor would be needed for a perfect "trouble" interpretation. |
| ...        | ...                                                        | ...                      | ...                | ...                                                                                                                                                                                                                                                                                             |

This kind of detailed evaluation will highlight the specific advantages of a Graph RAG approach for structured and interconnected IoT event data.
