import torch
from langchain.llms import LlamaCpp
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_llm(model_path, framework):
    if framework == "llama_cpp":
        return LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            temperature=0.7,
            verbose=True
        )
    elif framework == "safetensors":
        # 載入分詞器
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 載入模型
        # device_map="auto" 會自動將模型層分佈到可用的設備上 (GPU 或 CPU)
        # torch_dtype=torch.bfloat16 (或 torch.float16) 可以節省記憶體並加速
        # quantization_config (例如 load_in_4bit=True) 可以進一步減少記憶體佔用
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16, # 或 torch.float16
            # load_in_4bit=True, # 啟用 4-bit 量化，需要 bitsandbytes
            # quantization_config=BitsAndBytesConfig(load_in_4bit=True) # 更詳細的量化配置
        )

        # 創建 Hugging Face Pipeline
        # max_new_tokens: 每次生成允許的最大 token 數
        # temperature: 控制生成文本的隨機性
        # do_sample: 啟用採樣生成
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048, # 與 Langchain Chain 的 max_tokens 保持一致
            temperature=0.7,
            do_sample=True,
            torch_dtype=torch.bfloat16, # 匹配模型的 dtype
            device_map="auto", # 確保 pipeline 也利用 GPU
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm, tokenizer
    else:
        raise ValueError("Unsupported framework: {}".format(framework))