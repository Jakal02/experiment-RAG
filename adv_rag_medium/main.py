from pathlib import Path
import logging

import torch

from llama_index.core.prompts import PromptTemplate
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.file import PDFReader

from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings

from transformers import BitsAndBytesConfig

from config import creds

logging.basicConfig()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Document
logging.info("Loading Document")
loader = PDFReader()
docs = loader.load_data(file=Path("./data/QLoRa.pdf"))

# Parse Documents
logging.info("Parsing Document")
doc_text = "\n\n".join([d.get_content() for d in docs])
documents = [Document(text=doc_text)]

# Chunk Document
logging.info("Chunk Document")
node_parser = SentenceSplitter(chunk_size=1024)
base_nodes = node_parser.get_nodes_from_documents(documents)

##############
# Models
##############


# huggingface api token for downloading llama2
hf_token = creds.HUGGING_FACE_API_TOKEN

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


logging.info("Define LLM")
llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-alpha",
    tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

logging.info("Define Embedding Model")
# Embedding 
embed_model = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
)
# # set your ServiceContext for all the next steps
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model
)

####
# Baseline Retriever
####
logging.info("Baseline Retriever")
base_index = VectorStoreIndex(base_nodes, service_context=service_context)
base_retriever = base_index.as_retriever(similarity_top_k=2)
retrievals = base_retriever.retrieve(
    "Can you tell me about the Paged Optimizers?"
)

for n in retrievals:
    print("Node ID:", n.node.node_id)
    print("Similarity:", n.score)
    print("Text:", n.node.get_content(metadata_mode=MetadataMode.NONE).strip()[1500])
