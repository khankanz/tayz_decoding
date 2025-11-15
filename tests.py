from typing import Literal
from pydantic import BaseModel, Field
from tayz_decoding.core import Llama

from pathlib import Path
tl1B_path=str(Path("/home/zardar/Downloads/tinyllama-1.1b-chat-v1.0.Q2_K.gguf")) ; tl1B_path
llm = Llama(model_path=tl1B_path)

class HistologicType(BaseModel):
    histologic_type: Literal["Keritinzation", "Squamous Cell Carcinoma"]

messages = [
    {"role": "system", "content": "You are a pathology assistant."},
    {"role": "user", "content": "Classify the histologic type from the following report: Microscopic examination reveals nests of atypical squamous cells with keratinization, diagnostic of squamous cell carcinoma."}
]

print("--- Starting CRANE completion ---")
try:
    response = llm.create_crane_chat_completion(
        messages=messages,
        schema=HistologicType,
        s1="<<JSON>>",
        s2="</JSON>>",
        temperature=0.0,
        max_tokens_unconstrained=512,
        max_tokens_constrained=256
    )
    print("--- CRANE completion successful ---")
    print(response)
except Exception as e:
    print(f"--- Caught Python exception: {e} ---")

print("--- Script finished ---")