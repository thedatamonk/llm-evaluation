from typing import List, Dict, Any, Literal
from pydantic.json_schema import JsonSchemaValue
import ollama

# Ollama LLM call
DEFAULT_OLLAMA_MODEL = "llama3:instruct"

def llm_call(messages: List[Dict[str, Any]],
             model_name: str = DEFAULT_OLLAMA_MODEL,
             stream: bool = False,
             response_format: JsonSchemaValue | Literal['', 'json'] | None = None) -> str:
    

    try:
        if stream:
            full_response_content = []
            for chunk in ollama.chat(
                model=model_name, messages=messages, stream=True, format=response_format
            ):
                if chunk.get('done'):
                    break
  
                content = chunk["message"]["content"]
                print (content, end="", flush=True)
                full_response_content.append(content)
            
            return "".join(full_response_content)
        else:
            response = ollama.chat(model=model_name, messages=messages, format=response_format)
            return response["message"]["content"]
    
    except ollama.ResponseError as e:
        print (f"Error interacting with Ollama: {e}")
        print (
            f"Ensure Ollama server is running and the model "
            f"'{model_name}' is dowloaded (`ollama pull {model_name}`)."
        )
        return f"Error: {e}"
    

if __name__ == "__main__":
    response = llm_call(messages=[{
        "role": "user",
        "content": "Who is the PM of India? And is he older than the current president of USA?"
    }], stream=False)

    print (response)