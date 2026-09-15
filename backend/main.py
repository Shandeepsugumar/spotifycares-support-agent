import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

if "GROQ_MODEL" not in os.environ:
    os.environ["GROQ_MODEL"] = "qwen/qwen3.8-27b"

current_file_path = os.path.abspath(__file__)
backend_dir = os.path.dirname(current_file_path)
repo_root_dir = os.path.dirname(backend_dir)
sys.path.append(os.path.join(repo_root_dir, "src"))

from agent import load_fewshot_examples, build_prompt, call_groq, validate_output, apply_policy_override
from retrieval import Retriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
retriever = None
fewshot_examples = None

@app.on_event("startup")
def startup_event():
    global client, retriever, fewshot_examples
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY environment variable not set!")
    
    client = Groq(api_key=api_key)
    
    # Resolve paths reliably from backend/
    base_dir = repo_root_dir
    retriever_path = os.path.join(base_dir, "historical_retrieval_corpus_LEAKAGE_SAFE.csv")
    fewshot_path = os.path.join(base_dir, "data", "fewshot_pool_FINAL.csv")
    
    print(f"Loading retriever from {retriever_path}...")
    retriever = Retriever(corpus_path=retriever_path)
    
    print(f"Loading few-shot pool from {fewshot_path}...")
    fewshot_examples = load_fewshot_examples(path=fewshot_path)
    print("Backend ready!")

class ClassifyRequest(BaseModel):
    message: str

@app.post("/classify")
def classify(req: ClassifyRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        # Retrieve evidence
        evidence_records = retriever.retrieve(req.message, k=4)
        
        # Build prompt
        prompt = build_prompt(fewshot_examples, req.message, evidence_records)
        
        # Call Groq
        raw_output, call_err = call_groq(client, prompt)

            
        # Validate output schema
        if call_err:
            if "429" in call_err or "quota" in call_err.lower():
                raise HTTPException(status_code=429, detail="Rate limit or quota exceeded.")
            raise HTTPException(status_code=502, detail=f"LLM Error: {call_err}")
            
        validated, err = validate_output(raw_output, evidence_records)
        if err is not None:
            raise HTTPException(status_code=422, detail=f"Agent schema validation failed: {err}")
            
        validated = apply_policy_override(req.message, validated)
            
        return {
            "intent": validated["intent"],
            "draft_reply": validated["draft_reply"],
            "action": validated["action"],
            "reason": validated["reason"],
            "evidence_ids": validated["evidence_ids"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)










