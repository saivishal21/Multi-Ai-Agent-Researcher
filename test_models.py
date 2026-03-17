import os
os.environ["OPENAI_API_KEY"] = "lm-studio"

from crewai import LLM

models = {
    "Mistral": "openai/mistralai/mistral-7b-instruct-v0.3",
    "Gemma":   "openai/google/gemma-3-4b",
    "Llama":   "openai/llama-3.2-3b-instruct",
}

for name, model_id in models.items():
    print(f"\n{'='*40}")
    print(f"Testing {name}...")
    try:
        llm = LLM(
            model=model_id,
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )
        response = llm.call([{"role": "user", "content": f"Say exactly: I am {name} and I am working!"}])
        print(f"✅ {name} response: {response}")
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
