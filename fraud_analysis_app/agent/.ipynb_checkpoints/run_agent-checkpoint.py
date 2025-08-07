from gpt4all import GPT4All

def run_agent(mlflow_metrics: str, arize_metrics: str, prompt_template: str):
    print("📥 Loading prompt...")
    with open(prompt_template, "r") as f:
        template = f.read()

    prompt = template.replace("{{ mlflow }}", mlflow_metrics).replace("{{ arize }}", arize_metrics)

    print("📦 Initializing model...")
    model = GPT4All(
        model_name="Llama-3.2-3B-Instruct-Q4_0.gguf",
       model_path="agent/.gpt4all/models",
        allow_download=False
    )

    print("💬 Running LLM agent...")
    with model.chat_session() as session:
        response = session.generate(prompt)
        print("✅ Response received")
    return response
