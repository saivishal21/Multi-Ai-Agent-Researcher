import os
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from crewai import Agent, Task, Crew, LLM

mistral_llm = LLM(
    model="openai/mistralai/mistral-7b-instruct-v0.3",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

gemma_llm = LLM(
    model="openai/google/gemma-3-4b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

llama_llm = LLM(
    model="openai/llama-3.2-3b-instruct",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

# --- AGENTS ---

researcher = Agent(
    role="Researcher",
    goal="Research the given topic thoroughly and gather key facts and insights",
    backstory="You are an expert researcher who finds accurate and relevant information",
    llm=mistral_llm,  # Best model for research
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write an engaging and well structured blog post based on the research",
    backstory="You are a skilled content writer who turns research into compelling articles",
    llm=gemma_llm,  # Good at creative writing
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Polish and improve the written content for clarity grammar and impact",
    backstory="You are a professional editor with an eye for detail and quality",
    llm=llama_llm,  # Fast and efficient for editing
    verbose=True
)

# --- TASKS ---

topic = input("Enter a topic to research and write about: ")

research_task = Task(
    description=f"Research the topic: {topic}. Gather key facts, trends, statistics and insights.",
    expected_output="A detailed research summary with bullet points of key findings",
    agent=researcher
)

write_task = Task(
    description="Using the research provided, write a full blog post with introduction, body and conclusion.",
    expected_output="A complete 500 word blog post",
    agent=writer
)

edit_task = Task(
    description="Review and edit the blog post for grammar, clarity, flow and overall quality.",
    expected_output="A polished final version of the blog post ready to publish",
    agent=editor
)

# --- CREW ---

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    verbose=True
)

# --- RUN ---

result = crew.kickoff()
print("\n========== FINAL OUTPUT ==========\n")
print(result)
with open("output.txt", "w") as f:
    f.write(str(result))
print("Saved to output.txt!")