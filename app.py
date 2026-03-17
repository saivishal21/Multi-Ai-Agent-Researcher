import os
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

import time
import threading
import streamlit as st
from crewai import Agent, Task, Crew, LLM

st.set_page_config(page_title="AI Content Team", page_icon="🤖")
st.title("🤖 AI Multi-Agent Content Team")
st.write("3 different local LLMs working together — 100% private, no API costs")

col1, col2, col3 = st.columns(3)
col1.info("🔍 Researcher\nMistral 7B")
col2.info("✍️ Writer\nLlama 3.2 3B")
col3.info("🧑 Humanizer\nPhi-3.5 Mini")

topic = st.text_input("Enter a topic:", placeholder="e.g. Future of Electric Cars")

if st.button("🚀 Generate Blog Post"):
    if not topic:
        st.warning("Please enter a topic first!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        timer_text = st.empty()
        agent_status = st.empty()

        # Live output expanders
        st.markdown("---")
        st.markdown("### 📊 Live Agent Outputs")

        researcher_expander = st.expander("🔍 Researcher Output (Mistral 7B)", expanded=False)
        researcher_output = researcher_expander.empty()
        researcher_output.info("⏸️ Waiting to start...")

        writer_expander = st.expander("✍️ Writer Output (Llama 3.2 3B)", expanded=False)
        writer_output = writer_expander.empty()
        writer_output.info("⏸️ Waiting to start...")

        humanizer_expander = st.expander("🧑 Humanizer Output (Phi-3.5 Mini)", expanded=False)
        humanizer_output = humanizer_expander.empty()
        humanizer_output.info("⏸️ Waiting to start...")

        st.markdown("---")

        start_time = time.time()

        # --- LLMs ---
        mistral_llm = LLM(
            model="openai/mistralai/mistral-7b-instruct-v0.3",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            max_tokens=1500,
        )
        llama_llm = LLM(
            model="openai/llama-3.2-3b-instruct",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            max_tokens=1500,
        )
        phi_llm = LLM(
            model="openai/phi-3.5-mini-instruct",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            max_tokens=1500,
        )

        # Shared state for callbacks
        shared = {"research": None, "write": None, "humanize": None}

        def on_research_done(output):
            shared["research"] = str(output.raw)

        def on_write_done(output):
            shared["write"] = str(output.raw)

        def on_humanize_done(output):
            shared["humanize"] = str(output.raw)

        # --- AGENTS ---
        researcher = Agent(
            role="Researcher",
            goal="Research the given topic thoroughly and gather key facts and insights",
            backstory="You are an expert researcher who finds accurate and relevant information",
            llm=mistral_llm,
            verbose=False
        )
        writer = Agent(
            role="Writer",
            goal="Write an engaging and well structured blog post based on the research",
            backstory="You are a skilled content writer who turns research into compelling articles",
            llm=llama_llm,
            verbose=False
        )
        humanizer = Agent(
            role="Humanizer",
            goal="Rewrite the blog post to sound completely human written, natural and conversational",
            backstory="""You are an expert at rewriting AI generated text to sound like it was 
            written by a real human. You use varied sentence lengths, natural transitions, 
            contractions, first person perspective, and personal opinions. You completely avoid 
            robotic AI phrases like 'In conclusion', 'It is worth noting', 'In today's world', 
            'Delve into', 'Furthermore', 'It is important to note'. You write like a real 
            blogger — casual, engaging, and authentic.""",
            llm=phi_llm,
            verbose=False
        )

        # --- TASKS ---
        research_task = Task(
            description=f"Research the topic: {topic}. Gather key facts, trends, statistics and insights.",
            expected_output="A detailed research summary with key findings in bullet points",
            agent=researcher,
            callback=on_research_done
        )
        write_task = Task(
            description="Using the research provided, write a full blog post with introduction, body and conclusion.",
            expected_output="A complete and detailed blog post",
            agent=writer,
            callback=on_write_done
        )
        humanize_task = Task(
            description="""Rewrite the blog post to sound completely human and natural:
            - Use casual conversational language
            - Vary sentence lengths (mix short punchy sentences with longer ones)
            - Add personal opinions and relatable examples
            - Use contractions (don't, it's, we're, you'll)
            - Remove any corporate or AI sounding phrases
            - Make it feel like a real person wrote it
            - Keep all the same facts and information
            - Do NOT add new facts""",
            expected_output="A naturally written human sounding version of the blog post",
            agent=humanizer,
            callback=on_humanize_done
        )

        crew = Crew(
            agents=[researcher, writer, humanizer],
            tasks=[research_task, write_task, humanize_task],
            verbose=False
        )

        result_container = {}
        error_container = {}

        def run_crew():
            try:
                result_container["result"] = crew.kickoff()
            except Exception as e:
                error_container["error"] = str(e)

        thread = threading.Thread(target=run_crew)
        thread.start()

        prev = {"research": None, "write": None, "humanize": None}

        while thread.is_alive():
            time.sleep(2)
            elapsed = int(time.time() - start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            timer_text.markdown(f"⏱️ **Time elapsed: {mins}m {secs}s** — AI is thinking, please wait...")

            r = "✅ Done" if shared["research"] else "⏳ Running..."
            w = "✅ Done" if shared["write"] else ("⏳ Running..." if shared["research"] else "⏸️ Waiting")
            h = "✅ Done" if shared["humanize"] else ("⏳ Running..." if shared["write"] else "⏸️ Waiting")

            agent_status.markdown(f"""
| Agent | Model | Status |
|-------|-------|--------|
| 🔍 Researcher | Mistral 7B | {r} |
| ✍️ Writer | Llama 3.2 3B | {w} |
| 🧑 Humanizer | Phi-3.5 Mini | {h} |
            """)

            done_count = sum([1 for v in shared.values() if v is not None])
            progress_bar.progress(min(90, 10 + done_count * 27))

            if not shared["research"]:
                status_text.markdown("### 🔍 Step 1/3 — Researcher (Mistral 7B) gathering facts...")
            elif not shared["write"]:
                status_text.markdown("### ✍️ Step 2/3 — Writer (Llama 3.2 3B) writing the post...")
            elif not shared["humanize"]:
                status_text.markdown("### 🧑 Step 3/3 — Humanizer (Phi-3.5 Mini) making it human...")

            if shared["research"] and shared["research"] != prev["research"]:
                prev["research"] = shared["research"]
                researcher_output.markdown(shared["research"])

            if shared["write"] and shared["write"] != prev["write"]:
                prev["write"] = shared["write"]
                writer_output.markdown(shared["write"])

            if shared["humanize"] and shared["humanize"] != prev["humanize"]:
                prev["humanize"] = shared["humanize"]
                humanizer_output.markdown(shared["humanize"])

        thread.join()

        if "error" in error_container:
            st.error(f"Something went wrong: {error_container['error']}")
        else:
            result = result_container["result"]
            total_time = int(time.time() - start_time)
            total_mins = total_time // 60
            total_secs = total_time % 60

            progress_bar.progress(100)
            status_text.markdown("### ✅ All 3 agents finished!")
            timer_text.markdown(f"⏱️ **Total time: {total_mins}m {total_secs}s**")

            agent_status.markdown("""
| Agent | Model | Status |
|-------|-------|--------|
| 🔍 Researcher | Mistral 7B | ✅ Done |
| ✍️ Writer | Llama 3.2 3B | ✅ Done |
| 🧑 Humanizer | Phi-3.5 Mini | ✅ Done |
            """)

            if shared["research"]: researcher_output.markdown(shared["research"])
            if shared["write"]: writer_output.markdown(shared["write"])
            if shared["humanize"]: humanizer_output.markdown(shared["humanize"])

            st.markdown("---")
            st.subheader("📄 Final Humanized Blog Post")
            st.markdown(str(result))

            st.download_button(
                label="⬇️ Download as .txt",
                data=str(result),
                file_name=f"{topic.replace(' ', '_')}_blog.txt",
                mime="text/plain"
            )