# app.py — Prompt Engineering Playground with Help tab
import streamlit as st
from typing import Dict, Any, Tuple, List
import json, os, time, io, csv, re
from dotenv import load_dotenv
load_dotenv()

# OpenAI Azure client import for current SDK
from openai import AzureOpenAI

st.set_page_config(layout="wide", page_title="Prompt Engineering Playground")

# --------------------
# Sidebar: LLM config
# --------------------
st.sidebar.title("LLM Config")
sidebar_api_key = st.sidebar.text_input("API Key (paste)", type="password",
                                        help="Optional: paste Azure OpenAI API key for this session. Prefer .env or Streamlit secrets.")
deployment_name = st.sidebar.text_input("Azure deployment name (or model)", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"))
model_label = st.sidebar.selectbox("Model label (UI only)", ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.7, 0.05)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 1.0, 0.01)
max_tokens = st.sidebar.number_input("Max tokens", min_value=16, max_value=4000, value=512, step=16)
freq_penalty = st.sidebar.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
pres_penalty = st.sidebar.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
n_completions = st.sidebar.number_input("Number of completions (n)", min_value=1, max_value=3, value=1)
show_cot = st.sidebar.checkbox("Show chain-of-thought (if produced)", value=False)

st.sidebar.markdown("---")
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.sidebar.button("Clear History"):
    st.session_state['history'] = []
    st.experimental_rerun() if hasattr(st, "experimental_rerun") else None

# --------------------
# Presets & preset insertion
# --------------------
PRESETS: Dict[str, str] = {
    "Zero-shot: Summarize": "Summarize the following text in one paragraph emphasizing the main findings and implications: {input}",
    "Few-shot: QA (3-shot)": """Q: What is the capital of France?
A: Paris.

Q: Which planet is known as the Red Planet?
A: Mars.

Q: {question}
A:""",
    "Role: Product Manager": "You are an expert product manager with 10 years experience. Produce a 3-bullet roadmap for {feature}, include user impact and one KPI.",
    "CoT: Math/Reasoning": "You are allowed to show chain-of-thought. Show step-by-step reasoning and then give the final concise answer.\n\nQuestion: {q}\nThoughts:\n1."
}

st.sidebar.markdown("## Prompt presets")
preset_choice = st.sidebar.selectbox("Insert preset into panel", list(PRESETS.keys()))

_preset_to_style = {
    "Zero-shot: Summarize": "zero-shot",
    "Few-shot: QA (3-shot)": "few-shot",
    "Role: Product Manager": "role",
    "CoT: Math/Reasoning": "cot"
}

if st.sidebar.button("Insert into Panel 1"):
    st.session_state['preset_to_panel1'] = PRESETS[preset_choice]
    st.session_state['prompt_0'] = PRESETS[preset_choice]
    st.session_state['style_0'] = _preset_to_style.get(preset_choice, st.session_state.get('style_0', 'zero-shot'))

if st.sidebar.button("Insert into Panel 2"):
    st.session_state['preset_to_panel2'] = PRESETS[preset_choice]
    st.session_state['prompt_1'] = PRESETS[preset_choice]
    st.session_state['style_1'] = _preset_to_style.get(preset_choice, st.session_state.get('style_1', 'zero-shot'))

# default session keys if missing
if 'preset_to_panel1' not in st.session_state:
    st.session_state['preset_to_panel1'] = PRESETS["Zero-shot: Summarize"]
if 'preset_to_panel2' not in st.session_state:
    st.session_state['preset_to_panel2'] = PRESETS["Role: Product Manager"]

# --------------------
# Utility functions
# --------------------
def extract_total_tokens(meta: Dict[str, Any]) -> int:
    usage = meta.get("usage")
    if usage is None:
        return 0
    try:
        total = getattr(usage, "total_tokens", None)
        if total is not None:
            return int(total)
    except Exception:
        pass
    if isinstance(usage, dict):
        if "total_tokens" in usage:
            return int(usage["total_tokens"])
    try:
        s = str(usage)
        m = re.search(r"total_tokens\s*=\s*([0-9]+)", s)
        if m:
            return int(m.group(1))
        m2 = re.search(r"total_tokens[:=]\s*([0-9]+)", s)
        if m2:
            return int(m2.group(1))
    except Exception:
        pass
    return 0

def call_llm(api_key_sidebar: str,
             deployment: str,
             prompt: str,
             temperature: float = 0.7,
             top_p: float = 1.0,
             max_tokens: int = 512,
             freq_penalty: float = 0.0,
             pres_penalty: float = 0.0,
             n: int = 1) -> Tuple[str, Dict[str, Any]]:
    api_key = api_key_sidebar or os.getenv("AZURE_OPENAI_API_KEY") or st.secrets.get("AZURE_OPENAI_API_KEY", None)
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT", None)
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or st.secrets.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    if not api_key or not endpoint:
        raise RuntimeError("Azure OpenAI credentials not found. Ensure .env or Streamlit secrets are set.")
    client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
    messages = [{"role": "user", "content": prompt}]
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            n=n
        )
    except Exception as e:
        raise RuntimeError(f"Azure OpenAI request failed: {e}")
    elapsed_ms = int((time.time() - start_time) * 1000)
    outputs = []
    try:
        for ch in response.choices:
            outputs.append(ch.message.content)
    except Exception:
        outputs = [str(response)]
    text_output = "\n\n---\n\n".join(outputs)
    meta = {
        "model_or_deployment": deployment,
        "elapsed_ms": elapsed_ms,
        "choices": len(outputs),
        "usage": getattr(response, "usage", None),
    }
    return text_output, meta

# --------------------
# Top tabs: Playground and Help
# --------------------
tab1, tab2 = st.tabs(["Playground", "Help"])

# --------------------
# Playground tab: main app UI (existing functionality)
# --------------------
with tab1:
    st.title("Prompt Engineering Playground — Play")
    cols = st.columns(2)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Prompt Panel {i+1}")
            # style selectbox uses session_state to allow preset-driven initialization
            style_default = st.session_state.get(f"style_{i}", "zero-shot")
            style = st.selectbox("Prompt style", ["zero-shot","few-shot","role","cot"], key=f"style_{i}", index=["zero-shot","few-shot","role","cot"].index(style_default))
            # prompt area reads from session_state prompt_<i> if set
            default_prompt = st.session_state.get('preset_to_panel1') if i == 0 else st.session_state.get('preset_to_panel2')
            prompt_value = st.session_state.get(f'prompt_{i}', default_prompt)
            prompt = st.text_area("Prompt", value=prompt_value, height=200, key=f"prompt_{i}")
            input_field = st.text_input("User input / variable", key=f"input_{i}")
            run_btn = st.button(f"Run panel {i+1}", key=f"run_{i}")
            if run_btn:
                composed = prompt.format(input=input_field, question=input_field, feature=input_field, q=input_field)
                try:
                    response_text, meta = call_llm(
                        api_key_sidebar=sidebar_api_key,
                        deployment=deployment_name,
                        prompt=composed,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        freq_penalty=freq_penalty,
                        pres_penalty=pres_penalty,
                        n=n_completions,
                    )
                except Exception as e:
                    st.error(f"LLM call error: {e}")
                    response_text, meta = f"Error: {e}", {}
                st.subheader("Output")
                st.write(response_text)
                st.markdown("**Metadata**")
                st.json(meta)
                tokens = extract_total_tokens(meta)
                st.caption(f"Estimated total tokens (best-effort): {tokens}")
                rating = st.radio("Rate this output (1 worst - 5 best)", [1,2,3,4,5], index=4, key=f"rating_{i}")
                notes = st.text_area("Notes (optional)", key=f"notes_{i}", height=80)
                run_entry = {
                    "timestamp": int(time.time()),
                    "panel": i+1,
                    "style": style,
                    "prompt": prompt,
                    "input": input_field,
                    "response": response_text,
                    "meta": meta,
                    "tokens": tokens,
                    "rating": rating,
                    "notes": notes
                }
                st.session_state['history'].append(run_entry)
                st.success("Run saved to session history.")
    # History & export below the panels
    st.markdown("---")
    st.header("Session History")
    hist = st.session_state.get('history', [])
    if hist:
        max_display = min(20, len(hist))
        for idx, h in enumerate(reversed(hist[-max_display:]), 1):
            st.markdown(f"**Run {len(hist) - (max_display - idx)}** — Panel {h['panel']} | Style: {h['style']} | Rating: {h['rating']}")
            st.write("Prompt:")
            st.write(h['prompt'])
            st.write("Input:", h['input'])
            st.write("Response:")
            st.write(h['response'])
            st.write("Notes:", h.get('notes', ''))
            st.write("---")
        def history_to_csv_bytes(history_list: List[Dict[str,Any]]) -> bytes:
            output = io.StringIO()
            writer = csv.writer(output)
            header = ["timestamp","panel","style","prompt","input","response","tokens","rating","notes"]
            writer.writerow(header)
            for item in history_list:
                writer.writerow([
                    item.get("timestamp"),
                    item.get("panel"),
                    item.get("style"),
                    item.get("prompt"),
                    item.get("input"),
                    item.get("response"),
                    item.get("tokens"),
                    item.get("rating"),
                    item.get("notes"),
                ])
            return output.getvalue().encode("utf-8")
        csv_bytes = history_to_csv_bytes(hist)
        st.download_button("Download history as CSV", data=csv_bytes, file_name="prompt_playground_history.csv", mime="text/csv")
        st.download_button("Download history as JSON", data=json.dumps(hist, default=str, indent=2), file_name="prompt_playground_history.json", mime="application/json")
    else:
        st.info("No history yet. Run a prompt to populate the history.")

# --------------------
# Help tab: full instructions & examples
# --------------------
with tab2:
    st.title("Help — How to use the Prompt Engineering Playground")
    st.markdown("Welcome! This page explains every feature in the app and gives example inputs and troubleshooting tips.")
    st.header("Quick start (3 steps)")
    st.markdown("""
    1. Ensure Azure credentials are configured in a `.env` file or Streamlit secrets (see 'Credentials' below).
    2. Use the **Prompt presets** in the sidebar to load a starter prompt into Panel 1 or Panel 2.
    3. Fill the **User input / variable** for `{input}`, `{question}`, `{feature}`, or `{q}` and click **Run panel 1** (or **Run panel 2**).
    """)

    st.header("UI overview")
    st.markdown("""
    - **LLM Config (sidebar):** adjust temperature, top_p, max_tokens, frequency/presence penalties, and `n` (number of completions).
    - **Prompt presets (sidebar):** choose a built-in template and click _Insert into Panel X_. This overwrites the prompt text in that panel and sets the prompt style dropdown.
    - **Prompt Panel (main):**
      - *Prompt style:* UI label (zero-shot, few-shot, role, cot). It updates when you insert a preset.
      - *Prompt:* editable prompt template. Use `{input}`, `{question}`, `{feature}`, or `{q}` as variable placeholders.
      - *User input / variable:* value that will replace the placeholder when running.
      - *Run panel X:* execute the prompt with current LLM config.
      - *Output & Metadata:* result text, elapsed time, and token usage (best effort).
      - *Rating & Notes:* rate quality and record notes; runs are saved to session history.
    - **Session History:** saved runs, with ability to download CSV/JSON.
    """)

    st.header("Examples (copy-paste into Panel X)")
    st.subheader("Few-shot (QA 3-shot)")
    st.code("""Q: What is the capital of France?
A: Paris.

Q: Which planet is known as the Red Planet?
A: Mars.

Q: {question}
A:""")
    st.markdown("**User input / variable:** `Who wrote Pride and Prejudice?`")
    st.subheader("Zero-shot summarization")
    st.code("Summarize the following text in one paragraph emphasizing the main findings and implications: {input}")
    st.markdown("**User input / variable:** `Tech world is undergoing a lot of layoffs that will affect the job market in the worst way possible`")
    st.subheader("Role prompt")
    st.code("You are an expert product manager with 10 years experience. Produce a 3-bullet roadmap for {feature}, include user impact and one KPI.")
    st.markdown("**User input / variable:** `in-app payments for a mobile game`")
    st.subheader("Chain-of-Thought (CoT)")
    st.code("You are allowed to show chain-of-thought. Show step-by-step reasoning and then give the final concise answer.\\n\\nQuestion: {q}\\nThoughts:\\n1.")
    st.markdown("**User input / variable:** `If I have 3 apples and buy 2 more, how many total?`")

    st.header("Credentials (Windows/.env)")
    st.markdown("""
    Create a `.env` file in the project folder (same folder as `app.py`) with these lines:
    ```
    AZURE_OPENAI_API_KEY=\"your_key_here\"
    AZURE_OPENAI_ENDPOINT=\"https://your-resource-name.openai.azure.com\"
    AZURE_OPENAI_API_VERSION=2025-01-01-preview
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
    ```
    Then restart the app. You can also paste the API key into the sidebar field for a single session.
    """)

    st.header("Troubleshooting")
    st.markdown("""
    - **LLM call error: credentials not found** — ensure `.env` is in the same folder and `load_dotenv()` is present at the top of `app.py`.
    - **Authentication errors (401/403)** — double-check the API key and endpoint URL in the `.env` (no trailing slashes).
    - **SDK errors about `ChatCompletion` or `OpenAI.init`** — upgrade `openai` to latest: `pip install --upgrade openai` inside your venv.
    - **Preset not showing immediately** — if the prompt does not update after clicking Insert, refresh the browser or restart Streamlit. (Most installs will update immediately.)
    - **High token usage** — reduce `max_tokens` in the sidebar or lower temperature/top_p to reduce length and cost.
    """)

    st.header("Tips & best practices")
    st.markdown("""
    - Use **few-shot** for tasks needing format guidance (e.g., Q/A, classification).
    - Use **role** prompts when you want the model to adopt a persona or expertise.
    - Use **CoT** for step-by-step reasoning tasks (math, planning). Consider hiding chain-of-thought in final outputs for production.
    - Keep prompts short and explicit; include examples when exact format is required.
    - Rate outputs and save notes in session history — this helps iterate and evaluate prompt changes.
    """)

    st.header("If you need help")
    st.markdown("Paste any terminal/Streamlit error messages here and I will help debug. If you'd like, I can also generate a README.md file describing the project to include in a GitHub repo.")
