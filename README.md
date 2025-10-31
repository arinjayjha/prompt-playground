# Prompt Engineering Playground

An interactive Streamlit app that demonstrates different prompt engineering techniques (Zero-shot, Few-shot, Role, and Chain-of-Thought) and allows tweaking LLM configurations such as temperature, top_p, and max tokens.  
Built with **Azure OpenAI** (`gpt-4o-mini` deployment).

---

## 🚀 Features

| Feature | Description |
|----------|--------------|
| 🎯 **Prompt Modes** | Zero-shot, Few-shot, Role, and Chain-of-Thought (CoT) |
| ⚙️ **Configurable Parameters** | Temperature, top_p, max_tokens, frequency & presence penalties |
| 🧠 **Azure OpenAI Integration** | Works with Azure deployments using `.env` credentials |
| 🧩 **Presets** | Built-in prompt templates for each mode |
| 🧍‍♂️ **Role Prompts** | Switch model persona easily |
| 🔄 **Side-by-Side Panels** | Compare prompt styles and model behavior visually |
| ⭐ **Rating & Notes** | Evaluate each output with a 1–5 rating and notes |
| 📊 **Token Usage Display** | Shows approximate total token count per run |
| 💾 **Session History** | Saves all runs in session with metadata |
| ⬇️ **Export** | Download history as CSV or JSON |
| 💡 **Help Tab** | Built-in guide explaining every feature and usage examples |

---

## 🧰 Requirements

- Python 3.9+
- Streamlit ≥ 1.22
- openai ≥ 1.0.0
- python-dotenv ≥ 1.0.0

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚙️ Setup Instructions (Windows)

1. **Clone or create project folder**
   ```bash
   mkdir prompt-playground
   cd prompt-playground
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Create a `.env` file** (in the same folder as `app.py`):
   ```env
   AZURE_OPENAI_API_KEY="your_key_here"
   AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
   AZURE_OPENAI_API_VERSION=2025-01-01-preview
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   The app runs at [http://localhost:8501](http://localhost:8501)

---

## 🖥️ Using the App

### 🎛️ Sidebar
- **LLM Config:** adjust model parameters (temperature, top_p, etc.).  
- **Prompt Presets:** select one (e.g., Few-shot, Role) and click *Insert into Panel 1/2*.  
  The app automatically updates the style and prompt text area.

### 🧩 Panels
Each panel contains:
- Prompt style dropdown (for labeling)
- Editable prompt text area
- Input field for `{input}`, `{question}`, `{feature}`, or `{q}`
- Run button
- Output + Metadata
- Rating (1–5) and Notes (stored in history)

### 🕒 Session History
View previous runs with timestamps, styles, token usage, and personal notes.  
You can download results as **CSV** or **JSON** for analysis or reports.

---

## 💡 Example Prompts

| Type | Prompt Template | Example Input |
|------|------------------|---------------|
| **Zero-shot** | `Summarize the following text: {input}` | `Tech world is undergoing a lot of layoffs...` |
| **Few-shot** | `Q: What is the capital of France? ... Q: {question}` | `Who wrote Pride and Prejudice?` |
| **Role** | `You are an expert product manager... {feature}` | `in-app payments for a mobile game` |
| **Chain-of-Thought** | `Show step-by-step reasoning for: {q}` | `If I have 3 apples and buy 2 more, how many total?` |

---

## 🧾 Output Example

**Prompt:**  
```
Summarize the following text: Tech world is undergoing a lot of layoffs that will affect the job market in the worst way possible
```

**Output:**  
```
The tech industry is experiencing significant layoffs, which are expected to negatively impact the job market.
```

**Metadata:**  
```json
{
  "model_or_deployment": "gpt-4o-mini",
  "elapsed_ms": 8110,
  "choices": 1,
  "usage": "CompletionUsage(completion_tokens=19, prompt_tokens=33, total_tokens=52)"
}
```

---

## 🧑‍🔧 Troubleshooting

| Problem | Solution |
|----------|-----------|
| `RuntimeError: credentials not found` | Check `.env` file path and ensure `load_dotenv()` is in `app.py` |
| `Authentication error 403/401` | Verify API key and endpoint (no trailing `/`) |
| `ChatCompletion not found` | Upgrade OpenAI SDK → `pip install --upgrade openai` |
| `Preset not showing` | Refresh Streamlit page (some versions don’t auto-refresh) |
| High token usage | Lower `max_tokens` or temperature |

---

## 🧠 Tips for Effective Prompting

- **Few-shot:** Show the model clear examples of the expected format.  
- **Role prompts:** Use specific personas to guide tone and expertise.  
- **Chain-of-Thought:** Ask the model to reason step-by-step for complex problems.  
- **Experiment:** Compare outputs across different temperature values using the side-by-side panels.

---

## 📦 Export / Submission Info

All session runs are saved in memory. You can download:
- `prompt_playground_history.csv`
- `prompt_playground_history.json`

Include these files for submission or further analysis.

---

## 🏁 Project Summary

This mini-project fulfills the **Prompt Engineering Playground** brief:  
> Build a Prompt Playground that lets users try zero-shot, few-shot, role, and CoT prompts, while tweaking LLM configs (temp, top_p, etc.), with side-by-side prompt effect comparison.

Built and tested on Windows with Azure OpenAI (gpt-4o-mini).  
Developed by **Arinjay (2025)** as part of *Agentic AI Curriculum – Mini Project #1*.
