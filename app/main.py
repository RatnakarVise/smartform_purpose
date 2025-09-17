from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI(title="SmartForm Window Explanation API")

# ---- Strict input model for window explanation ----
class WindowSnippet(BaseModel):
    page_name: Optional[str] = None
    window_name: Optional[str] = None
    fields: List[str] = Field(default_factory=list)
    tables: List[str] = Field(default_factory=list)
    code: List[str] = Field(default_factory=list)

# ---- LLM chain builder ----
def build_chain(snippet: WindowSnippet):
    snippet_json = json.dumps(snippet.dict(), ensure_ascii=False, indent=2)

    SYSTEM_MSG = "You are a precise SAP SmartForm reviewer and explainer. Respond in strict JSON only."

    USER_TEMPLATE = """
You are an SAP ABAP SmartForm Expert with 20 years of experience.
Explain the following SmartForm window in a concise but professional way.
Include all fields, tables, and ABAP code this window contains.

Return ONLY strict JSON:
{{
  "name": "{window_name}",
  "explanation": "<clear explanation of this window and its role>"
}}

Window JSON:
{context_json}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    parser = JsonOutputParser()
    return prompt | llm | parser

# ---- LLM invocation ----
def llm_explain_window(snippet: WindowSnippet):
    chain = build_chain(snippet)
    return chain.invoke({
        "context_json": json.dumps(snippet.dict(), ensure_ascii=False, indent=2),
        "window_name": snippet.window_name  # <-- pass window_name to fix missing variable
    })
# ---- Traverse Pages → Windows ----
def traverse_smartform(data: Dict[str, Any]) -> List[Dict[str, str]]:
    results = []

    for page in data.get("PAGES", []):
        page_name = page.get("PAGE_NAME", "Unnamed Page")
        for win in page.get("WINDOWS", []):
            snippet = WindowSnippet(
                page_name=page_name,
                window_name=win.get("WINDOW_NAME", "Unnamed Window"),
                fields=win.get("FIELDS") or [],
                tables=win.get("TABLES") or [],
                code=win.get("CODE") or []
            )
            try:
                llm_result = llm_explain_window(snippet)
                results.append(llm_result)
            except Exception as e:
                results.append({"name": snippet.window_name, "explanation": f"LLM call failed: {e}"})
    return results

# ---- API endpoint ----
@app.post("/explain-smartform")
async def explain_smartform(data: List[Dict[str, Any]]):
    try:
        all_results = []
        for d in data:
            results = traverse_smartform(d)
            all_results.extend(results)
        return all_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
