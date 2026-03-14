from collections import defaultdict
import json
import re
import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Body
from openai import OpenAI

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from datetime import datetime
from fastapi import Request

import requests

from fastapi import UploadFile, File
from openai import APIConnectionError

BASE_DIR = Path(__file__).parent
SVG_DIR = BASE_DIR / "svg"
# in the future, avoid using the csv verison, use the svg_names.pkl; but maybe for now, this csv version is good enough for the FAST & SIMPLE keyword matching 
SVG_INDEX_FILE = BASE_DIR / "svg_names.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
PROMPT_LOG = OUTPUT_DIR / "prompts.csv"

# store uploaded pdf content for the session (simple global memory)
PDF_TEXT = ""
PDF_INJECTED = False
PDF_STATE = {}

# load svg names ONCE from csv
with SVG_INDEX_FILE.open() as f:
    # SVG_NAMES = [row[0].strip().lower() for row in csv.reader(f)]
    SVG_NAMES = [row[0].strip().lower() for row in csv.reader(f) if row and row[0].strip()] # skip empty rows



# load embeddings
with open("svg_embeddings.pkl", "rb") as f:
    DATA = pickle.load(f)

SVG_FILES = DATA["filenames"]
SVG_EMBEDS = np.array(DATA["embeddings"])
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# per-chat memory of shown svgs
CHAT_USED_SVGS = defaultdict(set)

def log_prompt(prompt: str, ip: str, response):
    # country = requests.get(f"https://ipapi.co/{ip}/country_name/").text
    write_header = not PROMPT_LOG.exists()
    with open(PROMPT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "country", "ip", "prompt", "response"])
        writer.writerow([datetime.utcnow().isoformat(), "", ip, prompt, response])


def find_svg_semantic(keyword: str, exclude=None, top_k: int = 8):
    # returns None if exclusion exhausts the top_k pool
    exclude = exclude or set()

    q = embed_model.encode([keyword], normalize_embeddings=True)[0]
    sims = SVG_EMBEDS @ q  # cosine since normalized

    # try best unused from top_k
    candidates = 0
    for idx in np.argsort(-sims)[:top_k]:
        svg = SVG_FILES[int(idx)]
        if not (SVG_DIR / svg).exists():
            continue
        candidates += 1
        if svg in exclude:
            continue
        return svg

    # if exclusion was active and we had candidates but all were excluded, signal exhaustion
    if exclude and candidates > 0:
        return None

    # no usable candidates at all, or exclusion not active: fallback to best
    best_idx = int(sims.argmax())
    best_svg = SVG_FILES[best_idx]
    return best_svg if (SVG_DIR / best_svg).exists() else "default.svg"



app = FastAPI(title="AI Whiteboard Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/svg", StaticFiles(directory=SVG_DIR), name="svg")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static") # contains hand pencil png

LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# "google/gemma-2-9b-it" no need for --model-len 8192
# "tiiuae/Falcon-H1-7B-Base"
# "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
# "TinyLlama/TinyLlama-1.1B-Chat-v1.0"    --max-model-len 2048
# "Qwen/Qwen2.5-7B-Instruct"    # "Qwen/Qwen2.5-14B-Instruct"
# "meta-llama/Llama-3.1-8B-Instruct" --pretty good
# "mistralai/Mistral-7B-Instruct-v0.2"  # "mistralai/Mistral-7B-Instruct-v0.3"
# "microsoft/Phi-3-mini-4k-instruct"  
VLLM_BASE_URL = "http://localhost:8000/v1"

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="not-needed",
    timeout=10.0
)
def classify_intent_llm(user_prompt: str) -> str:
    # returns "question" or "request"
    system = (
        "classify the user's message.\n"
        "output exactly one word: question or request.\n"
        "question = asking for information/explanation.\n"
        "request = asking you to produce/show/generate/do something.\n"
        "if unclear, output question.\n"
        "output only that one word."
    )
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt or ""},
            ],
            max_tokens=1,
            temperature=0.0,
        )
        out = (r.choices[0].message.content or "").strip().lower()

        # normalize model output
        m = re.search(r"(question|request)", out)
        return m.group(1) if m else "question"
    except Exception:
        return "question"

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# --- FAST & SIMPLE keyword to svg matching ---
def find_svg(keyword: str) -> str:
    k = keyword.lower()

    for name in SVG_NAMES:
        if k in name:
            return name + ".svg"

    return "default.svg"


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # size limit: reject > 5MB
    if file.size and file.size > 5 * 1024 * 1024:
        return JSONResponse({"success": False, "error": "too_big"}, status_code=413)

    global PDF_TEXT

    # verify file type
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"success": False, "error": "not a pdf"}, status_code=400)

    # read pdf
    from pypdf import PdfReader
    reader = PdfReader(file.file)
    extracted = []

    # extract text from each page
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted.append(text)

    PDF_TEXT = "\n\n".join(extracted).strip()

    # reset pdf injection state so the next question injects the new pdf once
    global PDF_INJECTED
    PDF_INJECTED = False
    
    return {"success": True, "text": PDF_TEXT}


@app.post("/generate_storyboard")
async def generate_storyboard(request: Request, body: dict = Body(...)):
    history = body.get("history", [])
    print("hit generate_storyboard")
    prompt = body.get("prompt")
    
    if not prompt:
        return JSONResponse(status_code=400, content={"detail": "missing prompt"})

    original_user_prompt = prompt
    intent = classify_intent_llm(original_user_prompt)
    use_exclusion = (intent == "question")
    print('intent: ', intent)
    
    prompt = 'Answer like I am Ten, but answer directly. If there is no direct question/request, respond with a friendly chat: acknowledge and thank, give no other answers or facts. Else, you give an answer to the best of your abilities. You do not have to say things like: "but I do not know much about it" if you are already answering. ' + prompt
    
    scene_count = int(body.get("scenes", 1))
    scene_count = max(1, min(scene_count, 8))
    
    system_prompt = f"""
    You are a JSON generator. You MUST follow the output format exactly.
    
    TASK:
    Generate EXACTLY {scene_count} descriptions or answers and EXACTLY {scene_count} Two-word phrases in response to the given prompt.
    The responses must describe the same topic with a smooth progression.
    
    OUTPUT FORMAT RULES (STRICT):
    1. Output MUST be a SINGLE JSON ARRAY.
    2. The array MUST contain EXACTLY {scene_count} OBJECTS.
    3. Each object MUST have EXACTLY these two fields:
       - "text"
       - "Two word phrase"
    4. DO NOT output anything outside the JSON array.
    5. DO NOT include markdown, comments, or explanations.
    6. DO NOT output multiple arrays.
    7. DO NOT output strings, lists, or numbers by themselves.
    
    FIELD RULES:
    - Derive the "Two word phrase"s directly from the user prompt.
    - "Two word phrase" MUST contain at least one word that appears verbatim in the USER PROMPT.
    - "text": Text must be Two to THREE full grammatical sentences, good descriptions or answers in response to. It MUST NOT be a short phrase.
    - "Two word phrase": MUST be EXACTLY two words, no more, no less.
    - If the prompt contains a famous person, subject, or object, the FIRST "Two word phrase" MUST be exatly that.
    - Otherwise, each "Two word phrase" MUST be a concrete identifier of the main subject, not a category, region, attribute, role, or summary.
    - All "Two word phrase" values MUST be UNIQUE.
    
    
    FAILURE RULE:
    If you cannot follow ALL rules exactly, output this EXACT JSON and NOTHING ELSE:
    [
      {{
        "text": "FORMAT_ERROR",
        "Two word phrase": "FORMAT_ERROR"
      }}
    ]
    
    USER PROMPT:
    """

    # build history context
    history_text = ""
    for turn in history:
        for sc in turn.get("scenes", []):
            history_text += sc.get("text", "") + "\n"
    
    # prepend history to the user prompt
    # prompt = history_text + "\n" + prompt

    # inject pdf text ONLY ONCE into the first conversation turn
    chat_id = request.headers.get("X-Chat-ID", "default")

    pdfText = body.get("pdfText", "")
    pdfUploaded = body.get("pdfUploaded", False)
    pdfInjected = body.get("pdfInjected", False)
    
    # first turn AFTER a PDF upload → inject PDF text
    if pdfUploaded and not pdfInjected:
    	prompt = (
    		f"Here is the uploaded PDF content you must reference:\n\n"
    		f"{pdfText}\n\n"
    		f"User question: {prompt}"
    	)
    
    	# also append the PDF as a memory turn in history (same behavior as before)
    	history.append({
    		"user": "__pdf__",
    		"scenes": [{"text": pdfText, "Two word phrase": "pdf memory"}]
    	})
    
    	# NOTE: do NOT set any PDF state here — the frontend handles pdfInjected per chat

    priority_rules = """
    PRIORITY RULES (STRICT):
    - if the CURRENT QUESTION is not a direct question/request, respond with a friendly chat: acknowledge and thank, give no other answers or facts.
    - you must answer the CURRENT QUESTION directly.
    - use HISTORY only as background for continuity.
    - do not continue an older thread unless the CURRENT QUESTION asks for it.
    - if CURRENT QUESTION conflicts with HISTORY, follow CURRENT QUESTION.
    """
        
    # for google gemma
    if LLM_MODEL_NAME == "google/gemma-2-9b-it":
        context_block = "HISTORY (reference only, do not answer this section):\n" + history_text
        question_block = "CURRENT QUESTION (answer this now):\n" + prompt
        
        prompt_text = system_prompt + "\n" + priority_rules + "\n\n" + context_block + "\n\n" + question_block

        # prompt_text = system_prompt + prompt
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=300,
                temperature=0.01,
        
            )
        except APIConnectionError:
            return JSONResponse(
                status_code=503,
                content=[{
                    "text": "Apologies! The AI model is offline as GPU is busy with other tasks. Please try again later.",
                    "Two word phrase": "Model offline",
                    "svg": "peace.svg"
                }]
            )
    else:
        # for others
        '''response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.01,
        )'''
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                # messages=[
                #     {"role": "system", "content": system_prompt},
                #     {"role": "user", "content": prompt}
                # ],
                messages = [
                    {"role": "system", "content": system_prompt + "\n" + priority_rules},
                    {"role": "user", "content": "HISTORY (reference only, do not answer this section):\n" + history_text},
                    {"role": "user", "content": "CURRENT QUESTION (answer this now):\n" + prompt},
                ],
                max_tokens=300,
                temperature=0.01,
            )
        except APIConnectionError:
            return JSONResponse(
                status_code=503,
                content=[{
                    "text": "Apologies! The AI model is offline as GPU is busy with other tasks. Please try again later.",
                    "Two word phrase": "Model offline",
                    "svg": "peace.svg"
                }]
            )

    
    raw = response.choices[0].message.content.strip()
    
    # for Qwen
    # cut to outermost JSON

    start = raw.find("[")
    end = raw.rfind("]")
    
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]
    
    # hard sanitize double braces
    while "{{" in raw or "}}" in raw:
        raw = raw.replace("{{", "{").replace("}}", "}")
    # FIX: multiple arrays bug
    if raw.count("[") > 1:
        raw = raw[:raw.find("]") + 1]
    
    try:
        data = json.loads(raw)
    except Exception as e:
        print('orig output: ', raw, '/n/n')
        #print("JSON ERROR:", e)
        #print("LLM RAW OUTPUT:", repr(raw))
        data = [{"text": "Sorry, I am not able to answer your question.", "Two word phrase": "Error"}]


    
    # collapse accidental double braces the model might copy
    while "{{" in raw or "}}" in raw:
        raw = raw.replace("{{", "{").replace("}}", "}")
        
    if isinstance(data, dict):
        data = [data]
    
    # enforce exactly 3 scenes
    # enforce exact dynamic scene count
    if len(data) < scene_count:
        while len(data) < scene_count:
            data.append(data[-1])
    elif len(data) > scene_count:
        data = data[:scene_count]


    # force valid svg filename
    # svg_file = find_svg(data[0]["keyword"])
    # svg_file = find_svg_semantic(data[0]["keyword"])
    
    # for i in range(len(data)):
    #     data[i]["svg"] = find_svg_semantic(data[i]["Two word phrase"])
    
    used = CHAT_USED_SVGS[chat_id]
    for i in range(len(data)):
        if use_exclusion:
            svg = find_svg_semantic(data[i]["Two word phrase"], exclude=used, top_k=8)
    
            # restart when we ran out of unused options in top_k
            if svg is None:
                used.clear()
                svg = find_svg_semantic(data[i]["Two word phrase"], exclude=None, top_k=8)
        else:
            svg = find_svg_semantic(data[i]["Two word phrase"], exclude=None, top_k=8)
    
        data[i]["svg"] = svg
        used.add(svg)

    
    #data[0]["svg"] = svg_file
    print("RETURNING:", data[0]["svg"])
    print("LLM KEYWORD:", data[0]["Two word phrase"])
    
    prompt = body.get("prompt")
    ip = request.client.host
    # print(prompt, ip, data)

    
    log_prompt(prompt, ip, data)

    # return JSONResponse(data)
    return JSONResponse(
        data,
        headers={"X-Chat-History": json.dumps(history)}
    )