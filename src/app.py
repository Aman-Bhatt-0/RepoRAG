import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .rag_utils import clone_and_process_repo, query_repo

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class RepoRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/load_repo")
async def load_repo(req: RepoRequest):
    try:
        repo_name = clone_and_process_repo(req.repo_url)
        return JSONResponse({"success": True, "repo_name": repo_name})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})

@app.post("/ask")
async def ask(req: QueryRequest):
    try:
        answer = query_repo(req.query)
        if answer == "⚠️ No repository loaded. Please load one first.":
            return JSONResponse({"success": False, "error": answer})
        return JSONResponse({"success": True, "answer": answer})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)