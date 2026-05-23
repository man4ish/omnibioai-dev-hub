import os

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "obsolete"}

def load_documents(repo_paths):
    docs = []
    for repo in repo_paths:
        if not os.path.exists(repo):
            print(f"⚠️  Skipping missing repo: {repo}")
            continue
        for root, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if fname.endswith(".md"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read().strip()
                        if text:
                            docs.append({
                                "id": fpath,
                                "text": text,
                                "source": fpath
                            })
                    except Exception as e:
                        print(f"⚠️  Could not read {fpath}: {e}")
    print(f"📄 Loaded {len(docs)} documents")
    return docs