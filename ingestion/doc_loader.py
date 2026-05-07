import os

def load_documents(repo_paths):
    docs = []

    for repo in repo_paths:
        readme_path = os.path.join(repo, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                docs.append({
                    "id": repo,
                    "text": f.read(),
                    "source": readme_path
                })

    return docs