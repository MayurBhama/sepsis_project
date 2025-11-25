import os

PROJECT_STRUCTURE = {
    "data": [],
    "models": [],
    "notebooks": [],
    "logs": [],
    "config": ["config.yaml"],
    "src": [
        "data_ingestion",
        "data_validation",
        "preprocessing",
        "feature_engineering",
        "model_training",
        "model_evaluation",
        "prediction",
        "pipelines"
    ]
}

def create_project():
    root = "sepsis_project"

    os.makedirs(root, exist_ok=True)

    for folder, subfolders in PROJECT_STRUCTURE.items():
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)

        # Subfolders
        if folder == "src":
            for sub in subfolders:
                os.makedirs(os.path.join(folder_path, sub), exist_ok=True)

        # Files
        for file in subfolders:
            if file.endswith(".yaml"):
                open(os.path.join(folder_path, file), "w").close()

    # Top level files
    open(os.path.join(root, "main.py"), "w").close()
    open(os.path.join(root, "requirements.txt"), "w").close()
    open(os.path.join(root, "README.md"), "w").close()

    print("🎉 Project structure created successfully!")

if __name__ == "__main__":
    create_project()
