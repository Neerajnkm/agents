"""
update_root_requirements.py

Scans all subfolders for requirements.txt files and updates the root requirements.txt to include them using the -r directive.
Run this script from the root of the repository.
"""
import os

ROOT_REQ = "requirements.txt"
EXCLUDE_DIRS = {"env", "__pycache__"}


def find_requirements_files(root_dir):
    req_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude certain directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for filename in filenames:
            if filename == "requirements.txt" and os.path.relpath(os.path.join(dirpath, filename), root_dir) != ROOT_REQ:
                req_files.append(os.path.relpath(os.path.join(dirpath, filename), root_dir))
    return sorted(req_files)


def update_root_requirements(root_dir):
    req_files = find_requirements_files(root_dir)
    header = (
        "# Root requirements.txt for the Agents codebase\n"
        "# This file includes requirements from all agent and child folders.\n"
        "# To add new requirements, create a requirements.txt in the relevant folder and run update_root_requirements.py\n\n"
    )
    with open(os.path.join(root_dir, ROOT_REQ), "w") as f:
        f.write(header)
        for req in req_files:
            f.write(f"-r {req}\n")
    print(f"Updated {ROOT_REQ} with {len(req_files)} included requirements files.")


if __name__ == "__main__":
    update_root_requirements(os.path.dirname(os.path.abspath(__file__)))
