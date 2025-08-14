# Created on 2025-03-10
# Since Feb 2025, NVIDIA isaaclab project has removed all its dependencies on omni.
# This project is originally based on Isaac Sim 4.2, but no longer supported by Isaac Lab v2.
# What's even worse is that Isaac Lab v2 is now using `isaaclab` as package name, conflicting with our project packages.
# We adapted the code to match with Isaac Sim 4.5 requirements, however, the new commits are not compatible with old isaac-sim and Isaac Lab v1 installed environments.
# Luckily we noticed that all changes are only replacing the old import names with new ones, no single line of code has been changed.
# Therefore, we are able to maintain both versions of the codebase by using a simple script to replace the import names.
# This script is executed pre-to-commit, allowing us to synchronize the code we are developing in 4.5 to also be compatible with 4.2.

import os
import shutil
import subprocess
import re
import tempfile
from pathlib import Path

isaaclab_replace_dict = {
    "isaaclab": "omni.isaac.lab",
    "isaaclab_tasks": "omni.isaac.lab_tasks",
    "isaaclab_rl.rsl_rl": "omni.isaac.lab_tasks.utils.wrappers.rsl_rl",
    "isaaclab_assets": "omni.isaac.lab_assets",
}

from GBC.utils.base.assets import PROJECT_ROOT_DIR

def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip()


def replace_imports_in_file(file_path, replace_dict):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for key, value in replace_dict.items():
        content = re.sub(
            rf'(from|import)\s+{re.escape(key)}(\.|\s)', rf'\1 {value}\2', content
        )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def convert_branch(source_branch: str, target_branch: str, secondary_dir: str):
    original_branch = subprocess.getoutput("git branch --show-current")
    stash_hash = None

    # Check for unstaged changes
    ret, unstaged = run_cmd("git status --porcelain")
    if unstaged:
        ret, stash_hash = run_cmd("git stash create")
        run_cmd("git stash")

    # Checkout source branch and get commit hash
    if not run_cmd(f"git checkout {source_branch}")[0]:
        return {"success": False, "message": f"Failed to checkout source branch {source_branch}."}

    ret, source_branch_hash = run_cmd("git rev-parse HEAD")


    secondary_dir = Path(secondary_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        TEMPLATE_DIR = Path(temp_dir)
        shutil.copytree(PROJECT_ROOT_DIR / secondary_dir, TEMPLATE_DIR / secondary_dir)

        # Replace imports
        for py_file in (TEMPLATE_DIR / secondary_dir).rglob('*.py'):
            replace_imports_in_file(py_file, isaaclab_replace_dict)

        # Checkout target branch (create if necessary)
        ret, branches = run_cmd(f"git branch --list {target_branch}")
        if branches == "":
            if not run_cmd(f"git checkout -b {target_branch}")[0]:
                return {"success": False, "message": f"Failed to create branch {target_branch}."}
        else:
            run_cmd(f"git checkout {target_branch}")

        # Copy files back
        for item in (TEMPLATE_DIR / secondary_dir).rglob('*'):
            if item.is_file():
                dest_path = PROJECT_ROOT_DIR / secondary_dir / item.relative_to(TEMPLATE_DIR / secondary_dir)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)


    
    # Git add and commit
    run_cmd("git add GBC")
    ret, commit_output = run_cmd(
        f'git commit -m "Synchronized from branch {source_branch} commit {source_branch_hash} to adapt program to isaacsim 4.2"'
    )

    # Restore original branch and stash if needed
    run_cmd(f"git checkout {original_branch}")
    if stash_hash:
        run_cmd("git stash pop")

    return {
        "success": True,
        "message": "Branch converted successfully.",
        "source_branch": source_branch,
        "target_branch": target_branch,
        "source_commit_hash": source_branch_hash,
        "stash_restored": stash_hash is not None,
        "cmd_rtn": ret,
        "cmd_output": commit_output,
    }


if __name__ == "__main__":
    source_branch = "isaacsim_4_5"
    target_branch = "master"
    secondary_dir = "GBC"

    result = convert_branch(source_branch, target_branch, secondary_dir)
    print(result)