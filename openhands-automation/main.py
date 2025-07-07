from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List

PROMPT = """\
Can you please identify potential issues in this pipeline and fix them? 

##CODE##

Please write the corrected code to ##OUTPUT_FILE## and summary of your fixes in the start of the file as comments, Mark every line you modify with the tag #FIXED at top.

You must not run any commands externaly, you just write the file to fixed.py, you have nothing to do with data or util.py anything.
Don't prompt me to run any commands, just write the file to fixed.py. I trust current directory.
"""

VENV_ACTIVATE = "/Users/usamabintariq/Documents/GitHub/emdl_project/edml_env/bin/activate"
OPENHANDS_TIMEOUT = 300 
WAIT_FOR_OUTPUT = 60
CHECK_INTERVAL = 5

def gather_examples(root: Path) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for script in root.glob("pipelines/*/example-0.py"):
        try:
            fixed_path = script.parent / "fixed.py"
            if fixed_path.exists():
                logging.info("Skipping %s, already has fixed.py", script.relative_to(root))
                continue
            content = script.read_text(encoding="utf-8")
            stripped = "\n".join(
                line for line in content.splitlines() if not line.lstrip().startswith("#")
            )
            examples.append(
                {
                    "name": script.parent.name,
                    "path": script.parent,
                    "content": stripped,
                }
            )
            logging.info("Loaded %s", script.relative_to(root))
        except Exception:
            logging.exception("Unable to read %s", script)
    return examples


def _stream_until(
    proc: subprocess.Popen[str],
    stop_check: Callable[[], bool],
    timeout: float,
) -> int:
    start = time.monotonic()
    while True:
        if proc.stdout and proc.stdout.readable() and not proc.stdout.closed:
            line = proc.stdout.readline()
            if line:
                print(line, end="")

        if stop_check():
            logging.info("Fixed file detected — terminating OpenHands early")
            proc.terminate()
            break
        if proc.poll() is not None:
            break
        if time.monotonic() - start > timeout:
            logging.error("Timed out after %s s — killing OpenHands", timeout)
            proc.kill()
            break
        time.sleep(0.1)

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    return proc.returncode or 0


def run_openhands(prompt_file: Path, work_dir: Path, stop_check: Callable[[], bool]) -> int:
    bash_cmd = f"source {VENV_ACTIVATE} && openhands --directory {work_dir} --file {prompt_file}"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    with subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    ) as proc:
        return _stream_until(proc, stop_check, OPENHANDS_TIMEOUT)


def process_example(example: Dict[str, str], output_file: str = "fixed.py") -> bool:
    with tempfile.TemporaryDirectory(prefix="openhands_ctx_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        prompt_file = tmp_path / f"prompt_{uuid.uuid4().hex[:8]}.txt"
        prompt_text = (
            PROMPT.replace("##CODE##", example["content"])
            .replace("##OUTPUT_FILE##", output_file)
            .strip()
            + "\n"
        )
        prompt_file.write_text(prompt_text, encoding="utf-8")

        out_path_tmp = tmp_path / output_file
        dest_path = example["path"] / output_file

        logging.info("Running OpenHands for %s in isolated dir %s",
                     example["name"], tmp_path)

        rc = run_openhands(prompt_file, tmp_path, stop_check=out_path_tmp.exists)
        if rc != 0:
            logging.error("OpenHands exited with code %s for %s", rc, example["name"])

        waited = 0
        while waited < WAIT_FOR_OUTPUT and not out_path_tmp.exists():
            logging.info("Waiting for %s to be created...", out_path_tmp)
            time.sleep(CHECK_INTERVAL)
            waited += CHECK_INTERVAL

        if not out_path_tmp.exists():
            logging.error("Gave up waiting for %s after %s s", out_path_tmp, WAIT_FOR_OUTPUT)
            return False

        shutil.move(str(out_path_tmp), dest_path)
        logging.info("✓ Fixed file moved to: %s", dest_path.relative_to(Path.cwd()))

        print("-" * 80)
        print(dest_path.read_text(encoding="utf-8"))
        print("-" * 80)
        return True


def silence_pydantic_warnings() -> None:
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    silence_pydantic_warnings()

    project_root = Path(__file__).resolve().parent
    for example in gather_examples(project_root):
        success = process_example(example)
        if not success:
            logging.error("Failed to process %s", example["name"])
        time.sleep(2)

    logging.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user")
        sys.exit(130)
