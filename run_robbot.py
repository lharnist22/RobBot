import os
import signal
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))


def run_step(cmd: list[str], label: str) -> None:
    print(f"\n==> {label}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    skip_ingest = "--skip-ingest" in sys.argv
    host = os.getenv("HOST", "127.0.0.1")
    port = os.getenv("PORT", "7777")

    try:
        if not skip_ingest:
            run_step([sys.executable, "PDFReader.py"], "Building knowledge_base.jsonl from toIngest PDFs")
            run_step([sys.executable, "build_vector_db.py"], "Rebuilding vector database")
        else:
            print("\n==> Skipping ingest/rebuild and using existing vector database")

        print(f"\n==> Starting RobBot web app on http://{host}:{port}")
        print("Open that URL in your browser.")
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "kb_service:app", "--host", host, "--port", port, "--reload"],
            cwd=ROOT,
            check=True,
        )

    except KeyboardInterrupt:
        print("\nStopping RobBot...")


if __name__ == "__main__":
    if os.name == "nt":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
