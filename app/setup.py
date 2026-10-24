"""Download the manual, extract its text, embed the paragraphs."""

import shutil
import subprocess
import urllib.request
from pathlib import Path

from app import config as cfg
from app import corpus, embed

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "data" / "ac43-13-1b.pdf"
TXT = ROOT / "data" / "ac43-13-1b.txt"
URL = cfg.CORPUS_URL


def main():
    PDF.parent.mkdir(exist_ok=True)
    if not PDF.exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as reply, PDF.open("wb") as out:
            shutil.copyfileobj(reply, out)
    if not TXT.exists():
        subprocess.run(["pdftotext", "-raw", str(PDF), str(TXT)], check=True)
    print(f"paragraphs: {len(list(corpus.paragraphs()))}")
    embed.main()


if __name__ == "__main__":
    main()
