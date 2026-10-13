"""Collect dropped documents once through the existing extraction and queue."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from .extract import GLMClient, extract_candidates, write_jsonl
from .queue import latest_records
from .timing import now_et

TEXT_SUFFIXES = {".txt", ".md", ".rst", ".log", ".csv", ".json", ".jsonl"}
DOCUMENT_SUFFIXES = TEXT_SUFFIXES | {".html", ".htm", ".docx", ".pdf"}


class DocumentHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[Any] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: Any, attrs: Any) -> Any:
        if tag in ("script", "style"):
            self.hidden_depth += 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_endtag(self, tag: Any) -> Any:
        if tag in ("script", "style"):
            self.hidden_depth = max(0, self.hidden_depth - 1)
        if tag in ("p", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data: Any) -> Any:
        if not self.hidden_depth:
            self.parts.append(data)


def read_document(path: Any) -> Any:
    """Return document text; unsupported or unreadable inputs fail explicitly."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if suffix in (".html", ".htm"):
        parser = DocumentHTML()
        parser.feed(path.read_text(encoding="utf-8"))
        return "".join(parser.parts).strip()
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            document = ElementTree.fromstring(archive.read("word/document.xml"))
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = []
        for paragraph in document.findall(".//w:p", namespace):
            parts = []
            for element in paragraph.iter():
                if element.tag.endswith("}t") and element.text:
                    parts.append(element.text)
                elif element.tag.endswith("}tab"):
                    parts.append("\t")
                elif element.tag.endswith("}br"):
                    parts.append("\n")
            paragraphs.append("".join(parts))
        return "\n".join(paragraphs)
    if suffix == ".pdf":
        result = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                                capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            raise ValueError("PDF text conversion failed: " + result.stderr.strip())
        if not result.stdout.strip():
            raise ValueError("PDF contains no extractable text; an OCR source is required")
        return result.stdout
    raise ValueError("unsupported document suffix: " + suffix)


def collect_intake(intake_dir: Any, root: Any, *, client: Any, policy: Any = None) -> Any:
    intake_dir = Path(intake_dir).resolve()
    root = Path(root).resolve()
    if not intake_dir.is_dir():
        raise ValueError("intake directory does not exist")
    if root == intake_dir or intake_dir in root.parents:
        raise ValueError("run storage must be outside the intake directory")
    journal = root / "intake.jsonl"
    completed = set()
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            receipt = json.loads(line)
            if receipt["status"] == "collected":
                completed.add(receipt["source_ref"])
    result: dict[Any, Any] = {"collected": [], "skipped": [], "errors": []}
    for path in sorted(intake_dir.rglob("*")):
        relative = path.relative_to(intake_dir)
        hidden = False
        for part in relative.parts:
            if part.startswith("."):
                hidden = True
        if hidden or path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() not in DOCUMENT_SUFFIXES:
            result["skipped"].append({"path": str(relative), "reason": "unsupported suffix"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_ref = "intake:" + relative.as_posix() + ":" + digest
        if source_ref in completed:
            result["skipped"].append({"path": str(relative), "reason": "already collected"})
            continue
        receipt = {"created_et": now_et(), "path": str(relative), "source_ref": source_ref,
                   "sha256": digest, "format": path.suffix.lower()}
        try:
            text = read_document(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("document changed during reading; retry its completed version")
            receipt["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            rows = extract_candidates(text, source_ref, client=client, policy=policy,
                                      queue_path=root / "queue" / "candidates.jsonl",
                                      timings_path=root / "timings.jsonl")
            receipt.update({"status": "collected", "candidate_ids": [], "new_candidate_ids": []})
            for row in rows:
                receipt["new_candidate_ids"].append(row["id"])
            for row in latest_records(root / "queue" / "candidates.jsonl"):
                if row["source"]["ref"] == source_ref:
                    receipt["candidate_ids"].append(row["id"])
            completed.add(source_ref)
            result["collected"].append(receipt)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError,
                zipfile.BadZipFile, ElementTree.ParseError) as error:
            receipt.update({"status": "failed", "error": str(error)})
            result["errors"].append(receipt)
        write_jsonl(journal, receipt)
        if receipt["status"] == "failed":
            # A failed source closes this pass. Preserve partial rows for resume.
            break
    return result


def main(argv: Any = None) -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--policy-config", type=Path)
    args = parser.parse_args(argv)
    policy = None
    if args.policy_config:
        from .policy import load_policy, record_policy

        configuration = load_policy(args.policy_config)
        record_policy(args.root, configuration)
        policy = configuration["extract"]
    client = GLMClient(args.endpoint, args.model, timeout=args.timeout,
                       trace_path=args.root / "traces" / "collection.jsonl")
    result = collect_intake(args.intake_dir, args.root, client=client, policy=policy)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
