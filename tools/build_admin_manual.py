"""Render the manual from portal help and check setting coverage against OpenAPI."""

import argparse
import json
from pathlib import Path

SCHEMA_GROUPS = {
    "RuleCreate": "rules", "RuleEdit": "rules", "Preset": "preset",
    "RuleCheck": "rule_check", "QueueEdit": "queue", "Approve": "queue",
    "Reject": "queue", "PolicyEdit": "policy", "Learn": "learn",
    "Control": "control", "Schedule": "schedule", "DreamCreate": "dream",
    "DreamCheck": "dream_check", "PocketCreate": "pocket", "PocketImport": "pocket",
    "PocketExtract": "pocket", "PocketCopy": "pocket", "PocketMerge": "pocket",
    "PocketRollover": "pocket", "Delete": "revision",
}


def check_content(content, spec=None, policy=None):
    pages = {}
    topics = {}
    fields = set()
    for page in content["pages"]:
        if page["id"] in pages:
            raise ValueError("Duplicate page: " + page["id"])
        pages[page["id"]] = page
    if set(pages) != {"rules", "queue", "dreams", "pockets", "autonomy", "system"}:
        raise ValueError("The manual requires all six task pages")
    for topic in content["topics"]:
        if topic["id"] in topics or topic["page"] not in pages:
            raise ValueError("Invalid topic identity: " + topic["id"])
        topics[topic["id"]] = topic
        for field in topic["fields"]:
            if field in fields:
                raise ValueError("Setting explained twice: " + field)
            fields.add(field)
    referenced = set()
    for page in pages.values():
        for topic_id in page["topic_ids"]:
            if topic_id not in topics or topics[topic_id]["page"] != page["id"]:
                raise ValueError("Invalid page topic: " + topic_id)
            if topic_id in referenced:
                raise ValueError("Topic appears twice: " + topic_id)
            referenced.add(topic_id)
        for layer in ("simple", "advanced"):
            if not page[layer]:
                raise ValueError("Missing task path: " + page["id"])
    if referenced != set(topics):
        raise ValueError("The manual has unused topics")
    raw = json.dumps(content, ensure_ascii=False)
    if "\u2014" in raw:
        raise ValueError("Use ordinary punctuation")
    if spec is not None:
        requests = set()
        for operations in spec["paths"].values():
            for operation in operations.values():
                body = operation.get("requestBody")
                if body is not None:
                    reference = body["content"]["application/json"]["schema"]["$ref"]
                    requests.add(reference.split("/")[-1])
        schemas = spec["components"]["schemas"]
        for name in sorted(requests):
            if name not in SCHEMA_GROUPS:
                raise ValueError("Review new request schema: " + name)
            for field in schemas[name]["properties"]:
                key = SCHEMA_GROUPS[name] + "." + field
                if field == "id":
                    key = "identity.id"
                if field == "expected_revision":
                    key = "revision.expected_revision"
                if key not in fields:
                    raise ValueError("Unexplained API field: " + key)
        window = schemas["Schedule"]["properties"]["windows"]["items"]
        for field in window["properties"]:
            if "window." + field not in fields:
                raise ValueError("Unexplained window field: " + field)
    if policy is not None:
        for group in ("extract", "grade", "control"):
            for field in policy[group]:
                if group + "." + field not in fields:
                    raise ValueError("Unexplained policy field: " + group + "." + field)
    return pages, topics, fields


def render(content, output):
    pages, topics, _ = check_content(content)
    output.mkdir(parents=True, exist_ok=True)
    index = ["# RSI Dreambox manual", "", "Simple is the default task view. Advanced opens the full settings and raw details. Both use the same API state and admin authority.", ""]
    for page in pages.values():
        index.append("- [" + page["title"] + "](" + page["id"] + ".md)")
        lines = ["# " + page["title"], "", page["intro"], ""]
        for layer in ("simple", "advanced"):
            lines.extend(["## " + layer.title(), ""])
            for section in page[layer]:
                lines.extend(["### " + section["title"], "", section["body"], ""])
        lines.extend(["## Settings and details", ""])
        for topic_id in page["topic_ids"]:
            topic = topics[topic_id]
            lines.extend(["### " + topic["label"], "", topic["body"], ""])
        lines.extend(["[All pages](README.md)", ""])
        (output / (page["id"] + ".md")).write_text("\n".join(lines))
    index.extend(["", "[Dream operator runbook](../DREAM_OPERATOR_RUNBOOK.md)", "", "These pages are generated from the portal's help.json. Edit that file, then run tools/build_admin_manual.py. The portal and manual share each setting explanation.", ""])
    (output / "README.md").write_text("\n".join(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    help_path = args.root / "src/glm_rsi_dream/resources/admin_portal/help.json"
    content = json.loads(help_path.read_text())
    spec = json.loads(args.spec.read_text()) if args.spec else None
    policy = json.loads(args.policy.read_text()) if args.policy else None
    pages, topics, fields = check_content(content, spec, policy)
    if not args.check:
        render(content, args.root / "docs/admin")
    print(json.dumps({"pages": len(pages), "topics": len(topics), "fields": len(fields), "api_coverage": spec is not None}))


if __name__ == "__main__":
    main()
