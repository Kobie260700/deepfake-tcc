import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotula uma sessao experimental.")
    parser.add_argument("session", type=Path)
    parser.add_argument("--class", dest="label", choices=("real", "fake"), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--platform", default="meet")
    parser.add_argument("--attack-start-seconds", type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    metadata = {
        "label": args.label,
        "scenario": args.scenario,
        "platform": args.platform,
        "attack_start_seconds": args.attack_start_seconds,
        "notes": args.notes,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
    }
    output = args.session / "metadata.json"
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Sessao rotulada em: {output}")


if __name__ == "__main__":
    main()

