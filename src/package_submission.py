"""Assemble the final submission zip in the layout required by the problem statement.

    python -m src.package_submission --team MyTeam

<team>_submission.zip
├── output/matching_results.tsv, output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md
"""

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    args = parser.parse_args()

    files = {
        "output/matching_results.tsv": PROJECT_ROOT / "output" / "matching_results.tsv",
        "output/candidate_pairs.tsv": PROJECT_ROOT / "output" / "candidate_pairs.tsv",
        "code/business_entity_resolution/README.md": PROJECT_ROOT / "submission_README.md",
        "code/business_entity_resolution/requirements.txt": PROJECT_ROOT / "requirements.txt",
        "Documentation_template.md": PROJECT_ROOT / "Documentation_template.md",
    }
    for source in sorted((PROJECT_ROOT / "src").glob("*.py")):
        files[f"code/business_entity_resolution/src/{source.name}"] = source

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")

    out_dir = PROJECT_ROOT / "submission"
    out_dir.mkdir(exist_ok=True)
    zip_path = out_dir / f"{args.team}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source in files.items():
            archive.write(source, arcname=name)
            print("added", name)
    print(f"wrote {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
