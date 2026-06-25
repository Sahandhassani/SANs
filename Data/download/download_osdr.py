import argparse
import subprocess
from pathlib import Path


def run_command(cmd):
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def list_osdr_bucket(study_id):
    cmd = [
        "aws",
        "s3",
        "ls",
        "--no-sign-request",
        "s3://nasa-osdr/",
        "--recursive",
    ]

    print(f"[INFO] Listing NASA OSDR bucket and filtering for {study_id}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    matched = []
    for line in result.stdout.splitlines():
        if study_id in line:
            matched.append(line)

    print(f"[FOUND] {len(matched)} files related to {study_id}")
    for m in matched[:100]:
        print(m)

    return matched


def download_osdr_study(study_id, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = list_osdr_bucket(study_id)

    if len(matched) == 0:
        print(f"[WARNING] No files found for {study_id}.")
        print("You may need to manually inspect the OSDR page or AWS bucket structure.")
        return

    keys = []
    for line in matched:
        parts = line.split()
        if len(parts) >= 4:
            key = parts[-1]
            keys.append(key)

    for key in keys:
        target_path = output_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "aws",
            "s3",
            "cp",
            "--no-sign-request",
            f"s3://nasa-osdr/{key}",
            str(target_path),
        ]

        try:
            run_command(cmd)
        except subprocess.CalledProcessError:
            print(f"[SKIPPED] Could not download: {key}")

    print(f"[DONE] {study_id} files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_id", type=str, default="OSD-679")
    parser.add_argument("--output_dir", type=str, default="Data/raw/OSDR/OSD-679")
    args = parser.parse_args()

    download_osdr_study(args.study_id, args.output_dir)
