import argparse
import shutil
from pathlib import Path

import kagglehub


def copy_dataset(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] KaggleHub downloaded dataset to: {src_dir}")
    print(f"[INFO] Copying dataset to: {dst_dir}")

    for item in src_dir.iterdir():
        target = dst_dir / item.name

        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print("[DONE] Dataset copied successfully.")


def download_kermany(output_dir):
    output_dir = Path(output_dir)

    print("[INFO] Downloading Kermany/OCT2017 dataset with kagglehub...")
    kagglehub_path = kagglehub.dataset_download("paultimothymooney/kermany2018")

    copy_dataset(kagglehub_path, output_dir)

    print(f"[DONE] Kermany dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/raw/Kermany2018",
        help="Directory to save Kermany OCT dataset",
    )
    args = parser.parse_args()

    download_kermany(args.output_dir)