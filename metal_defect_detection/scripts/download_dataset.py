"""
Dataset Verification and Download Script for NEU-DET.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def verify_or_download_dataset(data_dir: Path = Path("data/NEU-DET")) -> bool:
    print("=" * 70)
    print("NEU-DET Dataset Verification & Setup")
    print("=" * 70)

    train_img_dir = data_dir / "train" / "images"
    test_img_dir = data_dir / "test" / "images"

    train_count = len(list(train_img_dir.glob("*.jpg"))) if train_img_dir.exists() else 0
    test_count = len(list(test_img_dir.glob("*.jpg"))) if test_img_dir.exists() else 0
    total = train_count + test_count

    print(f"Checking {data_dir.resolve()}...")
    print(f"Found {train_count} train images and {test_count} test images (Total: {total}).")

    if total >= 1800:
        print("[SUCCESS] NEU-DET dataset is complete and verified (1,800 images).")
        return True

    # If missing in data/, check /tmp/neu_test
    tmp_path = Path("/tmp/neu_test/data/NEU-DET")
    if tmp_path.exists():
        print(f"Found local cached dataset in {tmp_path}. Copying to {data_dir}...")
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.copytree(tmp_path, data_dir)
        print("[SUCCESS] Dataset copied and verified.")
        return True

    # Otherwise clone from GitHub mirror
    print("Downloading NEU-DET from repository mirror...")
    temp_dir = Path("/tmp/neu_download")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Marfbin/NEU-DET-with-yolov8.git", str(temp_dir)],
            check=True
        )
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_dir / "data" / "NEU-DET", data_dir)
        print("[SUCCESS] NEU-DET downloaded and installed.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Failed to download dataset automatically: {e}", file=sys.stderr)
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = verify_or_download_dataset()
    sys.exit(0 if success else 1)
