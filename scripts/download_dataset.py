#!/usr/bin/env python3
"""
Dataset Download Script
=======================
Download and prepare common TTS datasets.
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm


DATASETS = {
    'ljspeech': {
        'url': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        'filename': 'LJSpeech-1.1.tar.bz2',
        'extract_name': 'LJSpeech-1.1',
        'description': 'LJSpeech - 24 hours, single speaker English',
        'size': '2.6 GB'
    },
    'vctk': {
        'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip',
        'filename': 'VCTK-Corpus-0.92.zip',
        'extract_name': 'VCTK-Corpus-0.92',
        'description': 'VCTK - 110 speakers, various accents',
        'size': '11 GB'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = None):
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract archive file."""
    print(f"Extracting {archive_path}...")

    if archive_path.suffix == '.bz2' or str(archive_path).endswith('.tar.bz2'):
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(extract_dir)
    elif archive_path.suffix == '.gz' or str(archive_path).endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def download_ljspeech(data_dir: Path):
    """Download and prepare LJSpeech dataset."""
    dataset = DATASETS['ljspeech']

    output_dir = data_dir / 'ljspeech'
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / dataset['filename']

    # Download if needed
    if not archive_path.exists():
        print(f"Downloading LJSpeech ({dataset['size']})...")
        download_file(dataset['url'], archive_path, desc='LJSpeech')
    else:
        print(f"Archive already exists: {archive_path}")

    # Extract
    extract_dir = output_dir / dataset['extract_name']
    if not extract_dir.exists():
        extract_archive(archive_path, output_dir)
        print(f"Extracted to: {extract_dir}")
    else:
        print(f"Already extracted: {extract_dir}")

    return extract_dir


def download_vctk(data_dir: Path):
    """Download and prepare VCTK dataset."""
    dataset = DATASETS['vctk']

    output_dir = data_dir / 'vctk'
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir / dataset['filename']

    # Download if needed
    if not archive_path.exists():
        print(f"Downloading VCTK ({dataset['size']})...")
        print("Note: This is a large dataset and may take a while.")
        download_file(dataset['url'], archive_path, desc='VCTK')
    else:
        print(f"Archive already exists: {archive_path}")

    # Extract
    extract_dir = output_dir / dataset['extract_name']
    if not extract_dir.exists():
        extract_archive(archive_path, output_dir)
        print(f"Extracted to: {extract_dir}")
    else:
        print(f"Already extracted: {extract_dir}")

    return extract_dir


def list_datasets():
    """List available datasets."""
    print("\nAvailable datasets:")
    print("-" * 60)
    for name, info in DATASETS.items():
        print(f"\n  {name}:")
        print(f"    Description: {info['description']}")
        print(f"    Size: {info['size']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Download TTS datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py ljspeech --data-dir ./data
  python download_dataset.py vctk --data-dir ./data
  python download_dataset.py --list
        """
    )

    parser.add_argument(
        'dataset',
        nargs='?',
        choices=['ljspeech', 'vctk'],
        help='Dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to save datasets'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset is None:
        parser.print_help()
        print("\nError: Please specify a dataset or use --list")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'ljspeech':
        download_ljspeech(data_dir)
    elif args.dataset == 'vctk':
        download_vctk(data_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
