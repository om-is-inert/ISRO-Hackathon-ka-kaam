"""
Download GOES-19 ABI Channel 13 (TIR 10.3μm) data from public AWS S3.
No authentication required — fully open data.

Usage:
    python scripts/download_goes19.py --date 2025-06-01 --hours 24 --output data/goes19/
"""
import os, argparse
from datetime import datetime, timedelta
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description='Download GOES-19 Ch13 from AWS S3')
    p.add_argument('--date', type=str, default='2025-06-01', help='Start date YYYY-MM-DD')
    p.add_argument('--hours', type=int, default=24, help='Hours to download')
    p.add_argument('--output', type=str, default='data/goes19')
    p.add_argument('--channel', type=int, default=13, help='ABI channel (13=TIR)')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    try:
        import s3fs
    except ImportError:
        print("Installing s3fs...")
        os.system('pip install s3fs')
        import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    dt = datetime.strptime(args.date, '%Y-%m-%d')
    ch = f'C{args.channel:02d}'
    downloaded = 0

    for hour in range(args.hours):
        current = dt + timedelta(hours=hour)
        day_of_year = current.timetuple().tm_yday
        year = current.year
        hr = current.hour

        # GOES-19 ABI L1b Full-Disk, Mode 6
        prefix = f'noaa-goes19/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hr:02d}/'

        try:
            files = fs.glob(f'{prefix}OR_ABI-L1b-RadF-M6{ch}_G19_*.nc')
        except Exception as e:
            # Try GOES-16 as fallback (more data available)
            prefix = f'noaa-goes16/ABI-L1b-RadF/{year}/{day_of_year:03d}/{hr:02d}/'
            try:
                files = fs.glob(f'{prefix}OR_ABI-L1b-RadF-M6{ch}_G16_*.nc')
            except Exception:
                print(f"  No data for {current}")
                continue

        for f in files:
            fname = Path(f).name
            local_path = os.path.join(args.output, fname)
            if os.path.exists(local_path):
                continue
            try:
                fs.get(f, local_path)
                downloaded += 1
                print(f"  [{downloaded}] {fname}")
            except Exception as e:
                print(f"  Error: {fname}: {e}")

    print(f"\nDownloaded {downloaded} files to {args.output}")
    print(f"Expected ~6 files/hour × {args.hours} hours = ~{6*args.hours} files")

if __name__ == '__main__':
    main()
