import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import re


def parse_motion_txt(txt_path):
    """Parse the motion index file"""
    motion_db = {}
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # Parse fields
            motion_id = int(parts[0])
            orig_start = int(parts[1])
            orig_end = int(parts[2])
            abs_start = int(parts[3])
            abs_end = int(parts[4])
            status = parts[5]
            filename = parts[6]
            filehash = parts[7]

            # Extract CMU id, assuming filename format xx_xx_poses.bvh
            match = re.search(r"(\d+).npy", filename)
            if match:
                index = int(match.group(1))
                key = (index, status)
                motion_db[key] = {
                    'motion_id': motion_id,
                    'orig_range': (orig_start, orig_end),
                    'abs_range': (abs_start, abs_end),
                    'filehash': filehash,
                    'filename': filename
                }
            else:
                print("Error with txt matching names")
                continue

    return motion_db


def convert_frames(index_path, sequence_path, output_path):
    """Main conversion function"""
    index_path = Path(index_path)
    sequence_path = Path(sequence_path)
    output_path = Path(output_path)

    # Load data
    motion_db = parse_motion_txt(sequence_path)
    df = pd.read_csv(index_path)

    results = []

    for _, row in df.iterrows():
        # Extract CMU id
        path = row['new_name']
        match = re.search(r"(\d+).npy", path)
        if not match:
            print("Error with matching names")
            continue

        index_main = int(match.group(1))
        key = (index_main, "Standard")
        key_m = (index_main, "Mirrored")

        # Find matching motion entries
        if key not in motion_db or key_m not in motion_db:
            print("Don't find correspond matching in txt")
            continue

        motion = motion_db[key]
        motion_m = motion_db[key_m]
        # orig_start, orig_end = motion['orig_range']
        # abs_start, abs_end = motion['abs_range']
        # abs_start_m, abs_end_m = motion_m['abs_range']
        abs_start, abs_end = motion['abs_range'][0], \
            motion['abs_range'][1] + 1
        abs_start_m, abs_end_m = motion_m['abs_range'][0],  \
            motion_m['abs_range'][1] + 1

        # Run frame conversion
        # def convert(frame):
        #     if frame < orig_start or frame > orig_end:
        #         raise ValueError(f"Frame {frame} is outside the original range [{orig_start}, {orig_end}]")
        #     return abs_start + (frame - orig_start)

        try:
            # global_start = convert(row['start_frame'])
            # global_end = convert(row['end_frame'])

            results.append({
                'key': row['new_name'].rstrip(".npy"),
                'new_name': row['new_name'].replace(".npy", ".txt"),
                'source_path': path,
                'global_start': abs_start,
                'global_end': abs_end,
                'global_start_m': abs_start_m,
                'global_end_m': abs_end_m,
                'motion_id': int(motion_m['motion_id'])/2,
                # 'cmu_main': cmu_main,
                # 'cmu_sub': cmu_sub,
                'filehash': motion['filehash']
            })

        except ValueError as e:
            print(f"Skipping invalid data row: {e}")

    # Saveresult
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Conversion complete. Result saved to {output_path}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent.resolve()
    default_dataset_path = script_dir.parent.parent.parent / 'Datasets' / 'HumanML3DwithRoot_Text'

    parser = argparse.ArgumentParser(
        description='Build HumanML3D text-to-phase metadata from index.csv and processed Sequences.txt.'
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=default_dataset_path / 'index.csv',
        help='Path to HumanML3D index.csv.',
    )
    parser.add_argument(
        '--sequence',
        type=Path,
        default=default_dataset_path / 'Sequences.txt',
        help='Path to the processed text-to-phase Sequences.txt.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=default_dataset_path / 'text2phase.csv',
        help='Path for the generated text2phase.csv.',
    )
    args = parser.parse_args()

    convert_frames(args.index, args.sequence, args.output)
