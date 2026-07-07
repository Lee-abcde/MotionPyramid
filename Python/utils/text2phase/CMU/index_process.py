import csv
import os
from pathlib import Path


def filter_cmu_rows(input_csv, output_csv):
    """Resolve paths dynamically for cross-platform compatibility"""
    # Convert to an absolute path
    input_path = Path(input_csv).resolve()
    output_path = Path(output_csv).resolve()

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check whether the input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    with open(input_path, mode='r', encoding='utf-8') as infile, \
            open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader)
        writer.writerow(headers)

        cmu_count = 0
        for row in reader:
            if '/CMU/' in row[0].replace('\\', '/'):
                writer.writerow(row)
                cmu_count += 1
        print(f"Filtering complete. Found {cmu_count} CMU records")


# Build paths automatically, assuming the script and CSV are in the same directory
script_dir = Path(__file__).parent.resolve()
input_csv = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'humanml3d' / 'index.csv'
output_csv = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'index_cmu.csv'

filter_cmu_rows(input_csv, output_csv)