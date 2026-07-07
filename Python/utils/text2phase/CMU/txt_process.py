import csv
from pathlib import Path


def process_files(csv_path, txt_path, output_path):
    try:
        # Read CSV
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

        with csv_path.open('r') as f:
            reader = csv.DictReader(f)
            csv_nums = {row['new_name'].split('.')[0] for row in reader}

        # Read TXT
        txt_path = Path(txt_path)
        if not txt_path.exists():
            raise FileNotFoundError(f"TXT file does not exist: {txt_path}")

        with txt_path.open('r') as f:
            txt_nums = {line.strip() for line in f if line.strip()}

        # Compute intersection
        common = sorted(txt_nums & csv_nums)

        # Write result
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as f:
            f.write('\n'.join(common))

        print(f"Successfully matched {len(common)} numbers")

    except Exception as e:
        print(f"Processing failed: {str(e)}")


from pathlib import Path

script_dir = Path(__file__).parent.resolve()

csv_path = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'index_cmu.csv'
txt_path = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'humanml3d' / 'val.txt'
output_path = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'val.txt'

process_files(
    csv_path=csv_path,
    txt_path=txt_path,
    output_path=output_path
)