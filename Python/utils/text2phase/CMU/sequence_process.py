def process_motion_file(input_path, output_path):
    """
    Process motion data files, compress frame intervals, and add absolute frame indices
    Parameters:
        input_path: input file path
        output_path: output file path
    """
    motion_groups = {}

    # Stage 1: read and group data
    # line_count = 0
    with open(input_path, 'r') as f:
        for line in f:
            # line_count += 1
            if not line.strip():
                print("Error!")
                continue

            parts = line.strip().split()
            if len(parts) < 5:
                print("Error!")
                continue

            try:
                motion_id = int(parts[0])
                frame_num = int(parts[1])
                status = parts[2]
                filename = ' '.join(parts[3:-1])  # Handle filenames containing spaces
                hash_value = parts[-1]
            except (ValueError, IndexError):
                continue

            # Update frame interval
            if motion_id not in motion_groups:
                motion_groups[motion_id] = {
                    'status': status,
                    'filename': filename,
                    'hash': hash_value,
                    'start': frame_num,
                    'end': frame_num
                }
            else:
                motion_groups[motion_id]['start'] = min(
                    motion_groups[motion_id]['start'], frame_num)
                motion_groups[motion_id]['end'] = max(
                    motion_groups[motion_id]['end'], frame_num)

    # Stage 2: compute absolute frame indices
    sorted_ids = sorted(motion_groups.keys())
    current_absolute = 0  # absolute-frame counter

    # Compute absolute frame interval for each action
    for motion_id in sorted_ids:
        group = motion_groups[motion_id]
        frame_count = group['end'] - group['start'] + 1
        group['abs_start'] = current_absolute
        group['abs_end'] = current_absolute + frame_count - 1
        current_absolute += frame_count  # Update global counter

    # Stage 3: write results
    with open(output_path, 'w') as f:
        for motion_id in sorted_ids:
            group = motion_groups[motion_id]
            f.write(
                f"{motion_id} "
                f"{group['start']} {group['end']} "  # original frame interval
                f"{group['abs_start']} {group['abs_end']} "  # absolute frame interval
                f"{group['status']} {group['filename']} {group['hash']}\n"
            )


from pathlib import Path

if __name__ == '__main__':
    script_dir = Path(__file__).parent.resolve()

    input_path = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'humanml3d' / 'Sequences.txt'
    output_path = script_dir.parent.parent.parent / 'Datasets' / 'CMU2withRoot_Text' / 'Sequences.txt'

    process_motion_file(input_path, output_path)