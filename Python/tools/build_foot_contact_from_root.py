import argparse
from pathlib import Path
import time

import numpy as np


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "Datasets" / "HumanML3DwithRoot"


def read_description(dataset_path):
    description_path = dataset_path / "Description.txt"
    with description_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    channel_names = lines[0].split(",")
    channel_dims = [int(x) for x in lines[1].split(",")]
    n_frames = int(lines[2])
    bone_names = lines[3].split(",") if len(lines) > 3 and lines[3] else []
    fps = int(lines[5]) if len(lines) > 5 and lines[5] else 60
    return channel_names, channel_dims, n_frames, bone_names, fps


def channel_slices(channel_names, channel_dims):
    slices = {}
    offset = 0
    for name, dim in zip(channel_names, channel_dims):
        slices[name] = slice(offset, offset + dim)
        offset += dim
    return slices, offset


def find_bone_indices(bone_names):
    lower_to_index = {name.lower(): i for i, name in enumerate(bone_names)}

    def find_pair(primary_names, fallback_indices):
        indices = []
        for name in primary_names:
            idx = lower_to_index.get(name.lower())
            if idx is not None:
                indices.append(idx)
        if len(indices) == len(primary_names):
            return indices
        return fallback_indices

    left = find_pair(("ltoe", "ltoeSite"), [4, 5])
    right = find_pair(("rtoe", "rtoeSite"), [9, 10])
    return right, left


def read_sequence_resets(sequences_path, n_frames):
    reset = np.zeros(n_frames, dtype=bool)
    unresolved_starts = 0
    prev_sequence = None
    prev_frame = None
    prev_mirror = None

    with sequences_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n_frames:
                break
            sequence_text, frame_text, mirror, *_ = line.split(maxsplit=3)
            sequence = int(sequence_text)
            frame = int(frame_text)

            is_reset = (
                idx == 0
                or sequence != prev_sequence
                or mirror != prev_mirror
                or frame != prev_frame + 1
            )
            reset[idx] = is_reset
            if is_reset and frame != 1:
                unresolved_starts += 1

            prev_sequence = sequence
            prev_frame = frame
            prev_mirror = mirror

    return reset, unresolved_starts


def compute_contacts_from_features(
    dataset_path,
    speed_threshold=0.35,
    position_threshold=0.05,
    chunk_size=262_144,
):
    channel_names, channel_dims, n_frames, bone_names, fps = read_description(dataset_path)
    slices, feature_count = channel_slices(channel_names, channel_dims)

    required = ["Velocities", "Positions"]
    missing = [name for name in required if name not in slices]
    if missing:
        raise ValueError(f"Dataset is missing required channels: {missing}")

    data_path = dataset_path / "Data.bin"
    expected_size = n_frames * feature_count * np.dtype(np.float32).itemsize
    actual_size = data_path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f"{data_path} is too small: expected {expected_size}, got {actual_size}")

    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(n_frames, feature_count))
    right_indices, left_indices = find_bone_indices(bone_names)
    foot_indices = right_indices + left_indices

    velocity_slice = slices["Velocities"]
    position_slice = slices["Positions"]
    contacts = np.empty((n_frames, 2), dtype=np.int64)

    start_time = time.time()
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        rows = data[start:end]

        foot_velocity = rows[:, velocity_slice].reshape(-1, len(bone_names), 3)[:, foot_indices, :]
        foot_position = rows[:, position_slice].reshape(-1, len(bone_names), 3)[:, foot_indices, :]

        right_speed = np.linalg.norm(foot_velocity[:, 0:2, :], axis=2).max(axis=1)
        left_speed = np.linalg.norm(foot_velocity[:, 2:4, :], axis=2).max(axis=1)
        right_height = foot_position[:, 0:2, 1]
        left_height = foot_position[:, 2:4, 1]

        contacts[start:end, 0] = (
            (right_speed < speed_threshold)
            & (right_height < position_threshold).all(axis=1)
        )
        contacts[start:end, 1] = (
            (left_speed < speed_threshold)
            & (left_height < position_threshold).all(axis=1)
        )

        processed = end
        elapsed = max(time.time() - start_time, 1e-6)
        print(
            f"Processed {processed}/{n_frames} frames "
            f"({processed / n_frames:.1%}, {processed / elapsed:.0f} frames/s)"
        )

    return contacts, {
        "method": "direct Velocities+Positions",
        "fps": fps,
        "right_indices": right_indices,
        "left_indices": left_indices,
        "unresolved_sequence_starts": 0,
    }


def compute_contacts_from_root_reconstruction(
    dataset_path,
    speed_threshold=0.35,
    position_threshold=0.05,
    chunk_size=262_144,
):
    channel_names, channel_dims, n_frames, bone_names, fps = read_description(dataset_path)
    slices, feature_count = channel_slices(channel_names, channel_dims)

    required = ["Positions", "RootPositions", "RootRotations"]
    missing = [name for name in required if name not in slices]
    if missing:
        raise ValueError(f"Dataset is missing required channels: {missing}")

    data_path = dataset_path / "Data.bin"
    expected_size = n_frames * feature_count * np.dtype(np.float32).itemsize
    actual_size = data_path.stat().st_size
    if actual_size < expected_size:
        raise ValueError(f"{data_path} is too small: expected {expected_size}, got {actual_size}")

    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(n_frames, feature_count))
    sequences_path = dataset_path / "Sequences.txt"
    if sequences_path.exists():
        reset_mask, unresolved_starts = read_sequence_resets(sequences_path, n_frames)
    else:
        reset_mask = np.zeros(n_frames, dtype=bool)
        reset_mask[0] = True
        unresolved_starts = 0

    right_indices, left_indices = find_bone_indices(bone_names)
    foot_indices = right_indices + left_indices

    pos_slice = slices["Positions"]
    root_pos_slice = slices["RootPositions"]
    root_rot_slice = slices["RootRotations"]

    contacts = np.empty((n_frames, 2), dtype=np.int64)
    dt = 1.0 / float(fps)
    last_global_foot_pos = None

    start_time = time.time()
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        rows = data[start:end]

        local_positions = rows[:, pos_slice].reshape(-1, len(bone_names), 3)[:, foot_indices, :]
        root_positions = rows[:, root_pos_slice]
        root_rotations = rows[:, root_rot_slice].reshape(-1, 3, 3)

        global_foot_pos = (
            np.einsum("nij,nkj->nki", root_rotations, local_positions)
            + root_positions[:, None, :]
        )

        prev_global_foot_pos = np.empty_like(global_foot_pos)
        if last_global_foot_pos is None:
            prev_global_foot_pos[0] = global_foot_pos[0]
        else:
            prev_global_foot_pos[0] = last_global_foot_pos
        prev_global_foot_pos[1:] = global_foot_pos[:-1]

        local_reset = reset_mask[start:end]
        prev_global_foot_pos[local_reset] = global_foot_pos[local_reset]

        foot_velocity = (global_foot_pos - prev_global_foot_pos) / dt

        right_speed = np.linalg.norm(foot_velocity[:, 0:2, :], axis=2).max(axis=1)
        left_speed = np.linalg.norm(foot_velocity[:, 2:4, :], axis=2).max(axis=1)
        right_height = global_foot_pos[:, 0:2, 1]
        left_height = global_foot_pos[:, 2:4, 1]

        contacts[start:end, 0] = (
            (right_speed < speed_threshold)
            & (right_height < position_threshold).all(axis=1)
        )
        contacts[start:end, 1] = (
            (left_speed < speed_threshold)
            & (left_height < position_threshold).all(axis=1)
        )

        last_global_foot_pos = global_foot_pos[-1].copy()

        processed = end
        elapsed = max(time.time() - start_time, 1e-6)
        print(
            f"Processed {processed}/{n_frames} frames "
            f"({processed / n_frames:.1%}, {processed / elapsed:.0f} frames/s)"
        )

    return contacts, {
        "method": "root reconstruction",
        "fps": fps,
        "right_indices": right_indices,
        "left_indices": left_indices,
        "unresolved_sequence_starts": unresolved_starts,
    }


def compare_contacts(generated, reference_path):
    reference = np.load(reference_path)["foot_contact"]
    if generated.shape != reference.shape:
        return {
            "same": False,
            "shape_mismatch": (generated.shape, reference.shape),
        }

    mismatch = generated != reference
    mismatch_count = int(mismatch.sum())
    result = {
        "same": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "total_values": int(generated.size),
        "right_mismatch_count": int(mismatch[:, 0].sum()),
        "left_mismatch_count": int(mismatch[:, 1].sum()),
    }

    if mismatch_count:
        rows, cols = np.where(mismatch)
        result["first_mismatches"] = [
            {
                "frame": int(rows[i]),
                "foot": "right" if cols[i] == 0 else "left",
                "generated": int(generated[rows[i], cols[i]]),
                "reference": int(reference[rows[i], cols[i]]),
            }
            for i in range(min(10, mismatch_count))
        ]

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build foot_contact_results.npz from HumanML3DwithRoot-style data. "
            "The default method uses Velocities and Positions, matching dataset.py."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <dataset>/foot_contact_results.npz",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional reference foot_contact_results.npz for debugging comparisons.",
    )
    parser.add_argument("--speed-threshold", type=float, default=0.35)
    parser.add_argument("--position-threshold", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=262_144)
    parser.add_argument(
        "--method",
        choices=["direct", "root-global"],
        default="direct",
        help=(
            "direct matches the existing dataset.py/global label calculation by using "
            "Velocities and Positions. root-global reconstructs world-space toe positions "
            "from RootPositions/RootRotations and is mainly for debugging."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset.resolve()
    output_path = args.output.resolve() if args.output else dataset_path / "foot_contact_results.npz"

    if args.method == "direct":
        contacts, info = compute_contacts_from_features(
            dataset_path,
            speed_threshold=args.speed_threshold,
            position_threshold=args.position_threshold,
            chunk_size=args.chunk_size,
        )
    else:
        contacts, info = compute_contacts_from_root_reconstruction(
            dataset_path,
            speed_threshold=args.speed_threshold,
            position_threshold=args.position_threshold,
            chunk_size=args.chunk_size,
        )

    print("Foot contact metadata:")
    print(f"  method: {info['method']}")
    print(f"  fps: {info['fps']}")
    print(f"  right foot bone indices: {info['right_indices']}")
    print(f"  left foot bone indices: {info['left_indices']}")
    print(f"  unresolved sequence starts: {info['unresolved_sequence_starts']}")

    if args.compare:
        compare_path = args.compare.resolve()
        result = compare_contacts(contacts, compare_path)
        print(f"Compared with: {compare_path}")
        if result.get("shape_mismatch"):
            print(f"Shape mismatch: generated={result['shape_mismatch'][0]}, reference={result['shape_mismatch'][1]}")
        else:
            print(f"Same labels: {result['same']}")
            print(f"Mismatched values: {result['mismatch_count']} / {result['total_values']}")
            print(f"Right mismatches: {result['right_mismatch_count']}")
            print(f"Left mismatches: {result['left_mismatch_count']}")
            if result.get("first_mismatches"):
                print("First mismatches:")
                for item in result["first_mismatches"]:
                    print(
                        f"  frame={item['frame']} foot={item['foot']} "
                        f"generated={item['generated']} reference={item['reference']}"
                    )

    if not args.no_save:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")
        np.savez(output_path, foot_contact=contacts)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
