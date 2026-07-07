import torch
import Library.Utility as utility
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
from tqdm import tqdm
import pandas as pd
from os.path import join as pjoin
import random
import codecs as cs
import clip
import os
import pickle
import torch.nn.functional as F
from utils.npz_writer import write_motion2npz
def compare_floating_arrays_with_indices(arr1, arr2, rtol=1e-05, atol=1e-06):
    # First check whether the shapes match
    if arr1.shape != arr2.shape:
        return False, "Arrays have different shapes."

    # Use np.allclose to check whether elements are equal within relative or absolute tolerance
    close_mask = np.isclose(arr1, arr2, rtol=rtol, atol=atol)

    # If all elements are equal within the tolerance
    if np.all(close_mask):
        return True, "Arrays are equal within the given tolerance."

    # Find positions of elements outside the tolerance
    unequal_indices = np.where(~close_mask)

    return False, unequal_indices


def create_dataset_from_args(args):
    paths = args.load.split(',') # divide different dataset
    motion_data = []
    # FeatureCombinedData is a custom dataset class
    for path in paths:
        motion_data.append(FeatureCombinedData(path, args.window, args.normalize, args.test_sequence_ratio,
                                               args.std_cap, args.extra_frames,
                                               needed_channel_names='all'))
    # A = motion_data[0].Data
    # B = motion_data[1].Data
    # ans, indice = compare_floating_arrays_with_indices(A, B)
    return motion_data

def create_mdm_dataset_from_args(args, diff_args):
    paths = diff_args.load.split(',') # divide different dataset
    motion_data = []
    for i, path in enumerate(paths):
        manifold_path = diff_args.pretrained_save + "/Manifolds_" + str(i) + "_final.npz"
        motion_data.append(FeatureCombinedData(path, diff_args.window, args.normalize, args.test_sequence_ratio,
                                               diff_args.std_cap, args.extra_frames,
                                               needed_channel_names='all', pretrained_manifold=True,
                                               pretrained_manifold_path=manifold_path))
    return motion_data


def create_txt2phase_dataset_from_args(args, diff_args, dataset_mode='train'):
    paths = diff_args.load.split(',')  # divide different dataset
    motion_data = []
    for i, path in enumerate(paths):
        manifold_path = diff_args.pretrained_save + "/Manifolds_" + str(i) + "_final.npz"
        motion_data.append(Text2PhaseData(path, args.normalize,
                                          diff_args.std_cap, args.num_embed_vq,
                                          needed_channel_names='all',
                                          pretrained_manifold_path=manifold_path, mode=dataset_mode))
    return motion_data


def get_shape(Load):
    try:
        return utility.LoadTxtAsInt(Load + "/DataShape.txt")
    except:
        _, d1, d0 = get_combined_shape(Load)
        return np.array([d0, sum(d1)])


def check_path(path):
    if 'Datasets' not in path:
        path = osp.join('Datasets', path)
    return path


def get_combined_shape(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    channel_names = lines[0].strip().split(',')
    channel_dims = [int(x) for x in lines[1].strip().split(',')]
    n_frames = int(lines[2].strip())
    return channel_names, channel_dims, n_frames


def get_fps(prefix):
    filename = osp.join(prefix, 'Description.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines) < 6:
        return 60
    return int(lines[5].strip())

def find_similar_motion(dataset, motion_npz_path, start=0, end=30):
    data = np.load(motion_npz_path)

    positions = data['Positions'][:, start:end]
    velocities = data['Velocities'][:, start:end]
    rotations = data['Rotations'][:, start:end]

    motion_clip = np.concatenate([velocities, positions, rotations], axis=0)
    motion_clip = np.swapaxes(motion_clip, 0, 1)
    min_distance = float('inf')
    best_match_index = -1

    window_size = end - start
    for i in tqdm(range(dataset.shape[0] - window_size + 1), desc="Sliding window progress"):
        # Extract the current window segment
        current_clip = dataset[i:i + window_size, :]  # shape is (30, 405)

        distance = np.linalg.norm(current_clip - motion_clip)

        if distance < min_distance:
            min_distance = distance
            best_match_index = i

    return best_match_index, min_distance

def load_single_dataset_bin(path, normalize, needed_feature_names, std_cap):
    print("Start to load binary file")
    channel_names, channel_dims, n_frames = get_combined_shape(path)
    if needed_feature_names == None or needed_feature_names == 'all':
        needed_feature_names = channel_names
    shape = (n_frames, sum(channel_dims))
    data = path + "/Data.bin"
    data = utility.ReadBinary(data, shape[0], shape[1])

    # Reorder the data channel according to the needed_feature_names
    if channel_names != needed_feature_names:
        print("We need only part of the whole channels")
        named_data = {}
        for i, name in enumerate(channel_names):
            named_data[name] = data[:, :channel_dims[i]]
            data = data[:, channel_dims[i]:]
        assert data.shape[-1] == 0

        channel_dims = []
        data = []
        for name in needed_feature_names:
            data.append(named_data[name])
            channel_dims.append(named_data[name].shape[-1])
        data = np.concatenate(data, axis=-1)

    # find_similar_motion(data, 'results/difftest101/generate/df6_motion_10.npz')
    cache_file = os.path.join(path, "cache", "data_stats.npz")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print("✅ Loading cached data_mean and data_std...")
        stats = np.load(cache_file)
        data_mean = stats["mean"]
        data_std = stats["std"]
    else:
        print("🔄 Start to compute mean and std from binary data...")
        data_std = data.std(axis=0)
        set_std_cap(data_std, std_cap)
        data_mean = data.mean(axis=0)
        if not normalize:
            data_std[:] = 1.0
            data_mean[:] = 0.0
        # we cancel normalize root rotation (use if to avoid without root dataset)
        if "Root" in path:
            print("Warning: Using Irregular code to stop normalization the root module")
            data_std[-12:] = 1.0
            data_mean[-12:] = 0.0
        np.savez(cache_file, mean=data_mean, std=data_std)
        print("✅ Mean and std saved to cache.")

    if "global" in path:
        print("Start to compute the foot contact!")
        foot_contact = compute_foot_contact(data)
        output_file = path + "/foot_contact_results.npz"
        np.savez(output_file, foot_contact=foot_contact)

        print(f"Result saved to {output_file}")

    data = (data - data_mean) / data_std
    # test_motion = torch.from_numpy(data).clone()
    # write_motion2npz(test_motion, data_std, data_mean, (36575-528, 36575),
    #                  "results/visualnpz/model_motion_gt.npz")
    return data, data_mean, data_std, channel_dims, needed_feature_names


def compute_foot_contact(data, speed_threshold=0.35, position_threshold=0.05):
    """
    Compute whether the left and right feet are in contact with the ground.

    Parameters:
        data: numpy.ndarray, shape=(151338, 420), input array
        speed_threshold: float, velocity-norm threshold; values below this are treated as near zero
        position_threshold: float, position threshold; values below this are treated as near the ground

    Returns:
        numpy.ndarray, shape=(151338, 2), returned contact state, 1 means contact and 0 means no contact
    """
    motion_emb_length = data.shape[1]
    velocity_emb_length = int(motion_emb_length/5)
    pos_emb_length = int(motion_emb_length/5)
    if motion_emb_length == 420:
    # Extract joint velocities and positions
        joint_speed = data[:, :84]
        joint_position = data[:, 84:168]

        # Extract velocities and positions of right-foot and left-foot joints
        right_foot_speed = joint_speed[:, 63:69].reshape(-1, 2, 3)  # convert to [151338, 2, 3], two groups per row [vx, vy, vz]
        right_foot_y_position = joint_position[:, 63:69][:, 1::3]  # extract y-axis data

        left_foot_speed = joint_speed[:, 78:84].reshape(-1, 2, 3)  # same for left-foot velocity
        left_foot_y_position = joint_position[:, 78:84][:, 1::3]  # extract y-axis data
    elif motion_emb_length == 405:
        joint_speed = data[:, :velocity_emb_length]
        joint_position = data[:, velocity_emb_length:velocity_emb_length+pos_emb_length]

        # Extract velocities and positions of right-foot and left-foot joints
        right_foot_speed = joint_speed[:, 27:33].reshape(-1, 2, 3)  # convert to [151338, 2, 3], two groups per row [vx, vy, vz]
        right_foot_y_position = joint_position[:, 27:33][:, 1::3]  # extract y-axis data

        left_foot_speed = joint_speed[:, 12:18].reshape(-1, 2, 3)  # same for left-foot velocity
        left_foot_y_position = joint_position[:, 12:18][:, 1::3]  # extract y-axis data
    elif motion_emb_length == 570:
        joint_speed = data[:, :velocity_emb_length]
        joint_position = data[:, velocity_emb_length:velocity_emb_length + pos_emb_length]

        # Extract velocities and positions of right-foot and left-foot joints
        right_foot_speed = joint_speed[:, 33:39].reshape(-1, 2, 3)  # convert to [151338, 2, 3], two groups per row [vx, vy, vz]
        right_foot_y_position = joint_position[:, 33:39][:, 1::3]  # extract y-axis data

        left_foot_speed = joint_speed[:, 15:21].reshape(-1, 2, 3)  # same for left-foot velocity
        left_foot_y_position = joint_position[:, 15:21][:, 1::3]  # extract y-axis data
    else:
        print("Use a skeleton that is not defined")
        import sys
        sys.exit(0)

        # Compute velocity norm
    right_foot_speed_norm = np.linalg.norm(right_foot_speed, axis=2).max(axis=1)  # compute the maximum velocity norm
    left_foot_speed_norm = np.linalg.norm(left_foot_speed, axis=2).max(axis=1)  # same for the left foot

    # Compute contact state: velocity norm is near zero and position is below the threshold
    right_foot_contact = (right_foot_speed_norm < speed_threshold) & \
                         (right_foot_y_position < position_threshold).all(axis=1)

    left_foot_contact = (left_foot_speed_norm < speed_threshold) & \
                        (left_foot_y_position < position_threshold).all(axis=1)

    # Merge left-foot and right-foot results
    foot_contact = np.stack((right_foot_contact, left_foot_contact), axis=1).astype(int)

    return foot_contact


def get_with_gather(Data, gather_window, sequence):
    gather = gather_window
    pivot = sequence[0]
    _min = sequence[1]
    _max = sequence[2]

    gather = np.clip(gather + pivot, _min, _max)

    data = Data[gather]
    data = torch.from_numpy(data).float()

    data = data.permute(1, 0)

    return data


def get_with_gather_numpy(Data, gather_window, sequence):
    gather = gather_window
    pivot = sequence[0]
    _min = sequence[1]
    _max = sequence[2]

    gather = np.clip(gather + pivot, _min, _max)

    data = Data[gather].astype(np.float32)
    return data


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.single_frame = False

    def set_single_frame(self, val):
        self.single_frame = val

    def prepare_sequence(self, frames, Load, extra_frames, test_sequence_ratio):
        Shape = get_shape(Load)
        Sequences, Sequences_full = utility.LoadSequences(Load + "/Sequences.txt", False, Shape[0], True)

        feature_dim = Shape[1]
        gather_padding = (int((frames - 1) / 2))
        gather_window = np.arange(frames + extra_frames) - gather_padding
        gather_window_test = np.arange(frames) - gather_padding

        with_predefined_train_set = False
        with_predefined_test_set = False

        ###################################
        # (Diffusion) Load predefined Set
        ###################################
        file_path = Load + "/train_set.txt"
        if osp.exists(file_path) and "Root" in file_path:
            print("Start to use predefined train set")
            with_predefined_train_set = True
            with open(file_path, "r") as file:
                self.train_set_index = [int(line.strip()) for line in file]
            print("Diffusion Train Sequences:", len(self.train_set_index))
        else:
            self.train_set_index = []

        file_path = Load + "/test_set.txt"
        if osp.exists(file_path) and "Root" in file_path:
            with_predefined_test_set = True
            print("Start to use predefined test set")
            with open(file_path, "r") as file:
                self.test_set_index = [int(line.strip()) for line in file]
            print("Diffusion Test Sequences:", len(self.test_set_index))
        else:
            self.test_set_index = []

        ###################################
        # (Diffusion) self-adjust to sample test index for different length motion to avoid repeated start/end frame
        ###################################
        if len(self.test_set_index) > 0 and len(self.train_set_index) > 0:
            min_index = min(self.test_set_index[0], self.train_set_index[0])
            assert min_index <= gather_padding, f"min_index ({min_index}) should be less than gather_padding ({gather_padding})"
            file_path = f"{Load}/test_set_{gather_padding}.txt"
            if min_index == gather_padding:
                pass
            elif osp.exists(file_path) and "Root" in file_path:
                with open(file_path, "r") as file:
                    self.test_set_index = [int(line.strip()) for line in file]
            else:
                print("Start to resample test index due to longer gather_padding!")
                sequence_lens = Sequences[-1]
                index = 0
                resample_test_list = []
                for i in range(sequence_lens):
                    indices = np.where(Sequences == (i + 1))[0]
                    while index < len(self.test_set_index) and indices[0] <= self.test_set_index[index] <= indices[-1]:
                        if (indices[0] + gather_padding) <= self.test_set_index[index] <= (
                                indices[-1] - gather_padding):
                            resample_test_list.append(self.test_set_index[index])
                        index += 1
                with open(f"{Load}/test_set_{gather_padding}.txt", "w") as file:
                    for item in resample_test_list:
                        file.write(f"{item}\n")  # one element per line
                self.test_set_index = resample_test_list

        ###################################
        # Get test_sequences for VQ-VAE
        ###################################
        cache_path = Load + "/cached_sequences.pkl"

        # If the cache file exists, load it directly
        loaded_from_cache = False
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data_sequences, test_sequences = pickle.load(f)
                print("✅ Cached sequences loaded.")
                loaded_from_cache = True
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}. Recreating...")
                os.remove(cache_path)

        if not loaded_from_cache:
            data_sequences = []
            test_sequences = []

            for i in range(Sequences[-1]):
                indices = np.where(Sequences == (i + 1))[0]
                for j in range(indices.shape[0]):
                    slice = [indices[j], indices[0], indices[-1]]
                    data_sequences.append(slice)
                    if np.random.uniform(0, 1) < test_sequence_ratio and (
                            indices[0] + gather_padding) <= indices[j] <= (indices[-1] - gather_padding):
                        test_sequences.append(j)

            # Save the cache after processing
            with open(cache_path, "wb") as f:
                pickle.dump((data_sequences, test_sequences), f)
            print("✅ Sequences processed and cached.")

        ###################################
        # (Diffusion) If not predefined set, divide train and test for diffusion
        ###################################
        if Load.endswith("HumanML3DwithRoot"):
            txt_path = os.path.join(Load, "HumanML3D_test.txt")
            assert os.path.exists(txt_path), f"'HumanML3D_test.txt' not found in {Load}"

            with open(txt_path, "r") as f:
                humanml3d_test_ids = []
                mirror_sequence_shift = int(Sequences[-1] / 2)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("M"):
                        val = int(line[1:]) + mirror_sequence_shift
                    else:
                        val = int(line)
                    humanml3d_test_ids.append(val)
        if "Root" in Load and (not with_predefined_train_set) and (not with_predefined_test_set):
            print("Start to divide train and test set!")
            diffusion_test_ratio = 0.1
            sequence_lens = Sequences[-1]
            for i in range(sequence_lens):
                indices = np.where(Sequences == (i + 1))[0]
                is_test_sequence = False
                if Load.endswith("HumanML3DwithRoot"):
                    if i in humanml3d_test_ids:
                        is_test_sequence = True
                else:
                    if np.random.uniform(0, 1) < diffusion_test_ratio:
                        is_test_sequence = True
                for j in range(indices.shape[0]):
                    motion_startframe = (indices[0] + gather_padding)
                    motion_endframe = (indices[-1] - gather_padding)
                    if is_test_sequence and motion_startframe <= indices[j] <= motion_endframe:
                        self.test_set_index.append(indices[j])
                    elif (not is_test_sequence) and motion_startframe <= indices[j] <= motion_endframe:
                        self.train_set_index.append(indices[j])

            with open(Load + "/test_set.txt", "w") as file:
                for item in self.test_set_index:
                    file.write(f"{item}\n")  # one element per line
            with open(Load + "/train_set.txt", "w") as file:
                for item in self.train_set_index:
                    file.write(f"{item}\n")  # one element per line
            print("Diffusion Train Sequences:", len(self.train_set_index))
            print("Diffusion Test Sequences:", len(self.test_set_index))

        print("Data Sequences:", len(data_sequences))
        print("Test Sequences:", len(test_sequences))

        data_sequences = np.array(data_sequences)

        self.Sequences = Sequences
        self.Sequences_full = Sequences_full
        self.data_sequences = data_sequences
        self.test_sequences = test_sequences
        self.sample_count = len(data_sequences)
        self.gather_window = gather_window
        self.gather_window_test = gather_window_test
        self.window_size = len(gather_window)
        self.window_size_test = len(gather_window_test)
        self.feature_dim = feature_dim

        self.Data = None
        self.data_mean = 0.
        self.data_std = 1.

    def get_window_starting_frame_index(self, item):
        gather = self.gather_window
        sequence = self.data_sequences[item]
        pivot = sequence[0]
        _min = sequence[1]
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)
        return gather[0]

    def __getitem__(self, item):
        gather = self.gather_window
        sequence = self.data_sequences[item]
        pivot = sequence[0]
        _min = sequence[1] + 1
        _max = sequence[2]

        gather = np.clip(gather + pivot, _min, _max)

        data = self.Data[gather]
        data = torch.from_numpy(data).float()

        data = data.permute(1, 0)

        sequence_info = [self.Sequences_full[idx] for idx in gather.tolist()]
        return data, gather, sequence_info

    # def sample_test_sequence(self):
    #     return self[np.random.choice(self.test_sequences)].unsqueeze(0)

    def get_window_bound(self, item):
        sequence = self.data_sequences[item]
        _min = sequence[1]
        _max = sequence[2]
        return _min, _max

    def sample_continuous_test_window(self):
        indices = self.gather_window_test + np.random.choice(self.test_sequences)
        if self.single_frame:
            return torch.from_numpy(self.Data[indices].astype(np.float32))
        return self.load_batches(indices)[..., :self.window_size_test]

    def load_batches(self, indices):
        res = []
        for i in indices:
            res.append(self[i])
        res = torch.stack(res, dim=0)
        return res

    def __len__(self):
        if self.single_frame:
            return self.Data.shape[0]
        return self.sample_count

    def sample_long_sequence(self, length):
        seq = self.Data[:length]
        seq = torch.from_numpy(seq).permute(1, 0)
        return seq


def get_dataset_name_from_path(path: str):
    path = path.strip().lower()
    if 'human' in path:
        if 'loco' in path:
            return 'human_loco'
        return 'human'
    if 'dog' in path:
        return 'dog'
    if 'mocha' in path:
        return path[path.index('mocha'):]
    return 'unknown'


class FeatureCombinedData(BaseDataset):
    def __init__(self, path, window, normalize, test_sequence_ratio, std_cap, extra_frames=0,
                 needed_channel_names=None, pretrained_manifold=False, pretrained_manifold_path=None):
        super().__init__()
        self.precomputed_data = None
        self.foot_contact_label = None
        self.feature_dim = None
        path = check_path(path)
        self.dataset_path = path
        self.fps = get_fps(path)

        if pretrained_manifold:  # Pad length to a multiple of 16 when root data is present
            add_frame = int(window * self.fps) % 16
            frames = int(window * self.fps + 16 - add_frame)
        else:
            frames = int(window * self.fps) + 1
        self.frames_per_window = frames
        self.prepare_sequence(frames, path, extra_frames, test_sequence_ratio)
        self.load_dataset(path, normalize, std_cap,
                          needed_channel_names)
        self.name = get_dataset_name_from_path(path)
        self.mainifold_path = pretrained_manifold_path
        self.manifold = None
        self.stylecode = None

        if self.mainifold_path:
            print(f"Loading ground truth phase and stylecode from '{self.mainifold_path}'...")
            try:
                data = np.load(self.mainifold_path)
                self.manifold = torch.from_numpy(data['manifold']).float()
                self.stylecode = torch.from_numpy(data['stylecode']).float()
                # self.check_phasejump(data)  # Uncomment if needed
            except FileNotFoundError:
                print(f"[Error] File not found: {self.mainifold_path}")
            except KeyError as e:
                print(f"[Error] Missing key in npz file: {e}")
            except Exception as e:
                print(f"[Error] Unexpected issue loading {self.mainifold_path}: {e}")

    def check_phasejump(self, data):
        data_sequence = self.data_sequences[:, 1:]
        unique_data_sequence = np.unique(data_sequence, axis=0)
        phase_index = torch.tensor(data['index'])
        motion_frequency = torch.tensor(data['frequency'])

        change_counts = []
        avg_change_counts = []
        for start, end in unique_data_sequence:
            segment_indices = phase_index[start:end+1]
            frequency = motion_frequency[start:end+1]
            frequency_mean = torch.mean(frequency)
            changes = (segment_indices[1:] != segment_indices[:-1]).sum().item()
            change_counts.append(changes)

            # segment_length = end - start + 1
            avg_changes = changes / frequency_mean
            avg_change_counts.append(avg_changes)

        ori_avg_change_counts = avg_change_counts[::2]
        mirror_avg_change_counts = avg_change_counts[1::2]
        ori_change_counts_mean = torch.mean(torch.tensor(ori_avg_change_counts))
        mirror_change_counts_mean = torch.mean(torch.tensor(mirror_avg_change_counts))

        avg_change_counts_tensor = torch.tensor(avg_change_counts)
        avg_change_counts_mean = torch.mean(avg_change_counts_tensor)

        return change_counts, avg_change_counts, avg_change_counts_mean, ori_change_counts_mean, mirror_change_counts_mean

    def load_dataset(self, path, normalize, std_cap, needed_channel_names=[]):
        data, data_mean, data_std, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize,
                                                                          needed_channel_names, std_cap)

        self.Data = data

        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = channel_dims
        self.channel_names = needed_channel_names

        self.indices = {}

        st = 0
        for i in range(len(needed_channel_names)):
            self.indices[needed_channel_names[i]] = slice(st, st + channel_dims[i])
            st += channel_dims[i]

        self.name_mask = None
        output_path = self.dataset_path + "/precomputed_data.npz"
        foot_contact_path = self.dataset_path + "/foot_contact_results.npz"

        if "Root" in self.dataset_path:
            if osp.exists(output_path):
                print("Load precomputed motion!")
                loaded_data = np.load(output_path)
                self.precomputed_data = torch.from_numpy(loaded_data["precomputed_data"]).float()
            else:
                print("Start to precompute the relative motion!")
                sequence = np.unique(self.data_sequences[:, 1])
                motion_segments = self.split_data_by_sequence(self.Data, sequence)
                all_relative_vals = []

                for segment in motion_segments:
                    relative_val, init_root_position, init_rotation = self.precomputed_motion_dataset(segment.copy())
                    all_relative_vals.append(relative_val)

                self.precomputed_data = np.concatenate(all_relative_vals, axis=0)

                np.savez(output_path, precomputed_data=self.precomputed_data)
                self.precomputed_data = torch.from_numpy(self.precomputed_data).float()

            if osp.exists(foot_contact_path):
                loaded_data = np.load(foot_contact_path)
                self.foot_contact_label = torch.from_numpy(loaded_data["foot_contact"]).float()
    @utility.numpy_wrapper
    def get_feature_by_names(self, all, names):
        res = []
        for name in names:
            res.append(all[..., self.indices[name], :])
        return torch.cat(res, dim=-2)

    def split_data_by_sequence(self, data, sequence):
        sequence_with_end = np.append(sequence, data.shape[0])
        segments = [
            data[sequence_with_end[i]:sequence_with_end[i + 1]]
            for i in range(len(sequence_with_end) - 1)
        ]
        return segments

    def precomputed_motion_dataset(self, motion):
        # initiate
        root_position = motion[:, -12:-9]
        initial_position = root_position[0].copy()
        rotation_matrices = motion[:, -9:].reshape(-1, 3, 3)
        initial_rotation = rotation_matrices[0].copy()

        # relative position calculation
        root_position -= root_position[0].copy()

        # relative velocity calculation
        relative_velocity = root_position[1:] - root_position[:-1]
        relative_velocity_local = []

        for t in range(relative_velocity.shape[0]):
            R_t_minus_1 = rotation_matrices[t]
            velocity_local = R_t_minus_1.T @ relative_velocity[t]
            relative_velocity_local.append(velocity_local)

        if len(relative_velocity_local) > 0:
            relative_velocity_local = np.stack(relative_velocity_local)
            relative_velocity_local = np.vstack([np.zeros((1, 3)), relative_velocity_local])
        else:
            relative_velocity_local = np.zeros((1, 3))

        motion[:, -12:-9] = relative_velocity_local

        # relative rotation calculation
        relative_rotations = []
        for t in range(rotation_matrices.shape[0] - 1):
            R_t_minus_1 = rotation_matrices[t]  # previous-frame rotation matrix
            R_t = rotation_matrices[t + 1]  # current-frame rotation matrix
            R_relative = R_t_minus_1.T @ R_t  # compute relative rotation
            relative_rotations.append(R_relative)

        identity = np.eye(3)[np.newaxis, :]
        if len(relative_rotations) > 0:
            relative_rotations = np.vstack([identity, np.stack(relative_rotations)])
        else:
            relative_rotations = identity

        motion[:, -9:] = relative_rotations.reshape(-1, 9)

        return motion, initial_position, initial_rotation
    def get_feature_dim_by_names(self, names):
        res = []
        for name in names:
            res.append(self.feature_dims[self.channel_names.index(name)])
        return res

    def get_n_channel_by_names(self, names):
        res = 0
        for name in names:
            idx = self.channel_names.index(name)
            res += self.feature_dims[idx]
        return res

    def simplify_sequence_info(self, sequence_info):
        ans = sequence_info[0]
        ans.append(sequence_info[-1][1])
        return ans

    def regularize_y_rotation(self, matrix):
        cos_theta = matrix[0, 0]
        sin_theta = matrix[0, 2]
        norm = torch.sqrt(cos_theta ** 2 + sin_theta ** 2)
        cos_theta /= norm
        sin_theta /= norm

        # Build the normalized rotation matrix
        return torch.tensor([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ], dtype=matrix.dtype, device=matrix.device)

    def transfer2absolute_velo_angular(self, motion, init_root_position=None, init_rotation=None):
        device = motion.device
        if init_root_position is None:
            init_root_position = torch.zeros(3, device=device)
        if init_rotation is None:
            init_rotation = torch.eye(3, device=device)
        assert isinstance(motion, torch.Tensor), "motion must be a torch.Tensor"
        assert isinstance(init_root_position, torch.Tensor), "init_root_position must be a torch.Tensor"
        assert isinstance(init_rotation, torch.Tensor), "init_rotation must be a torch.Tensor"

        # Extract relative velocity
        relative_velocity_local = motion[:, -12:-9]
        # Extract relative rotation matrix
        relative_rotations = motion[:, -9:].reshape(-1, 3, 3)
        cos_theta = relative_rotations[:, 0, 0]
        sin_theta = relative_rotations[:, 0, 2]
        angles_rad = torch.atan2(sin_theta, cos_theta)
        # init first angle using init_rotation
        cos_theta = init_rotation[0,0]
        sin_theta = init_rotation[0,2]
        first_angle_rad = torch.atan2(sin_theta, cos_theta)
        angles_rad[0] = first_angle_rad
        # accumulate to get the global rotation
        absolute_angles_rad = torch.cumsum(angles_rad, dim=0)

        def rotation_matrix_y(angle):
            """Generate a rotation matrix around the Y axis from an angle"""
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            return torch.tensor([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])

        # Generate absolute rotation matrices
        absolute_rotations_test = torch.stack([rotation_matrix_y(angle) for angle in absolute_angles_rad])
        absolute_rotations_test = absolute_rotations_test.to(device)
        # Initialize absolute position and rotation-matrix lists
        absolute_positions = []

        # Initialize the first position and rotation
        current_position = init_root_position
        # Iterate over frames to compute absolute position and rotation
        for t in range(motion.shape[0]):
            # Update absolute position
            if t > 0:
                velocity_global = absolute_rotations_test[t-1] @ relative_velocity_local[t]
                current_position = current_position + velocity_global
            absolute_positions.append(current_position)

        # Write absolute positions and rotation matrices back to motion
        absolute_positions = torch.stack(absolute_positions)
        motion[:, -12:-9] = absolute_positions

        motion[:, -9:] = absolute_rotations_test.reshape(absolute_rotations_test.shape[0], -1)
        return motion

    def transfer2relative_velo_angular(self, motion):
        # initiate
        root_position = motion[:, -12:-9]
        initial_position = root_position[0].clone()
        rotation_matrices = motion[:, -9:].reshape(-1, 3, 3)
        initial_rotation = rotation_matrices[0].clone()

        # relative position calculation
        root_position -= root_position[0].clone()
        relative_position = root_position.clone()
        relative2start_position = (initial_rotation.T @ relative_position.T).T

        # relative velocity calculation
        relative_velocity = root_position[1:] - root_position[:-1]
        relative_velocity_local = []

        for t in range(relative_velocity.shape[0]):
            R_t_minus_1 = rotation_matrices[t]
            velocity_local = R_t_minus_1.T @ relative_velocity[t]
            relative_velocity_local.append(velocity_local)

        relative_velocity_local = torch.stack(relative_velocity_local)
        relative_velocity_local = torch.cat([torch.zeros(1, 3), relative_velocity_local], dim=0)
        # relative_velocity_local = torch.cat([relative_velocity[:1], relative_velocity_local], dim=0)
        motion[:, -12:-9] = relative_velocity_local

        relative_rotations = []
        for t in range(rotation_matrices.shape[0] - 1):
            R_t_minus_1 = rotation_matrices[t]  # previous-frame rotation matrix
            R_t = rotation_matrices[t + 1]  # current-frame rotation matrix
            R_relative = R_t_minus_1.T @ R_t  # compute relative rotation
            relative_rotations.append(R_relative)

        # relative_rotations.insert(0, initial_rotation)
        # relative_rotations = torch.stack(relative_rotations)
        identity = torch.eye(3).unsqueeze(0)
        relative_rotations = torch.cat([identity, torch.stack(relative_rotations)], dim=0)

        motion[:, -9:] = relative_rotations.reshape(-1, 9)
        # motion_root = torch.concatenate((motion, relative2start_position), axis=1)
        return motion, initial_position, initial_rotation, relative2start_position

    def transfer2absolute_batch(self, motion, init_root_position=None, init_rotation=None):
        device = motion.device
        batch_size, seq_len, _ = motion.shape

        if init_root_position is None:
            init_root_position = torch.zeros(batch_size, 3, device=device)
        if init_rotation is None:
            init_rotation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        assert isinstance(motion, torch.Tensor), "motion must be a torch.Tensor"
        assert isinstance(init_root_position, torch.Tensor), "init_root_position must be a torch.Tensor"
        assert isinstance(init_rotation, torch.Tensor), "init_rotation must be a torch.Tensor"

        # Extract relative velocity and relative rotation matrix
        relative_velocity_local = motion[:, :, -12:-9]  # (batch_size, seq_len, 3)
        relative_rotations = motion[:, :, -9:].reshape(batch_size, seq_len, 3, 3)  # (batch_size, seq_len, 3, 3)

        # Compute the per-frame angle change
        cos_theta = relative_rotations[:, :, 0, 0]
        sin_theta = relative_rotations[:, :, 0, 2]
        angles_rad = torch.atan2(sin_theta, cos_theta)  # (batch_size, seq_len)

        # Initialize the first-frame angle
        init_cos_theta = init_rotation[:, 0, 0]  # (batch_size,)
        init_sin_theta = init_rotation[:, 0, 2]  # (batch_size,)
        first_angle_rad = torch.atan2(init_sin_theta, init_cos_theta)  # (batch_size,)
        angles_rad[:, 0] = first_angle_rad

        # Accumulate angle changes to obtain global angles
        absolute_angles_rad = torch.cumsum(angles_rad, dim=1)  # (batch_size, seq_len)

        def rotation_matrix_y(angle):
            """Generate a rotation matrix around the Y axis from an angle"""
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            zeros = torch.zeros_like(cos_theta)
            ones = torch.ones_like(cos_theta)
            return torch.stack([
                torch.stack([cos_theta, zeros, sin_theta], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sin_theta, zeros, cos_theta], dim=-1)
            ], dim=-2)

        # Generate absolute rotation matrices
        absolute_rotations = rotation_matrix_y(absolute_angles_rad)  # (batch_size, seq_len, 3, 3)

        # Initialize the absolute-position list
        absolute_positions = torch.zeros(batch_size, seq_len, 3, device=device)
        absolute_positions[:, 0, :] = init_root_position
        # Iterate over frames to compute absolute positions
        for t in range(1, seq_len):
            velocity_global = torch.bmm(absolute_rotations[:, t - 1],
                                        relative_velocity_local[:, t].unsqueeze(-1)).squeeze(-1)
            absolute_positions[:, t] = absolute_positions[:, t - 1] + velocity_global

        # Write absolute positions and rotation matrices back to motion
        updated_motion = motion.clone()
        updated_motion[:, :, -12:-9] = absolute_positions
        updated_motion[:, :, -9:] = absolute_rotations.reshape(batch_size, seq_len, -1)
        return updated_motion

    def rotate_motion_tensor_torch(self, motion_tensor, angle_deg):
        """
        Rotate a 64x432 motion tensor in place (PyTorch version).

        Parameters:
        - motion_tensor: torch.Tensor, shape is (64, 432), where the last 12 dimensions are root position (3D) and rotation matrix (9D).
        - angle_deg: rotation angle in degrees.

        Returns:
        - rotated motion tensor with the same shape as the input.
        """
        # Convert angle to radians
        angle_rad = angle_deg * (torch.pi / 180)  # compute radians first
        angle_rad = torch.tensor(angle_rad, dtype=motion_tensor.dtype, device=motion_tensor.device)  # convert to torch.Tensor

        # Build rotation matrix around the Y axis
        rotation_matrix = torch.tensor([
            [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
            [0, 1, 0],
            [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
        ], dtype=motion_tensor.dtype, device=motion_tensor.device)  # 3x3matrix

        # Extract initial position and rotation matrix
        root_positions = motion_tensor[:, -12:-9]  # extract root positions (3D)
        root_rotations = motion_tensor[:, -9:].view(-1, 3, 3)  # extract rotation matrix (9D -> 3x3)

        # Translate to the origin
        initial_position = root_positions[0]  # initial position
        translated_positions = root_positions - initial_position  # Translate to the origin

        # Apply rotation
        rotated_positions = torch.matmul(translated_positions, rotation_matrix.T)  # rotated positions
        rotated_rotations = torch.matmul(root_rotations, rotation_matrix)  # compose rotations

        # Translate back
        final_positions = rotated_positions + initial_position

        # Update tensor
        motion_tensor[:, -12:-9] = final_positions  # update positions
        motion_tensor[:, -9:] = rotated_rotations.view(-1, 9)  # update rotation matrix

        return motion_tensor
    def check_motion_transfer(self, absolute_val, test_absolute_val):
        # Compute absolute error
        position_diff = torch.norm(absolute_val[:, -12:-9] - test_absolute_val[:, -12:-9], dim=-1)
        rotation_diff = torch.norm(absolute_val[:, -9:] - test_absolute_val[:, -9:], dim=-1)

        # Print error
        print("Position difference (L2 norm):", position_diff)
        print("Rotation difference (L2 norm):", rotation_diff)

        # Set tolerance range
        position_tolerance = 1e-6
        rotation_tolerance = 1e-6

        # Verify consistency
        if torch.all(position_diff < position_tolerance) and torch.all(rotation_diff < rotation_tolerance):
            print("Test passed: absolute_val and test_absolute_val are consistent!")
        else:
            print("Test failed: There are discrepancies between absolute_val and test_absolute_val.")

    def check_cos_sin(self, motion):
        values_423 = motion[:, 423]
        values_425 = motion[:, 425]

        # Compute sum of squares
        squared_sum = values_423 ** 2 + values_425 ** 2

        # Check whether it equals 1, allowing floating-point error
        is_equal_to_one = np.isclose(squared_sum, 1.0)

        # Print result
        print("Does the sum of squares equal 1: ", is_equal_to_one)

    def check_relative2start_motion(self, relative2start_position, init_rotation, motion):
        root_position = motion[:, -12:-9]
        root_position -= root_position[0].clone()

        global_position = []
        for t in range(relative2start_position.shape[0]):
            global_position.append(init_rotation @ relative2start_position[t])
        global_position = torch.stack(global_position)
        are_close = np.allclose(root_position, global_position, atol=1e-6)
        print("The computation of relative position of the start frame is:", are_close)
    def __getitem__(self, item):
        # val [D, T]
        val, gather, sequence_info = super().__getitem__(item)

        if self.name_mask is not None:
            val = self.get_feature_by_names(val, self.name_mask)
        if self.mainifold_path is not None:
            # absolute_val [T, D]
            absolute_val = val.T
            init_root_position = absolute_val[0, -12:-9].clone()
            init_root_rotation = absolute_val[0, -9:].clone().view(3, 3)
            idx = gather
            relative_val = self.precomputed_data[idx].T
            manifold = self.manifold[idx].T
            stylecode = self.stylecode[idx].T
            foot_contact = self.foot_contact_label[idx]
            global_root = val[-12:, :]
            # debug
            # frame_num = 48
            # write_motion2npz(absolute_val.numpy(), self.data_std, self.data_mean, frame_num,
            #                  f'datasetcheck_gt{10}_motion.npz', True, None)
            return relative_val, manifold, init_root_position, init_root_rotation, foot_contact, global_root, stylecode
        return val


def set_std_cap(data_std, cap):
    print(f"Set {(data_std < cap).sum()} entries cap to", cap)
    print("The entries are", np.where(data_std < cap)[0])
    data_std[data_std < cap] = cap


class SequenceAndManifold(BaseDataset):
    def __init__(self, path, window, test_sequence_ratio, path4manifold, needed_channel_names, normalize, use_manifold_ori,
                 std_cap, extra_frames=0, frames=None, needed_manifold_names=['manifold', ], normalize_manifold=True,
                 requires_full_sequence=False, additional_manifold_names=[]):
        super().__init__()
        path = check_path(path)

        data, _, _, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize=False,
                                                                                 needed_feature_names=needed_channel_names,
                                                                                 std_cap=0)

        manifold = np.load(path4manifold)
        manifold_features = []
        manifold_dims = []

        additional_manifold_features = []
        additional_manifold_dims = []
        if path4manifold.endswith('.npz'):
            for name in needed_manifold_names:
                manifold_features.append(manifold[name])
                manifold_dims.append(manifold[name].shape[-1])

            for name in additional_manifold_names:
                additional_manifold_features.append(manifold[name])
                additional_manifold_dims.append(manifold[name].shape[-1])

        manifold = np.concatenate(manifold_features, axis=-1)
        data = np.concatenate((manifold, data), axis=-1)
        self.n_channel_manifold = manifold.shape[-1]

        self.additional_manifold_names = additional_manifold_names
        self.additional_manifold_features = additional_manifold_features
        self.additional_manifold_dims = additional_manifold_dims

        data_std = data.std(axis=0)
        data_mean = data.mean(axis=0)
        if not normalize:
            data_std[:] = 1.0
            data_mean[:] = 0.0
        if not normalize_manifold:
            manifold_dim = manifold.shape[-1]
            data_std[:manifold_dim] = 1.0
            data_mean[:manifold_dim] = 0.0

        set_std_cap(data_std, std_cap)
        data = (data - data_mean) / data_std

        self.fps = get_fps(path)
        if frames is None:
            frames = int(window * self.fps) + 1
        self.frames_per_window = frames
        self.prepare_sequence(frames, path, extra_frames, test_sequence_ratio)

        if requires_full_sequence:
            self.full_sequence = utility.LoadFullSequence(path + "/Sequences.txt", True, data.shape[0])
            self.restore_full_sequence_mapping()
        self.name = get_dataset_name_from_path(path)
        if use_manifold_ori:
            self.name += '_ori'
        self.Data = data
        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = manifold_dims + channel_dims
        self.channel_names = needed_manifold_names + needed_channel_names
        self.fps = get_fps(path)

    def get_manifold_feature(self, name):
        feature_idx = self.additional_manifold_names.index(name)
        return self.additional_manifold_features[feature_idx]

    def get_motion_feature(self, name):
        feature_idx = self.channel_names.index(name)
        all_features = self.Data
        for i in range(feature_idx):
            all_features = all_features[..., self.feature_dims[i]:]
        return all_features[..., :self.feature_dims[feature_idx]]

    def get_motion_window(self, name, item):
        data = self.get_motion_feature(name)
        return get_with_gather_numpy(data, self.gather_window, self.data_sequences[item])

    def get_one_cycle(self):
        if self.name.startswith('dog') or self.name.startswith('human'):
            return 1
        elif self.name.startswith('mocha'):
            return 2
        else:
            raise Exception("Unknown dataset")

    def get_num_states(self):
        return self.get_manifold_feature('index').max() + 1

    def get_manifold_window(self, item, extra_frames, name, extra_frames_rear_only=False):
        feature_idx = self.additional_manifold_names.index(name)
        data = self.additional_manifold_features[feature_idx]
        gather = self.gather_window
        if extra_frames > 0:
            gather0 = np.arange(-extra_frames, 0, dtype=np.int64) + gather[0] if not extra_frames_rear_only else np.zeros((0,), dtype=np.int64)
            gather1 = np.arange(0, extra_frames, dtype=np.int64) + gather[-1] + 1
            gather = np.concatenate([gather0, gather, gather1])
        return get_with_gather_numpy(data, gather, self.data_sequences[item])

    def get_phase(self, item, extra_frames=0):
        return self.get_manifold_window(item, extra_frames, 'phase')

    def restore_full_sequence_mapping(self):
        """
        This function exists because the id for motion is modified in order to remove breaking frames
        by cutting the motion into multiple sequences.
        """
        self.full_sequence_mapping = {}
        current_count = 1
        for i in range(1, max(self.Sequences) + 1):
            self.full_sequence_mapping[i] = current_count
            indices = np.where(self.Sequences == i)[0]
            start = indices[0]
            if start == 0 or \
                (self.full_sequence[start][-1] != self.full_sequence[start-1][-1] or  \
                        self.full_sequence[start][2] != self.full_sequence[start-1][2]):
                current_count += 1
            else:
                # print('Something is wrong')
                pass


class Text2PhaseData(BaseDataset):
    def __init__(self, path, normalize, std_cap, num_embed_vq,
                 needed_channel_names=None, pretrained_manifold_path=None, mode='train', max_text_len=20, clip_version='ViT-B/32'):
        super().__init__()
        self.precomputed_data = None
        self.foot_contact_label = None
        self.feature_dim = None
        self.dataset_path = check_path(path)
        self.fps = get_fps(path)
        self.textpath = path + '/texts/'
        self.max_text_len = max_text_len
        self.num_embed_vq = num_embed_vq
        self.clip_version = clip_version
        self.clip_path = path + '/clip/'
        self.clip_model = self.load_and_freeze_clip(clip_version, self.clip_path)
        self.encode_text = self.clip_encode_text  # bind method

        # load motion to check results
        self.load_dataset(path, normalize, std_cap,
                          needed_channel_names)
        self.mode = mode
        self.name = get_dataset_name_from_path(path)
        self.manifold_path = pretrained_manifold_path
        data = np.load(self.manifold_path)
        phase = torch.tensor(data['phase'])  # phase [-0.5, 0.5]
        # self.state = torch.tensor(data['state'])
        # manifold, angle_xy = self.get_phase_manifold(self.state, self.angle_rad)
        # self.manifold = torch.tensor(data['manifold'])
        angle_rad = torch.tensor(2 * np.pi, dtype=torch.float32) * (phase.unsqueeze(-1))
        self.angle_xy = self.get_angle_xy(angle_rad)
        self.stylecode = torch.tensor(data['stylecode'])
        self.manifold_index = torch.tensor(data['index'])  # we map it to one-hot vector in collate_fn()
        self.manifold_continuous = torch.tensor(data['manifold']).float()  # [N_total, n_latent_channel]

        # start to load Human3DML
        print("Start to load train/text txt")
        self.text2phase = None
        self._load_split_txt()
        self._load_csv()

        # process text infomation
        self.data_dict = {}
        self.name_list = []
        self.length_list = []
        self.cache_path = os.path.join(self.dataset_path, "preprocessed_data", f"{mode}_cache.pt")

        if self.try_load_cache():
            print("Loaded from cache")
        else:
            print(f"Building new {mode} dataset")
            data_list = self.train_txt_list if mode == 'train' else self.test_txt_list
            self.build_data(data_list)
            self.save_cache()
        print("Finish load Human3DML txt and correspond phase")

    def _load_split_txt(self):
        def read_txt(path):
            with open(path, "r") as f:
                return [line.strip() for line in f if line.strip()]

        self.all_txt_list = read_txt(os.path.join(self.dataset_path, "all.txt"))
        self.train_txt_list = read_txt(os.path.join(self.dataset_path, "train.txt"))
        self.test_txt_list = read_txt(os.path.join(self.dataset_path, "test.txt"))

    def _load_csv(self):
        csv_path = os.path.join(self.dataset_path, "text2phase.csv")
        df = pd.read_csv(csv_path).set_index("key")
        self.text2phase = df.to_dict("index")

    def try_load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                cached = torch.load(self.cache_path, map_location='cpu', weights_only=False)
                self.data_dict = cached['data_dict']
                self.name_list = cached['name_list']
                self.text2phase = cached['text2phase']
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
        return False

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        torch.save({
            'data_dict': self.data_dict,
            'name_list': self.name_list,
            'text2phase': self.text2phase
        }, self.cache_path)

    def build_data(self, data_list):
        for name in tqdm(data_list, desc="Loading motion data"):
            if name.startswith(("m", "M")):
                clean_name = name[1:]
                self._process_one_motion(clean_name, mirrored=True)
            else:
                self._process_one_motion(name, mirrored=False)

    def _process_one_motion(self, name, mirrored=False):
        min_motion_len, max_motion_len = 40, 200
        file_info = self.text2phase[int(name)]
        keyname = name if not mirrored else 'M' + name

        suffix = '' if not mirrored else '_m'
        angle_xy = self.angle_xy[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}']]
        stylecode = self.stylecode[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}']]
        manifold_index = self.manifold_index[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}']]
        manifold_cont = self.manifold_continuous[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}']]
        motion = self.Data[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}']]
        rela_traj = self.precomputed_data[file_info[f'global_start{suffix}']:file_info[f'global_end{suffix}'], -12:]
        try:
            if (len(motion)) <= min_motion_len or (len(motion) > 200):
                return
            text_data = []
            has_full_motion = False
            with cs.open(pjoin(self.textpath, keyname + '.txt'), encoding='utf-8') as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    with torch.no_grad():
                        text_dict['text_embed'] = self.clip_encode_text([caption]).squeeze(0).cpu()
                    if f_tag == 0.0 and to_tag == 0.0:
                        has_full_motion = True
                        text_data.append(text_dict)
                    else:
                        try:
                            # this case does happen because of the wrong annotation
                            # if int(to_tag * 20) > angle_xy.shape[0]:
                            #     print("Warning text contains more motion than required by index.csv")
                            n_angle_xy = angle_xy[int(f_tag * 20): int(to_tag * 20)]
                            n_stylecode = stylecode[int(f_tag * 20): int(to_tag * 20)]
                            n_manifold_index = manifold_index[int(f_tag * 20): int(to_tag * 20)]
                            n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
                            n_rela_traj = rela_traj[int(f_tag * 20): int(to_tag * 20)]
                            if (len(n_angle_xy)) < min_motion_len or (len(n_angle_xy) > max_motion_len):
                                # print("Phase too long/short")
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + keyname
                            while new_name in self.data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + keyname
                            n_manifold_cont = manifold_cont[int(f_tag * 20): int(to_tag * 20)]
                            self.data_dict[new_name] = {'angle_xy': n_angle_xy,
                                                        'stylecode': n_stylecode,
                                                        'manifold_index': n_manifold_index,
                                                        'manifold_continuous': n_manifold_cont,
                                                        'text': [text_dict],
                                                        'length': len(n_angle_xy),
                                                        'motion': n_motion,
                                                        'relative_trajectory':n_rela_traj}
                            self.name_list.append(new_name)
                            self.length_list.append(len(n_angle_xy))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag, to_tag, keyname)
            if has_full_motion:
                self.data_dict[keyname] = {'angle_xy': angle_xy,
                                        'stylecode': stylecode,
                                        'manifold_index': manifold_index,
                                        'manifold_continuous': manifold_cont,
                                        'text': text_data,
                                        'length': len(angle_xy),
                                        'motion': motion,
                                        'relative_trajectory':rela_traj}
                self.name_list.append(keyname)
                self.length_list.append(len(angle_xy))
        except Exception as e:
            print(f"Error processing {keyname}: {str(e)}")
            pass
    def load_and_freeze_clip(self, clip_version, clip_path):
        clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                                jit=False, download_root=clip_path)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = 'cuda'
        if isinstance(raw_text, list):
            processed_text = [' '.join(t) if isinstance(t, list) else t for t in raw_text]
        else:
            processed_text = raw_text
        max_text_len = self.max_text_len
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(processed_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(processed_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()
    def get_angle_xy(self, angles_rad):
        angles_rad = angles_rad.squeeze(dim=2)
        y0 = torch.cos(angles_rad)
        y1 = torch.sin(angles_rad)
        y = torch.cat([y0, y1], dim=1)
        return y
    def get_phase_manifold(self, state, angles):
        # This get_phase_manifold function generates a phase manifold and signal from the input state (the ellipse basis A) and phase angles.
        """
        :param state: (batch_size, n_channel_latent)
        :param angles: (batch_size, n_channel_phase, time_range)
        :return:
        """
        state = state.reshape((state.shape[0], angles.shape[1], -1, 2))
        y0 = torch.cos(angles)
        y1 = torch.sin(angles)
        y = torch.stack((y0, y1), dim=-2)
        signal = y
        y = state @ y
        y = y.reshape(y.shape[0], -1, y.shape[-1])
        # y: manifold computed from state and phase angles, with shape (batch_size, n_combined_features, time_range), representing combined state and phase information.
        # signal: the generated phase signal, with shape related toanglesand containing sine/cosine phase representations
        return y, signal
    def load_dataset(self, path, normalize, std_cap, needed_channel_names=[]):
        data, data_mean, data_std, channel_dims, needed_channel_names = load_single_dataset_bin(path, normalize,
                                                                                                needed_channel_names,
                                                                                                std_cap)

        self.Data = data

        self.data_mean = data_mean
        self.data_std = data_std
        self.feature_dims = channel_dims
        self.channel_names = needed_channel_names

        self.indices = {}

        st = 0
        for i in range(len(needed_channel_names)):
            self.indices[needed_channel_names[i]] = slice(st, st + channel_dims[i])
            st += channel_dims[i]

        # self.name_mask = None
        output_path = self.dataset_path.removesuffix("_Text") + "/precomputed_data.npz"
        # foot_contact_path = self.dataset_path + "/foot_contact_results.npz"
        #
        if "Root" in self.dataset_path:
            if osp.exists(output_path):
                loaded_data = np.load(output_path)
                self.precomputed_data = torch.from_numpy(loaded_data["precomputed_data"]).float()
        #     else:
        #         print("Start to precompute the relative motion!")
        #         sequence = np.unique(self.data_sequences[:, 1])
        #         motion_segments = self.split_data_by_sequence(self.Data, sequence)
        #         all_relative_vals = []
        #
        #         for segment in motion_segments:
        #             relative_val, init_root_position, init_rotation = self.precomputed_motion_dataset(segment.copy())
        #             all_relative_vals.append(relative_val)
        #
        #         self.precomputed_data = np.concatenate(all_relative_vals, axis=0)
        #
        #         np.savez(output_path, precomputed_data=self.precomputed_data)
        #         self.precomputed_data = torch.from_numpy(self.precomputed_data).float()
        #
        #     if osp.exists(foot_contact_path):
        #         loaded_data = np.load(foot_contact_path)
        #         self.foot_contact_label = torch.from_numpy(loaded_data["foot_contact"]).float()

    @utility.numpy_wrapper
    def get_feature_by_names(self, all, names):
        res = []
        for name in names:
            res.append(all[..., self.indices[name], :])
        return torch.cat(res, dim=-2)

    def split_data_by_sequence(self, data, sequence):
        sequence_with_end = np.append(sequence, data.shape[0])
        segments = [
            data[sequence_with_end[i]:sequence_with_end[i + 1]]
            for i in range(len(sequence_with_end) - 1)
        ]
        return segments

    def precomputed_motion_dataset(self, motion):
        # initiate
        root_position = motion[:, -12:-9]
        initial_position = root_position[0].copy()
        rotation_matrices = motion[:, -9:].reshape(-1, 3, 3)
        initial_rotation = rotation_matrices[0].copy()

        # relative position calculation
        root_position -= root_position[0].copy()

        # relative velocity calculation
        relative_velocity = root_position[1:] - root_position[:-1]
        relative_velocity_local = []

        for t in range(relative_velocity.shape[0]):
            R_t_minus_1 = rotation_matrices[t]
            velocity_local = R_t_minus_1.T @ relative_velocity[t]
            relative_velocity_local.append(velocity_local)

        relative_velocity_local = np.stack(relative_velocity_local)
        relative_velocity_local = np.vstack([np.zeros((1, 3)), relative_velocity_local])

        motion[:, -12:-9] = relative_velocity_local

        # relative rotation calculation
        relative_rotations = []
        for t in range(rotation_matrices.shape[0] - 1):
            R_t_minus_1 = rotation_matrices[t]  # previous-frame rotation matrix
            R_t = rotation_matrices[t + 1]  # current-frame rotation matrix
            R_relative = R_t_minus_1.T @ R_t  # compute relative rotation
            relative_rotations.append(R_relative)

        identity = np.eye(3)[np.newaxis, :]
        relative_rotations = np.vstack([identity, np.stack(relative_rotations)])

        motion[:, -9:] = relative_rotations.reshape(-1, 9)

        return motion, initial_position, initial_rotation

    def get_feature_dim_by_names(self, names):
        res = []
        for name in names:
            res.append(self.feature_dims[self.channel_names.index(name)])
        return res

    def get_n_channel_by_names(self, names):
        res = 0
        for name in names:
            idx = self.channel_names.index(name)
            res += self.feature_dims[idx]
        return res

    def simplify_sequence_info(self, sequence_info):
        ans = sequence_info[0]
        ans.append(sequence_info[-1][1])
        return ans

    def transfer2absolute_batch(self, motion, init_root_position=None, init_rotation=None):
        device = motion.device
        batch_size, seq_len, _ = motion.shape

        if init_root_position is None:
            init_root_position = torch.zeros(batch_size, 3, device=device)
        if init_rotation is None:
            init_rotation = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        assert isinstance(motion, torch.Tensor), "motion must be a torch.Tensor"
        assert isinstance(init_root_position, torch.Tensor), "init_root_position must be a torch.Tensor"
        assert isinstance(init_rotation, torch.Tensor), "init_rotation must be a torch.Tensor"

        # Extract relative velocity and relative rotation matrix
        relative_velocity_local = motion[:, :, -12:-9]  # (batch_size, seq_len, 3)
        relative_rotations = motion[:, :, -9:].reshape(batch_size, seq_len, 3, 3)  # (batch_size, seq_len, 3, 3)

        # Compute the per-frame angle change
        cos_theta = relative_rotations[:, :, 0, 0]
        sin_theta = relative_rotations[:, :, 0, 2]
        angles_rad = torch.atan2(sin_theta, cos_theta)  # (batch_size, seq_len)

        # Initialize the first-frame angle
        init_cos_theta = init_rotation[:, 0, 0]  # (batch_size,)
        init_sin_theta = init_rotation[:, 0, 2]  # (batch_size,)
        first_angle_rad = torch.atan2(init_sin_theta, init_cos_theta)  # (batch_size,)
        angles_rad[:, 0] = first_angle_rad

        # Accumulate angle changes to obtain global angles
        absolute_angles_rad = torch.cumsum(angles_rad, dim=1)  # (batch_size, seq_len)

        def rotation_matrix_y(angle):
            """Generate a rotation matrix around the Y axis from an angle"""
            cos_theta = torch.cos(angle)
            sin_theta = torch.sin(angle)
            zeros = torch.zeros_like(cos_theta)
            ones = torch.ones_like(cos_theta)
            return torch.stack([
                torch.stack([cos_theta, zeros, sin_theta], dim=-1),
                torch.stack([zeros, ones, zeros], dim=-1),
                torch.stack([-sin_theta, zeros, cos_theta], dim=-1)
            ], dim=-2)

        # Generate absolute rotation matrices
        absolute_rotations = rotation_matrix_y(absolute_angles_rad)  # (batch_size, seq_len, 3, 3)

        # Initialize the absolute-position list
        absolute_positions = torch.zeros(batch_size, seq_len, 3, device=device)
        absolute_positions[:, 0, :] = init_root_position
        # Iterate over frames to compute absolute positions
        for t in range(1, seq_len):
            velocity_global = torch.bmm(absolute_rotations[:, t - 1],
                                        relative_velocity_local[:, t].unsqueeze(-1)).squeeze(-1)
            absolute_positions[:, t] = absolute_positions[:, t - 1] + velocity_global

        # Write absolute positions and rotation matrices back to motion
        updated_motion = motion.clone()
        updated_motion[:, :, -12:-9] = absolute_positions
        updated_motion[:, :, -9:] = absolute_rotations.reshape(batch_size, seq_len, -1)
        return updated_motion

    def __len__(self):
        return len(self.data_dict)


    def __getitem__(self, item):
        key = self.name_list[item]
        data = self.data_dict[key]

        text_data = random.choice(data['text'])
        text_embed = text_data['text_embed']

        # debug
        # frame_num = data['length']
        # write_motion2npz(motion, self.data_std, self.data_mean, frame_num,
        #                  f'datasetcheck_gt{10}_motion.npz', True, None)
        return {
            'angle_xy': data['angle_xy'],
            'stylecode': data['stylecode'],
            'manifold_index': data['manifold_index'],
            'manifold_continuous': data.get('manifold_continuous', None),
            'm_length': data['length'],
            'text_embed': text_embed,
            'text': text_data['caption'],
            'motion': data['motion'],
            'relative_trajectory': data['relative_trajectory']
        }

def repeat_to_target(manifold_list, lengths, target_length):
    """
    manifold_list: list of tensors, each has shape (L_i, D)
    lengths: (B,) valid length of each sample
    target_length: int, common length (208)

    return: (B, target_length, D)
    """
    if manifold_list[0][-1, 0] != 0:
        return torch.stack(manifold_list)
    padded_list = []
    for mani, L in zip(manifold_list, lengths):
        valid = mani[:L]  # (L, D)
        repeats = (target_length + L - 1) // L  # minimum number of repeats needed
        tiled = valid.repeat((repeats, 1))  # (repeats*L, D)
        padded = tiled[:target_length]  # truncate to the target length
        padded_list.append(padded)
    return torch.stack(padded_list)  # (B, target_length, D)

def text2motion_collate_fn(batch, num_embed_vq, target_length=208):
    # Unpack each item in the batch
    angle_xy_list = [item['angle_xy'] for item in batch]
    stylecode_list = [item['stylecode'] for item in batch]
    manifold_index_list = [item['manifold_index'] for item in batch]
    manifold_list = [item['manifold'] for item in batch]
    text_list = [item['text'] for item in batch]
    text_embed_list = [item['text_embed'] for item in batch]
    lengths = [item['m_length'] for item in batch]
    motion = [item['motion'] for item in batch]
    relative_trajectory = [item['relative_trajectory'] for item in batch]
    gt_relative_trajectory = [item['gt_relative_trajectory'] for item in batch]

    # Ensure no sequence exceeds the target length
    assert all(l <= target_length for l in lengths), "Some sequences exceed the target length."

    # Pad angle_xy to target_length with zeros
    padded_angle_xy = torch.stack([
        F.pad(xy, (0, 0, 0, target_length - xy.size(0)))  # pad rows
        for xy in angle_xy_list
    ])  # Shape: (B, target_length, 2)

    # Pad stylecode to target_length
    padded_stylecode = repeat_to_target(stylecode_list, lengths, target_length)

    # Pad manifold index with -1
    padded_manifold_idx = torch.stack([
        F.pad(mi, (0, 0, 0, target_length - mi.size(0)), value=-1)
        for mi in manifold_index_list
    ])  # Shape: (B, target_length, 1) or (B, target_length)


    # Pad manifold to target_length
    padded_manifold = repeat_to_target(manifold_list, lengths, target_length)

    # Create mask (1 for valid positions, 0 for padding)
    mask = (padded_manifold_idx != -1).float()

    # Replace -1 with 0 to avoid indexing errors in one-hot
    safe_manifold_idx = padded_manifold_idx.clone()
    safe_manifold_idx[safe_manifold_idx == -1] = 0

    # If manifold_index is shape (B, T, 1), squeeze it to (B, T)
    if safe_manifold_idx.ndim == 3:
        safe_manifold_idx = safe_manifold_idx.squeeze(-1)

    # One-hot encode manifold indices
    manifold_onehot = F.one_hot(safe_manifold_idx.long(), num_classes=num_embed_vq).float()
    # Shape: (B, target_length, num_embed_vq)

    # Pad motion to target_length with zeros
    padded_motion = torch.stack([
        F.pad(torch.tensor(m, dtype=torch.float32), (0, 0, 0, target_length - m.shape[0]))
        for m in motion
    ])

    padded_rela_trajectory = repeat_to_target(relative_trajectory, lengths, target_length)
    padded_gt_rela_trajectory = repeat_to_target(gt_relative_trajectory, lengths, target_length)
    # Stack text embeddings
    text_embed = torch.stack(text_embed_list)  # Shape: (B, D_text)
    return {
        'angle_xy': padded_angle_xy,
        'stylecode': padded_stylecode,
        'manifold_onehot': manifold_onehot,
        'manifold': padded_manifold,
        'text_embed': text_embed,
        'mask': mask,
        'lengths': torch.tensor(lengths),
        'text': text_list,
        'motion': padded_motion,
        'relative_trajectory': padded_rela_trajectory,
        'gt_relative_trajectory': padded_gt_rela_trajectory
    }


class CachedText2MotionDataset(Dataset):
    def __init__(self, cache_path, gen_phase_path=None, gt_phase_path=None):
        # Original GT data
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        self.data_dict = cached['data_dict']
        self.name_list = cached['name_list']
        self.text2phase = cached.get('text2phase', None)

        # Load generated phase results if available
        self.gen_manifold = None
        self.gen_stylecode = None
        self.gen_traj = None
        self.gen_length = None
        if gen_phase_path is not None:
            gen_data = torch.load(gen_phase_path, map_location='cpu', weights_only=False)
            self.gen_manifold = gen_data['manifold'].permute(0, 2, 1)           # shape: (N, T, D)
            self.gen_stylecode = gen_data['stylecode'].permute(0, 2, 1)         # shape: (N, D, T)
            self.gen_traj = gen_data['relative_traj'].permute(0, 2, 1)
            self.gen_length = gen_data['length']                # list of int
        if gt_phase_path is not None:
            gt_data = torch.load(gt_phase_path, map_location='cpu', weights_only=False)
            self.gt_traj = gt_data['relative_traj'].permute(0, 2, 1)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        key = self.name_list[idx]
        data = self.data_dict[key]

        # Original data
        angle_xy = data['angle_xy']
        manifold = self.gen_manifold[idx]
        stylecode = self.gen_stylecode[idx]
        manifold_index = data['manifold_index']
        relative_traj = self.gen_traj[idx]
        m_length = self.gen_length[idx]
        text_list = data['text']
        motion = data['motion']
        gt_relative_traj = self.gt_traj[idx]

        # Randomly choose one text prompt
        text_data = random.choice(text_list)
        text_embed = text_data['text_embed']
        caption = text_data['caption']

        return {
            'angle_xy': angle_xy,
            'stylecode': stylecode,
            'manifold_index': manifold_index,
            'manifold': manifold,
            'm_length': m_length,
            'text_embed': text_embed,
            'text': caption,
            'motion': motion,
            'relative_trajectory': relative_traj,
            'gt_relative_trajectory': gt_relative_traj
        }