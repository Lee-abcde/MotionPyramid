import os
import os.path as osp

import Library.Utility as utility
import numpy as np


from option import TrainVQOptionParser


# from dataset import create_dataset_from_args



def write_motion2npz(motion_array, motion_std, motion_mean, frame_num, output_path="output_motion.npz", rootTransUInfo = False, Footcontact_label=None, writetrajectoryonly=False):
    # Assume motion_array has shape (frames, joints * 15)
    # Each joint has 3 position values, 3 velocity values, and 9 rotation values (3x3 rotation matrix)
    # frame_num can use [0, 32] to output frames 0-31
    assert motion_array.shape[1] in {432, 420, 405, 417,
                                     570, 582}, "The last dimension of motion_array must be 432, 420, 405, 417, or 570."
    assert Footcontact_label is None or isinstance(Footcontact_label, np.ndarray), "Footcontact_label must be None or a numpy array."

    motion_array = motion_array * motion_std + motion_mean
    # motion_array_debug = motion_array.numpy()
    joint_num = int(motion_array.shape[1] / 15)  # each joint has 15 values

    if isinstance(frame_num, (int, float)):
        frames = frame_num
        start_frame = 0
        end_frame = frame_num
    else:
        start_frame = frame_num[0]
        end_frame = frame_num[1]
        frames = end_frame - start_frame

    positions = np.zeros((joint_num * 3, frames), dtype=np.double)  # Joint positions: 3 coordinates
    velocities = np.zeros((joint_num * 3, frames), dtype=np.double)  # Joint velocities: 3 coordinates
    rotations = np.zeros((joint_num * 9, frames), dtype=np.double)  # Joint rotations: 9 matrix elements
    rootinfo = np.zeros((12, frames), dtype=np.double)
    foot_contactlabel = np.zeros((2, frames), dtype=np.double)

    i = 0
    # Iterate over data for each frame
    for f in range(start_frame, end_frame):
        for j in range(joint_num):
            # start index of each joint data block
            idx = j * 3
            rotation_idx = j * 9
            position_offset = 3 * joint_num
            rotation_offset = 6 * joint_num

            velocities[idx:idx + 3, i] = motion_array[f, idx:idx + 3]

            positions[idx:idx+3, i] = motion_array[f, position_offset + idx:position_offset + idx+3]

            # extract rotation matrix (3x3)
            rotations[rotation_idx:rotation_idx + 9, i] = motion_array[f, rotation_offset + rotation_idx:rotation_offset+rotation_idx + 9]
        rootinfo[:, i] = motion_array[f, -12:]
        if isinstance(Footcontact_label, np.ndarray):
            foot_contactlabel[:, i] = Footcontact_label[f, :]
        i = i + 1

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if writetrajectoryonly:
        positions = np.zeros_like(positions)
        velocities = np.zeros_like(velocities)
    # Save as an npz file
    if rootTransUInfo==True and isinstance(Footcontact_label, np.ndarray):
        np.savez_compressed(output_path, Positions=positions, Velocities=velocities, Rotations=rotations,
                            RootInfo=rootinfo, ContactLabel=foot_contactlabel)
    elif rootTransUInfo==True:
        np.savez_compressed(output_path, Positions=positions, Velocities=velocities, Rotations=rotations, RootInfo=rootinfo)
    else:
        np.savez_compressed(output_path, Positions=positions, Velocities=velocities, Rotations=rotations)

    print(f"Motion data saved to {output_path}")

def main():
    option_parser = TrainVQOptionParser()
    args = option_parser.parse_args()
    # make sure the binary data are same as the Unity asset
    # args.normalize = 0
    # the input storage path
    Save = args.save
    utility.MakeDirectory(Save)
    # write args.txt
    with open(osp.join(Save, "args.txt"), "w") as file:
        file.write(option_parser.text_serialize(args))
    args = option_parser.post_process(args)

    # Create log folder
    log_dir = osp.join(Save, 'log')
    if os.path.exists(log_dir) and 'test' not in log_dir:
        print('log dir exists, remove it [y/n]?')


    if osp.exists(log_dir):
        os.system(f'rm -rf {log_dir}')


    # load motion data
    motion_datas = create_dataset_from_args(args)
    # for motion_data in motion_datas:
    #     motion_data.Data = motion_data.Data * motion_data.data_std + motion_data.data_mean
    start_frame = 0
    frame_num = (start_frame, start_frame+2400)
    index = 0
    for motion_data in motion_datas:
        write_motion2npz(motion_data.Data, motion_data.data_std, motion_data.data_mean, frame_num, f"results/visualnpz/output_motion{index:03}.npz", True)
        index += 1  # increment index on each loop


def check_npz(file_path):
    data = np.load(file_path)
    for key in data.files:
        print(f"{key}: {data[key].shape}")
    # print(data['Positions'].shape)  # check Positions shape
    # print(data['Velocities'].shape)  # check Velocities shape
    # print(data['Rotations'].shape)  # check Rotations shape

if __name__ == '__main__':
    main()
    # check_npz("results/visualnpz/output_motion000.npz")
