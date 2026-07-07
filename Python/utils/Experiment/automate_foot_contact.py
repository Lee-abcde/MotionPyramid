import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from argparse import ArgumentParser

# Add the Python project root to sys.path so modules such as models and utils can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import random
from train_diff import create_model_and_diffusion
from option import TrainVQOptionParser
from dataset import create_mdm_dataset_from_args, create_dataset_from_args
from models import VQ as VQ_model
from utils.Experiment.phase_warping import detect_foot_contact, angle_to_phase, phase_mean_range
from generate import generate_args, load_model_wo_clip, clean_vq_state_dict, ClassifierFreeSampleModel, write_motion2npz
from utils.Experiment.phase_warping import get_manual_manifold_autofrequency

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def circular_mean_std_aggregate(phases_list):
    """
    Accept a 1D array/list containing all phases where ground contact occurs across batches and frames
    Compute circular mean and standard deviation, mapped to [-0.5, 0.5]
    """
    if len(phases_list) == 0:
        return 0.0, 0.0
    phases = np.array(phases_list)
    theta = 2 * np.pi * phases
    z = np.exp(1j * theta)
    z_mean = np.mean(z)
    mean = np.angle(z_mean) / (2 * np.pi)  # [-0.5, 0.5]
    R = np.abs(z_mean)
    circ_std = np.sqrt(-2 * np.log(R)) / (2 * np.pi)
    return mean, circ_std

def main():
    option_parser = TrainVQOptionParser()
    args = generate_args()
    
    # Force settings to match the single-point automated test
    args.batch_size = 32  # number of manifolds processed each time
    args.num_repetitions = 1
    gen_window = 4
    args.window = gen_window

    import json
    import os.path as osp
    file_path = osp.join(args.pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args_dict = json.load(f)
        import argparse
        vq_args = argparse.Namespace(**args_dict)
        vq_args = option_parser.post_process(vq_args)

    print('='*20)
    print('Loading dataset & models...')
    
    local_motion_datas = create_dataset_from_args(vq_args)
    mdm_motion_datas = create_mdm_dataset_from_args(vq_args, args)[0]
    motion_data = mdm_motion_datas
    data_frame_length = mdm_motion_datas.frames_per_window
    max_frames = int(data_frame_length) # num_repetitions is 1

    networks, VQ = VQ_model.create_model_from_args(vq_args, local_motion_datas)
    Save = args.pretrained_save
    
    ref_files = [f for f in os.listdir(Save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split('_')[0]))
    largest_epoch = ref_files[-1].split('_')[0]

    VQ_target_file = f'{largest_epoch}_{vq_args.phase_channels}Channels_VQ.pt'
    state_dict = torch.load(osp.join(Save, VQ_target_file), map_location='cpu')
    state_dict = clean_vq_state_dict(state_dict)
    VQ.load_state_dict(state_dict, strict=False)
    VQ = VQ.cuda().eval()

    from torch.utils.data import Subset, DataLoader
    test_dataset = Subset(motion_data, motion_data.test_set_index)
    test_dataset.data_std, test_dataset.data_mean = motion_data.data_std, motion_data.data_mean
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    model, diffusion = create_model_and_diffusion(args, vq_args, data_loader)
    model = model.cuda()
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.eval()

    from utils.manifold_editor import get_manual_manifold

    # Define an index range that covers the continuous/discrete distribution or selected 40 anchors
    # 512 codebook entries in VQ_target_file
    total_samples = min(512, VQ.get_weight()[0].shape[0]) # iterate according to VQ codebook size
    
    print('='*20)
    print(f'Starting automated batch evaluation for {total_samples} semantic codebook samples...')
    
    conditions = [
        {"name": "Origin (1.0x, style=0.5)", "freq_factor": 1.0, "style": 0.5},
        {"name": "Phase warping Slow (0.75x, style=0.5)", "freq_factor": 0.75, "style": 0.5},
        {"name": "Phase warping Fast (1.25x, style=0.5)", "freq_factor": 1.25, "style": 0.5},
        {"name": "Style Shift", "freq_factor": 1.0, "styles": [0.6, 0.4]},
    ]
    
    flat_conditions = []
    for cond in conditions:
        if "styles" in cond:
            for st in cond["styles"]:
                flat_conditions.append({
                    "name": f"{cond['name']} (style={st})", 
                    "freq_factor": cond["freq_factor"], 
                    "style": st
                })
        else:
             flat_conditions.append(cond)

    results = {cond["name"]: {"L": [], "R": []} for cond in flat_conditions}
    per_manifold_results = {m_id: {cond["name"]: {"L": [], "R": []} for cond in flat_conditions} for m_id in range(total_samples)}

    motion_len = max_frames
    
    # Assume the base frequency is a fixed constant, such as the paper default 26 * pi
    default_base_angle = 20 * torch.pi

    for i in tqdm(range(0, total_samples, args.batch_size), desc="Evaluating Semantic Codebook"):
        current_bs = min(args.batch_size, total_samples - i)
        
        # Manifold phase indices for this batch
        phase_indices = list(range(i, i+current_bs))

        for cond in flat_conditions:
            all_manual_manifolds = []
            all_angles = []
            
            # Generate a manifold for each semantic entry in the current batch
            for b_idx in range(current_bs):
                man_man, man_ang = get_manual_manifold(
                    angle_range=default_base_angle * cond["freq_factor"], 
                    phase_index=phase_indices[b_idx],
                    window_second=1,
                    VQ=VQ,
                    batch_size=1, 
                    time_range=motion_len
                )
                all_manual_manifolds.append(man_man)
                all_angles.append(man_ang)

            manifold = torch.cat(all_manual_manifolds, dim=0).cuda()  # [bs, ..., motion_len]
            angles = torch.cat(all_angles, dim=0).cuda()
            
            # Set style
            const_vals = torch.linspace(cond["style"], cond["style"], steps=current_bs, device='cuda')
            stylecode = const_vals.view(-1, 1, 1).expand(-1, 1, motion_len)
            
            model_kwargs = {
                'y': {
                    'text_embed': manifold,
                    'stylecode': stylecode,
                }
            }
            
            # Set relative displacement; use 0 if relative root is unknown
            relative_root = torch.zeros(current_bs, 12, max_frames).cuda()
            
            # The author's strategy to mask root and generate in place
            # relative2start_rootpos[:, -12, :] = 0
            # relative2start_rootpos[:, -11, :] = 0
            # ... and so on. We can keep it entirely zero for unconstrained masked root.
            model_kwargs['y']['relative2start_rootpos'] = relative_root

            sample_fn = diffusion.p_sample_loop
            model_kwargs['y']['skip_step'] = list(range(1000, 1000))

            # Fix seed exactly like generate.py before sampling for 1:1 reproducibility
            dynamic_seed = int(time.time()) if getattr(args, 'random_sample', False) else args.seed
            fixseed(dynamic_seed)

            sample, motion_record, df_contact_label = sample_fn(
                model,
                (current_bs, model.njoints * model.nfeats + model.nrootfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
            # Data post-processing
            sample = sample.permute(0, 2, 1) # [B, T, D]
            df_motion = diffusion.motion_postprocess_gen(sample)
            
            # Absolute root is not strictly necessary purely for foot contact detect, 
            # usually `detect_foot_contact` just needs the Cartesian positions.
            init_pos = torch.zeros([current_bs, 3]).cuda()
            init_rot = torch.eye(3).repeat(current_bs, 1, 1).cuda()
            df_absolute_motion = diffusion.transfer2absolute_batch(df_motion, init_pos, init_rot)

            # Convert to NumPy to align with the original data_std
            data_std = torch.from_numpy(motion_data.data_std).cuda().float()
            data_mean = torch.from_numpy(motion_data.data_mean).cuda().float()

            # Denormalize
            unnorm_motion = df_absolute_motion * data_std + data_mean
            unnorm_motion = unnorm_motion.reshape(current_bs, max_frames, -1) # [B, T, 405/417]
            
            # Extract Positions, usually using motion_data.indices["Positions"]
            pos_idx = motion_data.indices.get('Positions', slice(84, 165)) # according to the network definition
            positions = unnorm_motion[:, :, pos_idx] # [B, T, J*3]
            
            positions = positions.permute(2, 0, 1) # => [D, B, T] to match detect_foot_contact input
            foot_contact, l_contact, r_contact = detect_foot_contact(positions.cpu(), threshold=0.05)
            
            # Extract phase
            # angles shape: [B, 1, T] -> [B, T]
            phases = angle_to_phase(angles).squeeze(1).cpu()
            
            for b_idx in range(current_bs):
                manifold_id = phase_indices[b_idx]
                
                # All phases where the left foot contacts the ground
                contact_phases_L = phases[b_idx][l_contact[b_idx]]
                results[cond["name"]]["L"].extend(contact_phases_L.tolist())
                per_manifold_results[manifold_id][cond["name"]]["L"].extend(contact_phases_L.tolist())
                
                # All phases where the right foot contacts the ground
                contact_phases_R = phases[b_idx][r_contact[b_idx]]
                results[cond["name"]]["R"].extend(contact_phases_R.tolist())
                per_manifold_results[manifold_id][cond["name"]]["R"].extend(contact_phases_R.tolist())
                
                # Write sample motion to disk for Unity preview
                # if manifold_id < 32: # Keep disk usage reasonable by only exporting early ones
                #     npz_name = "".join([c if c.isalnum() else "_" for c in cond["name"]])
                #     output_file = os.path.join(args.output_dir, f'man{manifold_id}_{npz_name}_motion.npz')
                #
                #     df_absolute_motion_np = df_absolute_motion[b_idx].reshape(-1, df_absolute_motion.shape[2]).cpu().numpy()
                #
                #     # mock probabilities [T, 2] output exactly like test_vq / generate
                #     # Left contact logic typically mapped to 0, right to 1 depending on setup.
                #     probabilities = torch.zeros((max_frames, 2), dtype=torch.float32)
                #     probabilities[:, 0] = l_contact[b_idx].float()
                #     probabilities[:, 1] = r_contact[b_idx].float()
                #     df_contact_label = probabilities.numpy()
                #
                #     write_motion2npz(df_absolute_motion_np, motion_data.data_std, motion_data.data_mean, max_frames,
                #                      output_file, True, df_contact_label)

    print("\n" + "="*95)
    print(" " * 30 + "Per-Manifold Phase/Foot-Contact Results")
    print("="*95)
    
    valid_manifolds = []
    
    for m_id in range(total_samples):
        L_na_count = sum(1 for cond in flat_conditions if len(per_manifold_results[m_id][cond["name"]]["L"]) == 0)
        R_na_count = sum(1 for cond in flat_conditions if len(per_manifold_results[m_id][cond["name"]]["R"]) == 0)
        num_conds = len(flat_conditions)

        # Unless one foot is N/A under all conditions, meaning it is fully airborne and unused
        # If any partial N/A appears even once, discard the entire manifold
        is_L_valid = (L_na_count == 0 or L_na_count == num_conds)
        is_R_valid = (R_na_count == 0 or R_na_count == num_conds)
        
        # If both feet are N/A under all conditions, the motion is fully airborne and should also be discarded
        both_floating = (L_na_count == num_conds and R_na_count == num_conds)
        
        if both_floating or not (is_L_valid and is_R_valid):
            print(f"\n>>> [FILTERED] Manifold Phase Index: {m_id} (Inconsistent N/A or completely floating)")
            continue
            
        valid_manifolds.append(m_id)
        print(f"\n>>> Manifold Phase Index: {m_id}")
        print(f"{'Condition':<40} | {'Left Foot (Mean ± Std)':<25} | {'Right Foot (Mean ± Std)':<25}")
        print("-" * 95)
        for cond in conditions:
            if "styles" in cond:
                combined_L = []
                combined_R = []
                for st in cond["styles"]:
                    sub_name = f"{cond['name']} (style={st})"
                    combined_L.extend(per_manifold_results[m_id][sub_name]["L"])
                    combined_R.extend(per_manifold_results[m_id][sub_name]["R"])
                
                mean_L, std_L = circular_mean_std_aggregate(combined_L)
                mean_R, std_R = circular_mean_std_aggregate(combined_R)
                
                str_L = f"{mean_L:.3f} ± {std_L:.3f}" if len(combined_L) > 0 else "N/A"
                str_R = f"{mean_R:.3f} ± {std_R:.3f}" if len(combined_R) > 0 else "N/A"
            else:
                mean_L, std_L = circular_mean_std_aggregate(per_manifold_results[m_id][cond["name"]]["L"])
                mean_R, std_R = circular_mean_std_aggregate(per_manifold_results[m_id][cond["name"]]["R"])
                
                str_L = f"{mean_L:.3f} ± {std_L:.3f}" if len(per_manifold_results[m_id][cond["name"]]["L"]) > 0 else "N/A"
                str_R = f"{mean_R:.3f} ± {std_R:.3f}" if len(per_manifold_results[m_id][cond["name"]]["R"]) > 0 else "N/A"
            
            print(f"{cond['name']:<40} | {str_L:<25} | {str_R:<25}")

    print("\n" + "="*95)
    print(f" " * 15 + f"Phase/Foot-Contact Aggregated Results (Valid Manifolds: {len(valid_manifolds)}/{total_samples})")
    print("="*95)
    print(f"{'Condition':<40} | {'Left Foot (Mean ± Std)':<25} | {'Right Foot (Mean ± Std)':<25}")
    print("-" * 95)
    
    # Re-aggregate results from valid_manifolds instead of using the unfiltered results pool
    for cond in conditions:
        agg_L = []
        agg_R = []
        for m_id in valid_manifolds:
            if "styles" in cond:
                for st in cond["styles"]:
                    sub_name = f"{cond['name']} (style={st})"
                    agg_L.extend(per_manifold_results[m_id][sub_name]["L"])
                    agg_R.extend(per_manifold_results[m_id][sub_name]["R"])
            else:
                agg_L.extend(per_manifold_results[m_id][cond["name"]]["L"])
                agg_R.extend(per_manifold_results[m_id][cond["name"]]["R"])
            
        mean_L, std_L = circular_mean_std_aggregate(agg_L)
        mean_R, std_R = circular_mean_std_aggregate(agg_R)
        
        str_L = f"{mean_L:.3f} ± {std_L:.3f}" if len(agg_L) > 0 else "N/A"
        str_R = f"{mean_R:.3f} ± {std_R:.3f}" if len(agg_R) > 0 else "N/A"
        
        print(f"{cond['name']:<40} | {str_L:<25} | {str_R:<25}")

if __name__ == "__main__":
    main()
