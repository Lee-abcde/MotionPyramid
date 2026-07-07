# -*- coding: utf-8 -*-

import torch

def export_diffusion_to_onnx(
    model, motion_noise, motion_root12, stylecode, manifold, batch_size, export_path="diffusion_model.onnx"
):
    model.eval()
    timestep_tensor = torch.full((batch_size,), 999, dtype=torch.long)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, motion_noise, stylecode, manifold, motion_root12, timesteps, **kwargs):
            x = torch.cat([motion_noise, stylecode, manifold, motion_root12], dim=1)
            out = self.model(x, timesteps)
            return out

    wrapper = Wrapper(model)

    torch.onnx.export(
        wrapper,
        (motion_noise, stylecode, manifold, motion_root12, timestep_tensor),
        export_path,
        input_names=["motion_noise", "stylecode", "manifold", "motion_root12", "timesteps"],
        output_names=["predicted_motion"],
        dynamic_axes={
            "motion_noise": {0: "batch_size", 2: "seq_len"},
            "stylecode": {0: "batch_size", 2: "seq_len"},
            "manifold": {0: "batch_size", 2: "seq_len"},
            "motion_root12": {0: "batch_size", 2: "seq_len"},
            "timesteps": {0: "batch_size"},
            "predicted_motion": {0: "batch_size", 2: "seq_len"},
        },
        opset_version=18,
    )

    print(f"Exported to {export_path}")




