import torch
import numpy as np
from modules import NormalizedMLP, NormalizedCNN, StyleConditionedDecoder
import torch.nn as nn

# Experimental phase decoder; the MLP input here is a single phase
# It reconstructs a single pose
def create_model_from_args(args, dataset):
    sep_dim = dataset.feature_dims[0]

    std_in = dataset.data_std[:sep_dim]
    mean_in = dataset.data_mean[:sep_dim]
    std_out = dataset.data_std[sep_dim:]
    mean_out = dataset.data_mean[sep_dim:]

    model = NormalizedMLP(std_in, mean_in, std_out, mean_out, n_layers=args.n_layers, activation=args.activation)
    return model


def create_cnn_model_from_args(args, dataset):
    sep_dim = dataset.feature_dims[0]

    std_in = dataset.data_std[:sep_dim][..., None]
    mean_in = dataset.data_mean[:sep_dim][..., None]
    std_out = dataset.data_std[sep_dim:][..., None]
    mean_out = dataset.data_mean[sep_dim:][..., None]

    model = NormalizedCNN(std_in, mean_in, std_out, mean_out, n_layers=args.n_layers, kernel_size=args.kernel_size,
                          activation=args.activation, use_down_up=args.use_down_up)
    return model


def create_model_from_args2(args, dataset):
    std_out = dataset.get_feature_by_names(dataset.data_std[:, None], args.needed_channel_names_phase_decoder)[:, 0]
    mean_out = dataset.get_feature_by_names(dataset.data_mean[:, None], args.needed_channel_names_phase_decoder)[:, 0]
    std_in = np.ones(args.n_latent_channel)
    mean_in = np.zeros(args.n_latent_channel)
    style_dim = args.stylecode_dim if hasattr(args, 'stylecode_dim') else 1
    model = StyleConditionedDecoder(std_in, mean_in, std_out, mean_out, style_dim=style_dim, n_layers=args.n_layers_phase_decoder, activation=args.activation_phase_decoder)
    return model


class NamedOutputModel(nn.Module):
    def __init__(self, model, feature_dims, style_dims):
        super().__init__()
        self.model = model
        self.feature_dims = feature_dims
        self.style_dims = style_dims

    def forward(self, x, style_code):
        output = self.model.forward(x, style_code)
        if int(output.shape[-1]) != int(sum(self.feature_dims)):
            raise ValueError(f"Output shape {output.shape[-1]} does not match feature_dims {sum(self.feature_dims)}")
        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d])
            output = output[..., d:]
        return outputs


class NamedCNNModel(nn.Module):
    def __init__(self, model, feature_dims):
        super().__init__()
        self.model = model
        self.feature_dims = feature_dims

    def forward(self, x):
        output = self.model.forward(x)
        assert output.shape[-2] == sum(self.feature_dims)
        outputs = []
        for d in self.feature_dims:
            outputs.append(output[..., :d, :])
            output = output[..., d:, :]
        return outputs


def export_named_onnx(model, input_shapes, filename, output_names, feature_dims, style_dims=1, dynamic_axes=None, is_cnn=False):
    model.eval()
    device = list(model.parameters())[0].device
    dummy_inputs = [torch.randn(*shape, device=device) for shape in input_shapes]
    if is_cnn:
        named_model = NamedCNNModel(model, feature_dims)
    else:
        named_model = NamedOutputModel(model, feature_dims, style_dims)
    torch.onnx.export(named_model, tuple(dummy_inputs), filename, verbose=False, input_names=['input'],
                      output_names=output_names, dynamic_axes=dynamic_axes)
