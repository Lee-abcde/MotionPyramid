import numpy as np
import torch
import torch.nn as nn
from models.Unet import CondUnet1D
import torch.nn.functional as F
import clip


class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, nrootfeats, nstylefeats, ncond_feats, ncontactfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, generated_phase=False, use_manifold=False, **kargs):
        super().__init__()
        
        self.use_manifold = use_manifold

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.nrootfeats = nrootfeats
        self.ncond_feats = ncond_feats
        self.nstylefeats = nstylefeats
        self.ncontactfeats = ncontactfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        if dataset == 'text2phase' and arch == 'trans_enc':
            self.input_feats = self.nstylefeats
        else:
            self.input_feats = (self.njoints * self.nfeats + self.nrootfeats) + self.nstylefeats + self.ncond_feats + self.nrootfeats
        self.output_feats = self.njoints * self.nfeats + self.nrootfeats + self.ncontactfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.root_module_pos_dim = 3
        self.root_module_rot_dim = 9
        self.stylecode_replacement = nn.Parameter(
            torch.zeros(self.nstylefeats)
        )
        self.phase_replacement = nn.Parameter(
            torch.zeros(self.ncond_feats)
        )
        self.pos_replacement = nn.Parameter(
            torch.zeros(self.root_module_pos_dim)
        )
        self.rot_replacement = nn.Parameter(
            torch.zeros(self.root_module_rot_dim)
        )
        self.mask_prob = 0.1
        self.generated_phase = generated_phase

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation,
                                                              batch_first=True)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'unet':
            print("Unet init")
            n_feats = 512
            text_latent_dim = 256
            dim_mults = [2, 2, 2, 2]
            self.seqTransEncoder = CondUnet1D(
                input_dim=n_feats,
                cond_dim=text_latent_dim,
                dim_mults=dim_mults,
                adagn=True,
                zero=True,
                dropout=0.1,
                no_eff=False,
            )
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                print('EMBED TEXT')
                self.text_encoder_type = kargs.get('text_encoder_type', 'clip')
                # if self.text_encoder_type == "clip":
                #     print('Loading CLIP...')
                #     self.clip_version = clip_version
                #     self.clip_model = self.load_and_freeze_clip(clip_version)
                #     self.encode_text = self.clip_encode_text
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                print('EMBED ACTION')

        self.output_process = OutputProcess(self.data_rep, self.output_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        if self.dataset == 'text2phase':
            if self.use_manifold:
                self.manifold_head = nn.Sequential(
                    nn.Linear(latent_dim, self.nstylefeats - 13) # ncond_feats is used to store n_latent in this context
                )
                self.style_head = nn.Linear(latent_dim, 1)
                self.trajecotry_head = nn.Sequential(
                    nn.Linear(latent_dim, 12)
                )
            else:
                self.phase_head = nn.Sequential(
                    nn.Linear(latent_dim, 512)
                )
                self.angle_head = nn.Linear(latent_dim, 2)
                self.style_head = nn.Linear(latent_dim, 1)
                self.trajecotry_head = nn.Sequential(
                    nn.Linear(latent_dim, 12)
                )

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cuda',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    def clip_encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        if isinstance(raw_text, list):
            processed_text = [' '.join(t) if isinstance(t, list) else t for t in raw_text]
        else:
            processed_text = raw_text
        max_text_len = 20 if self.dataset in ['humanml', 'kit', 'text2phase'] else None  # Specific hardcoding for humanml dataset
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
        return self.clip_model.encode_text(texts).float().unsqueeze(0)
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        batch_size = len(raw_text)  # Get the batch size of input text
        device = next(self.parameters()).device  # Determine runtime device

        # Randomly generate a float tensor with shape (batch_size, 512) as a substitute
        return torch.randn(batch_size, 512, device=device).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if self.training and self.dataset != 'text2phase':
            # we randomly mask phase, position, rotation signal during phase2motion training
            batch_size, n_feats, seq_len = x.shape  # [batch, features, seq_len]

            def apply_mask(x_section, mask_prob, replacement):
                mask = (torch.rand(batch_size, 1, 1, device=x.device) < mask_prob).float()
                expanded_mask = mask.expand_as(x_section)
                replacement = replacement.unsqueeze(0).unsqueeze(-1).expand_as(x_section)
                return x_section * (1 - expanded_mask) + replacement * expanded_mask

            # Phase masking
            x_phase = x[:, -(self.ncond_feats + self.nrootfeats):-self.nrootfeats, :]
            x_phase = apply_mask(x_phase, self.mask_prob, self.phase_replacement)
            x[:, -(self.ncond_feats + self.nrootfeats):-self.nrootfeats, :] = x_phase

            # Position and Rotation masking
            x_pos = x[:, -12:-9, :]
            x_rot = x[:, -9:, :]

            x_pos = apply_mask(x_pos, self.mask_prob, self.pos_replacement)
            x_rot = apply_mask(x_rot, self.mask_prob, self.rot_replacement)

            x[:, -12:, :] = torch.cat([x_pos, x_rot], dim=1)
        # elif self.dataset != 'text2phase' and self.generated_phase:
        #     batch_size, n_feats, seq_len = x.shape
        #     pos_replacement = self.pos_replacement.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len)
        #     x[:, -12:-9, :] = pos_replacement
        #     rot_replacement = self.rot_replacement.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len)
        #     x[:, -9:, :] = rot_replacement
        #     x[:, -12:, :] = torch.cat([pos_replacement, rot_replacement], dim=1)

            # phase mask
            # x[:, -(self.ncond_feats+self.nrootfeats):-(self.nrootfeats), :] = self.phase_replacement.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len)

        # we add diffusion step t to encoding
        if self.arch != 'unet':
            time_emb = self.embed_timestep(timesteps)  # [1, bs, d]
        if self.cond_mode != 'no_cond':
            force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            if 'text_embed' in y.keys():  # caching option
                enc_text = y['text_embed']
            else:
                enc_text = self.encode_text(y['text']).squeeze(dim=0)
            # text_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))  # casting mask for the single-prompt-for-all case
            text_emb = self.embed_text(enc_text)
            if self.arch == 'unet':
                text_emb = text_emb.unsqueeze(2).repeat(1, 1, x.shape[2])
                # text_emb = text_emb.permute(2, 0, 1)  # (208, 32, 512)
                # text_emb = self.sequence_pos_encoder(text_emb)  # (208, 32, 512)
                # text_emb = text_emb.permute(1, 2, 0)  # (32, 512, 208)
                x = torch.cat([x, text_emb], dim=1)
            else:
                emb = text_emb + time_emb

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the timestep embed
            xseq = torch.cat([emb, x], dim=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            xseq = xseq.permute((1, 0, 2))
            output = self.seqTransEncoder(xseq).permute((1, 0, 2))  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]
            output = output[1:]
        elif self.arch == 'unet':
            xseq = x.permute((1, 2, 0))
            output = self.seqTransEncoder(xseq, timesteps, None, None)
            output = output.permute((2, 0, 1))

        if self.dataset != 'text2phase':
            output = self.output_process(output)  # input[seqlen,batch,dim] output[bs, njoints * nfeats + nconds, nframes]
            return output
        else:
            if self.use_manifold:
                manifold_output = self.manifold_head(output)
                style_output = self.style_head(output)
                trajectory_output = self.trajecotry_head(output)
                final_output = torch.cat([manifold_output, style_output, trajectory_output], dim=2)
            else:
                phase_logits = self.phase_head(output)
                angle_output = self.angle_head(output)
                style_output = self.style_head(output)
                angle_output = F.normalize(angle_output, dim=2)  # normalize along dim=1
                trajectory_output = self.trajecotry_head(output)
                final_output = torch.cat([phase_logits, angle_output, style_output, trajectory_output], dim=2)
            return final_output.permute(1, 2, 0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        pos_emd = self.pe[:x.shape[0], :]
        x = x + pos_emd
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # bs, njoints, nfeats, nframes = x.shape
        # bs, nall_feats, nframes = x.shape
        x = x.permute((2, 0, 1))

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), dim=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        # output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 0)  # [bs, njoints * nfeats + nconds, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output