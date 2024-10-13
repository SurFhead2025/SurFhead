import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from utils.rigid_utils import exp_se3
from utils.quaternion_utils import init_predefined_omega
# from utils.general_utils import linear_to_srgb
# from utils.ref_utils import generate_ide_fn
# import nvdiffrast.torch as dr


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def positional_encoding(positions, freqs):
    freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


class RenderingEquationEncoding(torch.nn.Module):
    def __init__(self, num_theta, num_phi, device, type='asg'):
        super(RenderingEquationEncoding, self).__init__()

        self.num_theta = num_theta
        self.num_phi = num_phi
        if type == 'asg':
            sample_type = 'full'
        elif type == 'lasg':
            sample_type = 'frontal'
        else:
            sample_type = 'full'
        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi, type=sample_type)
        self.omega = omega.view(1, num_theta, num_phi, 3).to(device)#! incoming direction; lobe direction
        self.omega_la = omega_la.view(1, num_theta, num_phi, 3).to(device)#! incoming tangent direction
        self.omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to(device)#! incoming bitangent direction

        #! theta = 4, phi = 8
    
    def forward(self, omega_o, a, la, mu, sg_type):
        #! omega_o :: input direction(reflection direction from view direction)
        # breakpoint()
        if sg_type == 'asg' or sg_type == 'lasg':
            Smooth = F.relu((omega_o[:, None, None] * self.omega).sum(dim=-1, keepdim=True))  # N, num_theta, num_phi, 1
            #! smooth.shape = N,4,8,1
            la = F.softplus(la - 1) #! lambda
            mu = F.softplus(mu - 1) #! mu
            #! self.omega_la.shape = 1,4,8,3
            #! omega_o.shape = N,3->N,1,1,3
            #! la.shape = N,4,8,1 -> scalar
            #! a.shape = N,4,8,2
            # breakpoint()
            exp_input = -la * (self.omega_la * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2) - mu * (
                    self.omega_mu * omega_o[:, None, None]).sum(dim=-1, keepdim=True).pow(2)
            # if sg_type == 'asg': 
            #     out = a * Smooth * torch.exp(exp_input) #! a.shape = N,4,8,2
            # else:
            #     out = Smooth * torch.exp(exp_input) #! a.shape = N,4,8,1
            
            out = a * Smooth * torch.exp(exp_input)

            #!out.shape = N,4,8,2
        if sg_type == 'sg':
            # breakpoint()
            la = F.softplus(la - 1) #! lambda
            mu = F.softplus(mu - 1) #! mu
            
            
            # C = 1 / (np.sqrt(2) * np.pi **(2/3) * la)
            cos_value = (self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)
            clamped_cos = torch.clamp(cos_value, -1 + 1e-11, 1 - 1e-11)  # 범위 조정
            #! torch.Size([10144, 4, 8, 1])
            exp_input = la * (clamped_cos - 1)
            # exp_input = - 0.5 * (torch.arccos((self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)) / la ).pow(2)
            # out = a * C * torch.exp(exp_input)
            out = (mu * torch.exp(exp_input)).repeat(1,1,1,2)
        
        if sg_type == 'sg_angle':    
            # la = torch.abs(la)
            breakpoint()
            la = torch.exp(la)
            la = torch.clamp(la, min=1e-2)  
            C = 1 / (np.sqrt(2) * np.pi **(2/3) * la)
            cos_value = (self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)
            clamped_cos = torch.clamp(cos_value, -1 + 1e-11, 1 - 1e-11)  # 범위 조정
            exp_input = -0.5 * (torch.arccos(clamped_cos) / la).pow(2)
            # exp_input = - 0.5 * (torch.arccos((self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)) / la ).pow(2)
            # out = a * C * torch.exp(exp_input)
            out = (C * torch.exp(exp_input)).repeat(1,1,1,2)
        if False:
            la = torch.abs(la)
            la = torch.clamp(la, min=1e-2)  
            C = 1 / (np.sqrt(2) * np.pi **(2/3) * la)
            cos_value = (self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)
            clamped_cos = torch.clamp(cos_value, -1 + 1e-11, 1 - 1e-11)  # 범위 조정
            exp_input = -0.5 * (torch.arccos(clamped_cos) / la).pow(2)
            # exp_input = - 0.5 * (torch.arccos((self.omega * omega_o[:, None, None]).sum(dim=-1, keepdim=True)) / la ).pow(2)
            # out = a * C * torch.exp(exp_input)
            out = (C * torch.exp(exp_input)).repeat(1,1,1,2)
        return out


class SGEnvmap(torch.nn.Module):
    def __init__(self, numLgtSGs=32, device='cuda'):
        super(SGEnvmap, self).__init__()

        self.lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
        self.lgtSGs.data[..., 3:4] *= 100.
        self.lgtSGs.data[..., -3:] = 0.
        self.lgtSGs.requires_grad = True

    def forward(self, viewdirs):
        lgtSGLobes = self.lgtSGs[..., :3] / (torch.norm(self.lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-7)
        lgtSGLambdas = torch.abs(self.lgtSGs[..., 3:4])  # sharpness
        lgtSGMus = torch.abs(self.lgtSGs[..., -3:])  # positive values
        pred_radiance = lgtSGMus[None] * torch.exp(
            lgtSGLambdas[None] * (torch.sum(viewdirs[:, None, :] * lgtSGLobes[None], dim=-1, keepdim=True) - 1.))
        reflection = torch.sum(pred_radiance, dim=1)

        return reflection


class ASGRender(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, sg_type='asg'):
        super(ASGRender, self).__init__()

        self.num_theta = 4
        if sg_type == 'lasg':
            self.num_theta = 2
        self.num_phi = 8
        self.ch_normal_dot_viewdir = 1
        # self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2
        self.in_mlpC = 2 * viewpe * 3 + 3 + self.num_theta * self.num_phi * 2 + self.ch_normal_dot_viewdir
        
        self.viewpe = viewpe
        self.ree_function = RenderingEquationEncoding(self.num_theta, self.num_phi, 'cuda', type=sg_type)

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        if sg_type == 'asg':
            layer3 = torch.nn.Linear(featureC, 3)
        elif sg_type == 'lasg':
            layer3 = torch.nn.Linear(featureC, 1)
        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def safe_normalize(self, x, eps=1e-8):
        return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

    def forward(self, pts, viewdirs, features, normal, sg_type):
        # asg_params = features.view(-1, self.num_theta, self.num_phi, 1)  # [N, 8, 16, 4] / [N, 8, 16, 1]
        
        if False:
            asg_params = features.view(-1, self.num_theta, self.num_phi, 1)
            # la = torch.split(asg_params, 1, dim=-1)
            la = asg_params
        else:
            asg_params = features.view(-1, self.num_theta, self.num_phi, 4)
            a, la, mu = torch.split(asg_params, [2, 1, 1], dim=-1)

        reflect_dir = self.safe_normalize(self.reflect(-viewdirs, normal))
        if torch.isnan(reflect_dir).any():
            breakpoint()
        # breakpoint()
        color_feature = self.ree_function(reflect_dir, a, la, mu, sg_type)
        # breakpoint()
        # color_feature = self.ree_function(reflect_dir, la)
        # color_feature = color_feature.view(color_feature.size(0), -1, 3)
        color_feature = color_feature.view(color_feature.size(0), -1)  # [N, 256]

        normal_dot_viewdir = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)  # [N, 1]
        indata = [color_feature, normal_dot_viewdir]
        if self.viewpe > -1:
            indata += [viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        #! 64 + 1 + 3(view) + 12(view) = 80
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        if sg_type == 'lasg':
            rgb = rgb.repeat(1, 3)
        # breakpoint()
        # rgb = torch.sum(color_feature, dim=1)
        # rgb = torch.sigmoid(rgb)

        return rgb


class IdentityActivation(nn.Module):
    def forward(self, x): return x


class ExpActivation(nn.Module):
    def __init__(self, max_light=5.0):
        super().__init__()
        self.max_light = max_light

    def forward(self, x):
        return torch.exp(torch.clamp(x, max=self.max_light))


def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid',
                   exp_max=0.0) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation == 'none':
        activation = IdentityActivation()
    elif activation == 'relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module = nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module



class SpecularNetwork(nn.Module):
    def __init__(self, D=4, W=128, input_ch=3, output_ch=59, view_multires=4, multires=4, sg_type='asg'):
        super(SpecularNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.view_multires = view_multires
        self.skips = [D // 2]

        self.asg_feature = 24
        self.num_theta = 4
        if sg_type == 'lasg':
            self.num_theta = 2
        # print(sg_type)
        self.num_phi = 8
        # self.asg_hidden = self.num_theta * self.num_phi * 5
        self.asg_hidden = self.num_theta * self.num_phi * 4

        # self.embed_view_fn, view_input_ch = get_embedder(view_multires, 3)
        # self.embed_fn, xyz_input_ch = get_embedder(multires, self.asg_feature)
        # self.input_ch = xyz_input_ch

        # self.linear = nn.ModuleList(
        #     [nn.Linear(self.input_ch, W)] + [
        #         nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
        #         for i in range(D - 1)]
        # )
        # self.env_module = SGEnvmap()

        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)

        self.render_module = ASGRender(self.asg_hidden, 2, 2, 128, sg_type = sg_type)

    def forward(self, x, view, normal, sg_type):
        # v_emb = self.embed_view_fn(view)
        # x_emb = self.embed_fn(x)
        # h = torch.cat([x_emb, v_emb], dim=-1)
        # h = x
        # for i, l in enumerate(self.linear):
        #     h = self.linear[i](h)
        #     h = F.relu(h)
        #     if i in self.skips:
        #         h = torch.cat([x_emb, h], -1)
# 
        # feature = self.gaussian_feature(x)
        # spec = self.render_module(x, view, feature)
        
    # def forward(self, x, view, normal):
        feature = self.gaussian_feature(x)
        spec = self.render_module(x, view, feature, normal, sg_type)

        return spec
        # reflect = self.env_module(reflect_dir)

        return spec