# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation, compute_E_inverse, compute_E_inverse_GA
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, rotmat_to_rotvec
# from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype = m.dtype) @ Vh
    return  u, p

class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, sg_degree: int, disable_flame_static_offset=False, not_finetune_flame_params=False,\
        n_shape=300, n_expr=100, train_normal=False, tight_pruning=False, train_kinematic=False, DTF=False,
        invT_Jacobian = False,  densification_type='arithmetic_mean', detach_eyeball_geometry = False, \
            train_kinematic_dist = False,detach_boundary = False):
        super().__init__(sh_degree, sg_degree)
        
        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        self.flame_model = FlameHead(
            n_shape, 
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None
        self.face_adjacency = None
        self.L_points = None
        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()
        
        
        self.train_normal = train_normal
        self.tight_pruning = tight_pruning
        self.train_kinematic = train_kinematic
        self.train_kinematic_dist = train_kinematic_dist
        self.DTF = DTF
        self.invT_Jacobian = invT_Jacobian
        
   
        self.densification_type = densification_type
        self.detach_eyeball_geometry = detach_eyeball_geometry
        self.detach_boundary = detach_boundary
        
    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes): #! activated very initial when combining with dataset #! 1
        if self.flame_param is None:#!
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
            
            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if not self.disable_flame_static_offset:#!
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset, #! mayb canonical offset
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()
            
            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False): #! False #! 3
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param
        #! else
        # breakpoint()
        
        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        # breakpoint()
        
        self.update_mesh_properties(verts, verts_cano)
    
    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]
        # triangles = verts_cano[:, faces]
        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        # if self.scaling_aniso:
        #     return_dim = 2
        # else:
        return_dim = 1
        # breakpoint()
        if self.DTF:
            return_type = 'DTF'
        else:
            return_type = 'GA'
        # breakpoint()
        if return_type == 'GA':
            self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), \
                            return_scale=True,scale_dim =return_dim, return_type = return_type)
            # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma
                
        else:
            self.face_orien_mat, self.face_scaling = compute_face_orientation(verts_cano.squeeze(0), faces.squeeze(0), \
                            return_scale=True,scale_dim =return_dim, return_type = 'GA')
            
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat)) 
           
            self.E_inverse = compute_E_inverse(verts_cano.squeeze(0), faces.squeeze(0))


            
            
            def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
            #! this convention 
                #! https://blog.naver.com/PostView.naver?blogId=richscskia&logNo=222179474476
                #! https://discuss.pytorch.org/t/polar-decomposition-of-matrices-in-pytorch/188458/2
                # breakpoint()
                U, S, Vh = torch.linalg.svd(m)
                U_new = torch.bmm(U, Vh) #! Unitary
                P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S).to(dtype = m.dtype)), Vh) #! PSD
                # P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S)), Vh)
                return U_new, P
            self.face_trans_mat = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), \
                            return_scale=True,scale_dim =return_dim, return_type = return_type, E_inverse = self.E_inverse)
            self.face_R_mat, self.face_U_mat = polar_decomp(self.face_trans_mat)

            self.blended_Jacobian = None
            self.blended_R = None
            self.blended_U = None
            # print("FLUSHED, Blended Jacobian,R and U")
            # self.R_rotvec = rotmat_to_rotvec(self.R_mat).detach()           
        
           
           
        #! Q = V^tilda *V^-1 = V^tilda * I = V^tilda == Jacobian
        # for mesh rendering
        self.verts = verts
        self.faces = faces
        #* 
        #?
        # for mesh regularization
        self.verts_cano = verts_cano
        #! 0602 compute adjacent triangle
     
        if self.face_adjacency is None:
            print('Calculating Face Adjacency!')
            num_faces = faces.size(0)
            edge_to_faces = {}
            
            def add_edge(face_idx, v1, v2):
                edge = tuple(sorted([v1.item(), v2.item()]))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
            
            # Add all edges to the dictionary
            for i, face in enumerate(faces):
                add_edge(i, face[0], face[1])
                add_edge(i, face[1], face[2])
                add_edge(i, face[2], face[0])
            
            # Initialize adjacency tensor
            face_adjacency = torch.full((num_faces, 3), -1, dtype=torch.long)
            
            # Populate adjacency tensor
            for edge, face_list in edge_to_faces.items():
                if len(face_list) > 1:
                    for face in face_list:
                        adj_faces = set(face_list) - {face}
                        for adj_face in adj_faces:
                            if -1 in face_adjacency[face]:
                                idx = (face_adjacency[face] == -1).nonzero(as_tuple=True)[0][0]
                                face_adjacency[face, idx] = adj_face
         
            face_adjacency_identity = torch.arange(face_adjacency.shape[0])
       
            face_adjacency[face_adjacency == -1] = face_adjacency_identity.view(-1, 1).repeat(1, 3)[face_adjacency == -1]
            
            self.face_adjacency = torch.cat([face_adjacency_identity[...,None],face_adjacency],-1)
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):#! 2
        super().training_setup(training_args)

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)
        
        #! why comments out..?
        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)
        # breakpoint()
        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param['translation'] = flame_param['translation']
            self.flame_param['rotation'] = flame_param['rotation']
            self.flame_param['neck_pose'] = flame_param['neck_pose']
            self.flame_param['jaw_pose'] = flame_param['jaw_pose']
            self.flame_param['eyes_pose'] = flame_param['eyes_pose']
            self.flame_param['expr'] = flame_param['expr']
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)
          
            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
