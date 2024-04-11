import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from homo_transformer.model.norm_opt.interpolation import Interpolator
from homo_transformer.model.norm_opt.optimization import calculate_delta, DampingNet
from homo_transformer.model.norm_opt.losses import scaled_barron

import torchvision
from PIL import Image

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def inverse(E):
    # inverse E(pre->cur) to (cur->pre)
    R = torch.transpose(E[..., :3, :3], -2, -1) # inverse R: b n 3 3
    T = -R @ E[..., :3, -1:]  # inverse T: b n 3 1
    return R,T

def getRT(E):
    R = E[..., :3, :3] # R: b n 3 3
    T = E[..., :3, -1:]  # T: b n 3 1
    return R,T


def homography_trans(image, I_cur_inv, I_ref, E_ref2cur, height_camera, norm):
    # inputs:
    #   image: ref image (bn,c,h,w)
    #   I_cur_inv:(b,3,3); I_ref: 3*3 K matrix (b,n,3,3)
    #   E_ref2cur: extrinsic matrix(cur->ref) (b,n,3,3)
    #   height_camera: float
    #   norm: 1*3 matrix, road surface norm vec (b,1,3)
    # return:
    #   out: homography transformed to cur image (b,n,c,h,w)

    _,c,h,w = image.size()
    b,n,_,_ = I_ref.size()

    # get back warp matrix
    R,T = getRT(E_ref2cur) #RT cur2ref
    i = torch.arange(0, h)
    j = torch.arange(0, w)
    ii, jj = torch.meshgrid(i, j)  # i:h,j:w
    ones = torch.ones_like(ii)
    uv1 = torch.stack([jj, ii, ones], dim=-1).float()  # shape = [h, w, 3]

    # H = K(R-tn/d)inv(K)
    H = I_ref @ (R - (T @ norm[:,None,:,:] / height_camera)) @ I_cur_inv[:,None,:,:] #[b,n,3,3]

    # project to camera
    uv1 = torch.einsum('bnij, hwj -> bnhwi', H, uv1.to(H))  # shape = [b,n,h,w,3]
    # only need view in front of camera ,Epsilon = 1e-6
    uv_last = torch.maximum(uv1[..., 2:], torch.ones_like(uv1[..., 2:]) * 1e-6)
    uv = uv1[..., :2] / uv_last  # shape = [b,n,h,w,2]

    # lefttop to center
    uv_center = uv - torch.tensor([w // 2, h // 2]).to(uv)  # shape = [h,w,2]
    # u:south, v: up from center to -1,-1 top left, 1,1 buttom right
    scale = torch.tensor([w // 2, h // 2]).to(uv_center)
    uv_center /= scale

    out = F.grid_sample(image, uv_center.flatten(0,1), mode='bilinear',
                        padding_mode='zeros')
    return out

def save_homography_trans_image(image_ref, img_cur, I_cur_inv, I_ref, E_ref2cur, height_camera, norm, idx):
    homo_img = homography_trans(image_ref[None,...], I_cur_inv[None,...], I_ref[None,None,...], E_ref2cur[None,None,...], height_camera, norm[None,...])
    homo_img = torchvision.transforms.functional.to_pil_image(homo_img.squeeze(0), mode='RGB')
    homo_img = homo_img.convert("RGBA")
    ori_img = torchvision.transforms.functional.to_pil_image(img_cur.squeeze(0), mode='RGB')
    ori_img = ori_img.convert("RGBA")
    save_img = Image.blend(ori_img, homo_img, alpha=.5)
    save_img.save('check_norm_%d.png' % (idx))
    return


def gen_sample_points(h_start, h_end, w_ignore, height, width):
    xs_start = torch.linspace(0.49, w_ignore, height)
    xs = torch.linspace(0., 1., width)[None,:] * (1 - 2 * xs_start)[:,None] + xs_start[:,None]
    ys = torch.linspace(h_start, h_end, height)
    indices = torch.stack((xs, ys[:,None].repeat(1,width)), -1)  # 2 h w
    return indices

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices

def calculate_J_2d_3d(p3d):
    # jacobin of [u,v] = [x/z, y/z]
    x, y, z = p3d[..., 0], p3d[..., 1], p3d[..., 2]
    zero = torch.zeros_like(z)
    z = z.clamp(min=1e-7)
    J = torch.stack([
        1 / z, zero, -x / z ** 2,
        zero, 1 / z, -y / z ** 2], dim=-1)

    J = J.reshape(p3d.shape[:-1] + (2, 3))
    return J

def calculate_J_3n_2n(norm):
    # # jacobin of [n1, n2, sqrt(1-n1**2-n2**2)] [[1,0],[0,1],[-n1/sqrt(1-n1**2-n2**2],-n2/sqrt(1-n1**2-n2**2]]
    # zero = torch.zeros_like(norm[..., 0])
    # one = torch.ones_like(norm[..., 0])
    # n3 = torch.where(norm[..., 2]==0, 1., norm[..., 2]) # for \0
    # J = torch.stack([
    #     one, zero,
    #     zero, one,
    #     -norm[..., 0]/n3, -norm[..., 1]/n3], dim=-1)
    #
    # J = J.reshape(norm.shape[:-1] + (3, 2))
    # jacobin of [n1, n2, n3] = [-sin(roll)*cos(pitch), -cos(roll)cos(pitch), sin(pitch)]
    # pitch&roll [sin(roll)sin(pitch), -cos(roll)cos(pitch)][cos(roll)sin(pitch),sin(roll)cos(pitch)][cos(pitch),0]
    # [-n1*n3/sqrt(1-n3**2), n2][-n2*n3/sqrt(1-n3**2),-n1][sqrt(1-n3**2),0]
    zero = torch.zeros_like(norm[..., 0])
    cos_pitch = torch.sqrt(1-norm[..., 2]**2) #sqrt(1-n3**2)
    J = torch.stack([
        -norm[..., 0]*norm[..., 2]/cos_pitch, norm[..., 1],
        -norm[..., 1]*norm[..., 2]/cos_pitch, -norm[..., 0],
        cos_pitch, zero], dim=-1)

    J = J.reshape(norm.shape[:-1] + (3, 2))
    return J

def calculate_norm(pitch, roll):
    n1 = -torch.sin(roll) * torch.cos(pitch)
    n2 = -torch.cos(roll) * torch.cos(pitch)
    n3 = torch.sin(pitch)
    return torch.cat((n1,n2,n3), dim=-1)

def fuse_features(features, batch_size):
    """ fuse feature maps in different level.
    Args:
        features: list of torch.Tensor with different size.
    Returns:
        fused first two feature maps.
    """
    fused = None
    _,_,H,W = features[0].size() #bn, c, h w
    for level in range(len(features)):
        # choose first two
        _, c, h, w = features[level].size()
        f_cur = features[level].reshape(batch_size,-1,c,h,w)
        f_cur = f_cur[:,:2] #b, 2, c, h w
        f_cur = torch.flatten(f_cur, 0, 1) #b2, c, h w
        if w != W or h != H:
            f_cur = torch.nn.functional.interpolate(f_cur, size=(H,W), mode='bilinear')
        if fused is None:
            fused = f_cur
        else:
            fused = torch.cat((fused, f_cur), dim=1) # b2, c+c~, H, W
    return fused.view(batch_size,2,-1,H,W)


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class HomoGuideAttention(nn.Module):
    def __init__(self, dim, norm=nn.LayerNorm):
        super().__init__()
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)
        self.interpolator = Interpolator()

    def forward(self, query, ref, homo_uv, skip=None):
        """
        query: (b d h w)
        ref: (b n d h w)
        homo_uv: (b n 2 h w) in 0~1, 00: left top; 11:right bottom
        """
        b, n, d, H, W = ref.shape

        # get k
        homo_uv = rearrange(homo_uv, 'b n c H W -> (b n) (H W) c')  #(bn hw 2)
        homo_uv = homo_uv * torch.tensor([W,H]).to(homo_uv) #0~1 to 0~w or h
        val, mask, _ = self.interpolator(ref.flatten(0,1), homo_uv)  #(bn hw d)

        # normalize
        key = torch.nn.functional.normalize(val, dim=2)  #(bn hw d)
        key = key.view(b,n,-1,d) #(b n hw d)
        q = torch.nn.functional.normalize(query.flatten(-2), dim=1)  # (b d hw)

        # Dot product attention along n
        dot = torch.einsum('b d p, b n p d -> b n p', q, key)
        dot = dot * mask.view(b,n,-1)
        att = dot.softmax(dim=1)  # (b n hw)

        # Combine values (image sequence features).
        z = torch.einsum('b n p, b n p d -> b p d', att, val.view(b,n,-1,d)) # (b hw d)
        z = rearrange(z, 'b (H W) d-> b d H W', H=H, W=W) # (b d h w)
        z = query + z

        # Optional skip connection
        if skip is not None:
            z = z + skip

        z = rearrange(z, 'b d H W -> b (H W) d')
        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)
        return z


class HomoAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        image_height: int,
        image_width: int,
        heads: int = 4,
        sequence_cnt: int = 2,
        skip: bool = True,
    ):
        super().__init__()

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        # self.img_embed = nn.Conv2d(2, dim, 1, bias=False)

        self.attend = HomoGuideAttention(dim)
        self.skip = skip

        self.image_height = image_height
        self.image_width = image_width

        # learnable road norm
        #norm = torch.tensor([[0., -1., 0.]])
        #self.register_parameter('norm', torch.nn.Parameter(norm))

    def forward(
        self,
        # x: torch.FloatTensor,
        feature: torch.FloatTensor,
        I_src: torch.FloatTensor,
        I_tar_inv: torch.FloatTensor,
        E: torch.FloatTensor,
        dis: torch.FloatTensor,
        norm: torch.FloatTensor,
    ):
        """
        H = I_tar(R-tn/dis)I_src_inv
        x: (b, c, H, W) # current image, target feature
        feature: (b, n, dim_in, h, w) # homo trans source feature
        I_src_inv: (b, n, 3, 3)
        I_tar: (b, 3, 3)
        E: (b, n, 4, 4)
        dis: scale
        norm:(b, 1, 3)

        Returns: (b, c, H, W)
        """
        b, n, _, h, w = feature.shape # b n c h w

        pixel = generate_grid(h, w)[None].to(feature)                   # b n 3 h w

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)

        if 0: #debug no homo
            homo = pixel_flat[:,:,:2].repeat(b,n,1,1)           # b n 2 (h w)
        else:
            img_wh1 = torch.tensor([[self.image_width], [self.image_height], [1]]).to(pixel_flat)
            pixel_flat = pixel_flat * img_wh1  # form 0~1 to 0~h/w

            # calculate Homography matrix
            R, T = getRT(E)
            H = I_src@(R-T@norm[:,None,:,:]/dis)@I_tar_inv[:,None,:,:]  # b n 3 3

            homo = H.float() @ pixel_flat                                           # b n 3 (h w)
            homo = homo[:,:,:2]/homo[:,:,2:]                                        # b n 2 (h w)

            # out image check range(0~1)
            # mask = torch.logical_and(torch.logical_and(homo[:,:,0] >= 0, homo[:,:,0] <= self.image_width), torch.logical_and(homo[:,:,1] >=0, homo[:,:,1] <=self.image_height))
            # homo = homo * mask[:,:,None,:] + pixel_flat[:,:,:2] * ~mask[:,:,None,:]

            # to 0~1
            homo = homo/(img_wh1[:2])
        homo = homo.view(b,n,-1,h,w)                                            # b n 2 h w

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) c h w
        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        # embed = self.img_embed(pixel[0][:,:2])                                    # 1 d h w
        # query = val[:,0] + embed                                                # b n d h w

        #return self.attend(query, val, homo, skip=val[:, 0] if self.skip else None)
        return self.attend(val[:,0], val, homo)


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            sequence_view: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
            norm_start_h: float = 0.,
            norm_end_h: float = 1.,
            norm_ignore_w: float = 0.,
            norm_init_pitch: float = 0.,
            norm_init_roll: float = 0.,
            camera_height: float = 0.
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        homo_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            homo_att = HomoAttention(feat_height, feat_width, feat_dim, dim, **sequence_view)
            homo_views.append(homo_att)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.homo_views = nn.ModuleList(homo_views)
        self.layers = nn.ModuleList(layers)

        self.sample_points = gen_sample_points(norm_start_h, norm_end_h, norm_ignore_w, 30, 60)
        self.interpolator = Interpolator()
        self.robust_loss = scaled_barron(0, 0.1)

        self.init_pitch = norm_init_pitch
        self.init_roll = norm_init_roll
        self.camera_dis = camera_height

        if sequence_view.sequence_cnt > 1:
            self.dampingnet = DampingNet()

        #debug
        # self.save_index = 0


    def optimize_norm(self, feature, I_src, I_tar_inv, E, dis, norm, damping):
        # features: b s+1,c,h,w
        # I_src: b,s,3,3
        # I_tar_inv: b,3,3
        # E: b,s,4,4
        # dis: float
        # norm: b,1,3
        # return: delta_norm: b 2

        # merge first two features in different level
        b, s, _, _ = I_src.shape

        # get feature in sample points
        _,_,_,h,w = feature.shape
        cur_sample_points = self.sample_points * torch.tensor([w,h]) # h w 2
        cur_sample_points = cur_sample_points.view(1,-1,2).repeat(b,1,1).to(feature) #b,n,2
        cur_feature, valid, _ = self.interpolator(feature[:,0], cur_sample_points)

        # add 1 to homogeneous
        cur_sample_homogeneous = torch.cat((cur_sample_points, torch.ones_like(cur_sample_points[:,:,:1])), dim=-1) # b n 3

        R, T = getRT(E)
        H = I_src @ (R - T @ norm[:,None] / dis) @ I_tar_inv[:,None]  # b s 3 3

        ref_sample_points = torch.einsum('bsij,bnj->...bsni', H.float(), cur_sample_homogeneous) # b s n 3
        ref_sample_points_uv = ref_sample_points[..., :2] / ref_sample_points[..., 2:]  # b s n 2
        ref_feature, valid_ref, gradients = self.interpolator(feature[:,1:].flatten(0,1), ref_sample_points_uv.flatten(0,1), return_gradients=True)
        ref_feature = rearrange(ref_feature, '(b s) n c -> b s n c', b=b, s=s)
        valid_ref = rearrange(valid_ref, '(b s) n -> b s n', b=b, s=s)
        gradients = rearrange(gradients, '(b s) n c i -> b s n c i', b=b, s=s)
        valid = valid[:,None] & valid_ref
        valid = valid.flatten(1, 2)  # b sn

        # calculate residual
        res = (ref_feature - cur_feature[:,None]).flatten(1,2)*valid[..., None] # b sn c
        # calculate Jacobin (df/dp dp/dn)
        left = -torch.einsum('bsij,bsjk->...bsik', I_src, T) # b s 3 1
        right = torch.einsum('bij,bnj->...bni', (I_tar_inv/dis).float(), cur_sample_homogeneous)# b n 3
        J_3d_3n = torch.einsum('bsij,bnjk->...bsnik', left.float(), right[:,:,None,:])# b s n 3 3
        J_3n_2n = calculate_J_3n_2n(norm[:,0]) #b,3,2
        J_3d_2n = torch.einsum('bsnij,bjk->...bsnik', J_3d_3n, J_3n_2n.float()) # b s n 3 2
        J_2d_3d = calculate_J_2d_3d(ref_sample_points) # b s n 2 3
        J_2d_2n = torch.einsum('bsnij,bsnjk->...bsnik', J_2d_3d, J_3d_2n) # b s n 2 2
        J = torch.einsum('bsnci,bsnij->...bsncj', gradients, J_2d_2n) # b s n c 2
        J = J.flatten(1,2)  # b sn c 2

        # compute the cost and aggregate the weights
        cost = (res ** 2).sum(-1) # b sn
        cost, w_loss, _ = self.robust_loss(cost)
        weights = w_loss * valid.float() # b sn

        delta_n = calculate_delta(J, res, weights, damping)
        return delta_n

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w

        z = []
        I_ref = batch['intrinsics'].float()                     # b n 3 3
        I_cur_inv = batch['intrinsics'][:,0].inverse().float()               # b 3 3
        # norm = torch.tensor([[[0., -1., 0.]]]).repeat(b,1,1).to(I_ref) # b 1 3
        pitch = torch.ones_like(I_cur_inv[:,:1,:1])*self.init_pitch # b 1 1
        roll = torch.ones_like(I_cur_inv[:, :1, :1])*self.init_roll
        norm = calculate_norm(pitch, roll) # b 1 3
        features = [self.down(y) for y in self.backbone(self.norm(image))]

        if "norm" in batch.keys():
            norm = batch['norm'].float()
        else:
            if n >= 2:
                # calculate delta_norm through plane fit
                loop = 0
                damping = self.dampingnet()
                while (loop < 20):
                    _,c,h,w = features[0].shape
                    delta_pitch_roll = self.optimize_norm(features[0].view(b,-1,c,h,w), I_ref[:, 1:], I_cur_inv, batch['extrinsics'][:, 1:].float(),
                                                    self.camera_dis, norm, damping)
                    # delta_pitch_roll = self.optimize_norm(batch['image'], I_ref[:,1:], I_cur_inv, batch['extrinsics'][:, 1:].float(),
                    #                                 self.camera_dis, norm) # estimate norm form rgb
                    #print(loop, ": ", torch.min(delta_pitch_roll).item(), '~', torch.max(delta_pitch_roll).item())
                    pitch = pitch + delta_pitch_roll[:, None, 0:1]
                    roll = roll + delta_pitch_roll[:, None, 1:]
                    norm = calculate_norm(pitch, roll) # b 1 3

                    if torch.max(abs(delta_pitch_roll)) < 1e-4:
                        break
                    #print(loop, ": ", torch.min(delta_pitch_roll).item(), '~', torch.max(delta_pitch_roll).item())
                    loop += 1
                if loop > 5:
                    print(loop, " final pitch: ", torch.mean(pitch).item(), torch.min(pitch).item(), '~', torch.max(pitch).item()," roll:",torch.min(roll).item(), '~', torch.max(roll).item())
                if 0:#debug
                    print("pitch:", torch.min(pitch).item(), " ~ ", torch.max(pitch).item(),
                          " roll:", torch.min(roll).item(), " ~ ", torch.max(roll).item() )

                    # homography transform one image
                    debug_b = torch.argmax(pitch.view(-1))
                    save_homography_trans_image(batch['image'][debug_b,1], batch['image'][debug_b,0], I_cur_inv[debug_b],
                                                I_ref[debug_b,1], batch['extrinsics'][debug_b,1].float(), self.camera_dis, norm[debug_b], self.save_index)
                    self.save_index =0 #+= 1

        #debug for homotransformed feature maps
        if 0:
            homo_features = []
            for feature in features:
                homo_features.append(homography_trans(feature, I_cur_inv, I_ref, batch['extrinsics'].float(), self.camera_dis, norm))
            features = homo_features

        for homo_view, feature, layer in zip(self.homo_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = homo_view(feature, I_ref, I_cur_inv, batch['extrinsics'].float(), self.camera_dis, norm)
            x = layer(x)
            z.append(x)

        return z
