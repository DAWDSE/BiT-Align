import torch

import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F

from models.dinov2.hub.backbones import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14
from models.dinov2.models import vision_transformer as vits
from models.dinov2.utils.utils import load_pretrained_weights

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


from models.model_util import *
from fast_pytorch_kmeans import KMeans

from torchvision.transforms import Compose
from tqdm import tqdm

original_forward = clip_model.encode_text


def forward_with_prompts(text, prompt_embeddings, max_length):
    

    text_embeddings = clip_model.token_embedding(text).type(clip_model.dtype)
    batch_size, num_prompts, embed_dim = text_embeddings.shape

    prompt_embeddings_expanded = prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1).type(clip_model.dtype)

    embeddings = torch.cat([prompt_embeddings_expanded, text_embeddings], dim=1).type(clip_model.dtype)

    pos_embedding = clip_model.positional_embedding.type(clip_model.dtype)#[:num_prompts + max_length]

    embeddings = embeddings + pos_embedding

    embeddings = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
    embeddings = clip_model.transformer(embeddings)
    embeddings = embeddings.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)

    x = clip_model.ln_final(embeddings)

    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
    
    return x

clip_model.encode_text = forward_with_prompts

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Net(nn.Module):

    def __init__(self, aff_classes=36):
        super(Net, self).__init__()

        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- hyper-parameters --- #
        self.aff_cam_thd = 0.8
        self.part_iou_thd = 0.6
        self.cel_margin = 0.5

        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 14
        self.patch = 14

        self.temperature = 0.1
        
        self.vit_model = torch.hub.load('/root/autodl-tmp/LOCATE-main/models/dinov2', 'dinov2_vits14',source='local')
        for name, param in self.vit_model.named_parameters():
            if "mpg_" in name:
                param.requires_grad = True
            # elif "mfa_" in name:
            #     continue
            else :
                param.requires_grad = False
        #load_pretrained_weights(self.vit_model, '/root/autodl-tmp/LOCATE-main/models/dinov2/base/dinov2_vits14_reg4_pretrain.pth', None)
        
        # --- learning parameters --- #
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)
        
        self.ego_self_attention = AttentionPool2d(16, self.vit_feat_dim, 64)#nn.MultiheadAttention(embed_dim=self.vit_feat_dim, num_heads=8, batch_first=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.ego_cls_token = nn.Parameter(torch.randn(1, 1, self.vit_feat_dim))
        self.clip_line = nn.Linear(512, self.vit_feat_dim)
        
        self.num_prompts = 6
        self.clip_embed_dim = clip_model.token_embedding.weight.size(1)  

        self.prompt_embeddings = nn.Parameter(torch.randn(self.num_prompts, self.clip_embed_dim), requires_grad=True)
        self.clip_numbers = 0


    def forward(self, exo, ego, exo_depth, ego_depth, aff_label, ego_label_name, epoch):

        num_exo = exo.shape[1]
        exo = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224
        exo_depth = exo_depth.flatten(0, 1)
        
        b = aff_label.size(0)
        
        name_list = [entry[0].replace("_", " ") for entry in ego_label_name]
        template = "a good photo of {}"
        text_list = [template.format(name) for name in name_list]    
        max_length = clip_model.context_length - self.num_prompts  
        tokens = clip.tokenize(text_list, truncate=True, context_length=max_length).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens, self.prompt_embeddings, max_length).to(torch.float32)
            
        if self.clip_numbers == 0:
            self.clip_numbers = 1
            print(text_features.size()) 
            print(text_list)

        text_features = self.clip_line(text_features)
        # --- Extract deep descriptors from DINO-vit --- #

        ego_list = self.vit_model.My_forward_features(ego, ego_depth.repeat(1, 3, 1, 1))  # attn: b x 6 x (1+hw) x (1+hw)
        exo_list = self.vit_model.My_forward_features(exo, exo_depth.repeat(1, 3, 1, 1))

        ego_desc = ego_list['x_norm_patchtokens'].detach()
        exo_desc = exo_list['x_norm_patchtokens'].detach()  
        ego_attn = ego_list['x_attention']#torch.Size([16, 6, 257, 257])
        
        ego_proj = self.aff_proj(ego_desc)
        exo_proj = self.aff_proj(exo_desc)


        
        ego_desc = self._reshape_transform(ego_desc, self.patch, self.stride)
        exo_desc = self._reshape_transform(exo_desc, self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)
        
        pre_ego = ego_proj
        image_features, ego_proj, image_features_attn  = self.ego_self_attention(ego_proj)

        image_features = F.normalize(image_features, dim=1, p=2)
        text_features = F.normalize(text_features, dim=1, p=2)

        b, c, h, w = ego_desc.shape

        #------clip_ego - fusion branch ------#

        logit_scale = self.logit_scale.exp()
        similarity_matrix = logit_scale * image_features @ text_features.t()
        text_f = torch.ones((b, 384)).cuda()
        for i in range(b):
            text_f[i] = text_features[aff_label[i]]
        
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = logit_scale *att_egoproj.permute(1, 0, 2)@text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, c)
        ego_proj = attego.permute(1, 2, 0).view(b, c, h, w)*pre_ego + pre_ego 

        
        #------- ego attention--------#

        image_features_attn = image_features_attn[:, 0, 1:].reshape(b, h, w)
        image_features_attn = (image_features_attn > image_features_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()

        ######b, c, h, w = ego_desc.shape
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, ego_attn.size(1), h, w) #b*c*h*w 
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float()
##-------------------------------------------------------------------------- 
        ego_samss = ego_cls_attn
        head_num = ego_samss.size(1)
        for head in range(head_num):
            channel_data = ego_samss[:, head, :, :]  
            normalized_channel_data = normalize_minmax(channel_data)  
            ego_samss[:, head, :, :] = normalized_channel_data  
##--------------------------------------------------------------------------        

        attn_cosine_sim = torch.zeros((b,ego_attn.size(1)))

        for i in range(ego_attn.size(1)):
            ego_proj_cls_attn = ego_cls_attn[:, i].unsqueeze(1) * pre_ego
            ego_proj_cls_attn = ego_proj_cls_attn.flatten(-2, -1).permute(0, 2, 1) # b*c*(h*w)
            attn_cosine_sim[:, i] = F.cosine_similarity(text_f, ego_proj_cls_attn.mean(dim=1), dim=-1)

        head_idxs = torch.argmax(attn_cosine_sim,dim=1)
        counter = torch.bincount(head_idxs, minlength=ego_attn.size(1))

        ego_sam = 0.5*image_features_attn + 0.5*ego_cls_attn[: ,head_idxs].mean(1)
        ego_sam = normalize_minmax(ego_sam) # batch*16*16
        ego_sam_flat = ego_sam.flatten(-2, -1)


        # --- Affordance CAM generation --- #
        exo_proj = self.aff_exo_proj(exo_proj)
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x 36 x h x w
        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes)
        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w)

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda()
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, aff_label[b_]]

        # --- Clustering extracted descriptors based on CAM --- #
        ego_desc_flat = ego_desc.flatten(-2, -1)  # b x 384 x hw
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1)
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()
        part_score = torch.zeros(b, self.cluster_num).cuda()
        part_proto = torch.zeros(b, c).cuda()
        for b_ in range(b):
            exo_aff_desc = []
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10)
                tmp_desc = exo_desc_re_flat[b_, n]
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T  # n x c
                exo_aff_desc.append(tmp_top_desc)
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)  # (n1 + n2 + n3) x c

            if exo_aff_desc.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())
            clu_cens = F.normalize(kmeans.centroids, dim=1)

            # save the exocentric similarity maps for visualization in training
            for n_ in range(num_exo):
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_, n_], dim=0))

            # find object part prototypes and background prototypes
            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_], dim=0))  # self.cluster_num x hw
            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], \
                                       torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float()
            sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float()

            inter = (sim_map_hard * sam_hard[b_]).sum(1)
            union = sim_map_hard.sum(1) + sam_hard[b_].sum() - inter
            p_score = (inter / sim_map_hard.sum(1) + sam_hard[b_].sum() / union) / 2

            sim_maps[b_] = sim_map
            part_score[b_] = p_score

            if p_score.max() < self.part_iou_thd:
                continue

            part_proto[b_] = clu_cens[torch.argmax(p_score)]

        sim_maps = sim_maps.reshape(b, self.cluster_num, h, w)
        exo_sim_maps = exo_sim_maps.reshape(b, num_exo, self.cluster_num, h, w)
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)
        aff_logits_ego = self.gap(ego_pred).view(b, self.aff_classes)


        # --- contrastive loss --- #

        loss_contra = torch.zeros(1).cuda()
        temperature = 0.5  
        clip_logits = similarity_matrix / temperature

        loss_contra = F.cross_entropy(clip_logits, aff_label)

        # --- concentration loss --- #
        gt_ego_cam = torch.zeros(b, h, w).cuda()
        loss_con = torch.zeros(1).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])

        gt_ego_cam = normalize_minmax(gt_ego_cam)
        loss_con /= b

        # --- prototype guidance loss --- #
        loss_proto = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = gt_ego_cam[b_]
                    tmp_feat = ego_desc[b_] * mask
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum()
                    loss_proto += torch.max(
                        1 - F.cosine_similarity(embedding, part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda())
                    valid_batch += 1
            loss_proto = loss_proto / (valid_batch + 1e-15)

        masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam,'ego_samss': ego_samss,
                 'pred': (sim_maps, exo_sim_maps, part_score, gt_ego_cam)}
        logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return masks, logits, loss_proto, loss_con, loss_contra, counter

    @torch.no_grad()
    def test_forward(self, ego, ego_depth, aff_label, ego_label_name):

        b = aff_label.size(0)
        name_list = [entry[0].replace("_", " ") for entry in ego_label_name]
        template = "a good photo of {}"
        text_list = [template.format(name) for name in name_list]    
        max_length = clip_model.context_length - self.num_prompts  
        tokens = clip.tokenize(text_list, truncate=True, context_length=max_length).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens, self.prompt_embeddings, max_length).to(torch.float32)
        text_features = self.clip_line(text_features)
        ego_list = self.vit_model.My_forward_features(ego, ego_depth.repeat(1, 3, 1, 1))

        ego_desc = ego_list['x_norm_patchtokens']
        
        ego_proj = self.aff_proj(ego_desc)

        ego_desc = self._reshape_transform(ego_desc, self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        pre_ego = ego_proj
        image_features, ego_proj, mu_att  = self.ego_self_attention(ego_proj)

        b, c, h, w = ego_desc.shape

        image_features = F.normalize(image_features, dim=1, p=2)
        text_features = F.normalize(text_features, dim=1, p=2)

        logit_scale = self.logit_scale.exp()
        text_f = torch.ones((b, 384)).cuda()
        for i in range(b):
            text_f[i] = text_features[aff_label[i]]
        
        att_egoproj = F.normalize(ego_proj, dim=1, p=2)
        attego = logit_scale *att_egoproj.permute(1, 0, 2)@text_f.unsqueeze(2)
        attego = torch.sigmoid(F.normalize(attego, dim=1, p=2)).permute(1, 0, 2).repeat(1, 1, c)
        ego_proj = attego.permute(1, 2, 0).view(b, c, h, w)*pre_ego + pre_ego

        ########b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]

        return gt_ego_cam

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result

    def select_above_average_and_pool(self, image_features):

        patch_mean = image_features.mean(dim=(-2, -1), keepdim=True)  
        above_average = image_features > patch_mean  
        selected_features = image_features * above_average.float()  
        selected_mean = selected_features.sum(dim=(-2, -1)) / ( above_average.sum(dim=(-2, -1)) + 1e-6) 
        
        return selected_mean # [batch_size, dim]

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x_1 = x + self.positional_embedding[:, None, :].to(x.dtype)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x[0], x[1:], att[:, :, :]