# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
# from fasterRCNN.get_region_proposal import 
from models.ops.modules import MSDeformAttn

import matplotlib.pyplot as plt
import numpy as np


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, visualize_reference_point=True):
        super().__init__()
        self.visualizer_reference_point = visualize_reference_point
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        # 입력으로 받은 memory의 shape를 가져옵니다. 여기서 N_는 배치 크기, S_는 시퀀스 길이, C_는 채널 수를 의미합니다
        N_, S_, C_ = memory.shape
        # base_scale = 4.0: 기본 스케일을 설정합니다. 이 값은 바운딩 박스의 크기를 조정하는 데 사용됩니다.
        base_scale = 4.0
        proposals = []
        _cur = 0
        # spatial_shapes는 각 피쳐 맵의 높이와 너비를 담고 있습니다. 이를 순회하면서 각 피쳐 맵에 대한 영역 제안을 생성합니다.
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            # 각 피쳐 맵의 그리드를 생성합니다. 이 그리드는 각 피쳐 맵의 픽셀 위치를 나타냅니다.
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # 각 피쳐 맵의 유효한 너비와 높이를 가져와서 스케일을 계산합니다. 이 스케일은 그리드의 좌표를 정규화하는 데 사용됩니다.
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # 각 제안의 너비와 높이를 계산합니다. 이 값은 바운딩 박스의 크기를 결정
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # 그리드의 좌표와 바운딩 박스의 크기를 결합하여 제안을 생성합니다. 각 제안은 4개의 좌표를 가지며, 이는 바운딩 박스의 좌측 상단과 우측 하단 좌표를 의미합니다.
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            # torch.Size([1, 10880, 4]) -> 바운딩박스 xywh의 정보가 각 피처맵의 크기만큼 존재함 
            proposals.append(proposal)
            _cur += (H_ * W_)
        # 모든 제안을 결합하여 최종 제안을 생성합니다.
        output_proposals = torch.cat(proposals, 1)
        # 제안의 유효성을 검사합니다. 제안의 좌표가 [0.01, 0.99] 범위 내에 있는 경우에만 유효한 제안으로 간주합니다
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # 제안의 좌표를 로그-오즈 변환합니다. 이는 바운딩 박스의 좌표를 [0, 1]이 함수는 인코더의 출력을 기반으로 영역 제안(Region Proposals)을 생성하는 역할을 합니다.
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 메모리 패딩 마스크를 사용하여 제안의 마스킹을 수행합니다. 
        # 메모리 패딩 마스크는 입력 이미지의 유효한 부분을 나타냅니다. 마스크가 True인 부분, 즉 유효하지 않은 부분은 제안에서 무한대(inf)로 채워집니다.
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 유효하지 않은 제안을 마스킹합니다. 유효하지 않은 제안은 좌표가 [0.01, 0.99] 범위를 벗어나는 제안을 의미합니다. 이러한 제안은 무한대(inf)로 채워집니다.
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        # 메모리를 복사하여 output_memory를 생성합니다.
        
        # 포지셔널인코딩인듯?
        output_memory = memory
        # 메모리 패딩 마스크를 사용하여 output_memory의 마스킹을 수행합니다. 마스크가 True인 부분은 output_memory에서 0으로 채워집니다
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        # 유효하지 않은 제안에 해당하는 부분을 output_memory에서 마스킹합니다. 이 부분은 0으로 채워집니다.
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        # output_memory를 정규화합니다. 이는 네트워크의 학습을 안정화하고 빠르게 만드는 데 도움이 됩니다.
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    # def gen_encoder_output_proposals_2(self, memory, memory_padding_mask, spatial_shapes, frcnn_boxes):
    #     # 입력으로 받은 memory의 shape를 가져옵니다. 여기서 N_는 배치 크기, S_는 시퀀스 길이, C_는 채널 수를 의미합니다
    #     N_, S_, C_ = memory.shape
    #     # base_scale = 4.0: 기본 스케일을 설정합니다. 이 값은 바운딩 박스의 크기를 조정하는 데 사용됩니다.
    #     base_scale = 4.0
    #     proposals = []
    #     # memory : torch.Size([1, 14512, 256])
    #     # memory_padding_mask shape:torch.Size([1, 14512])
    #     # spatial_shapes : ([4,2])
    #     _cur = 0
    #     # spatial_shapes는 각 피쳐 맵의 높이와 너비를 담고 있습니다. 이를 순회하면서 각 피쳐 맵에 대한 영역 제안을 생성합니다.
    #     for lvl, (H_, W_) in enumerate(spatial_shapes):
    #         mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
    #         valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
    #         valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
    #         # 각 피쳐 맵의 그리드를 생성합니다. 이 그리드는 각 피쳐 맵의 픽셀 위치를 나타냅니다.
    #         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
    #                                         torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
    #         grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
    #         # 각 피쳐 맵의 유효한 너비와 높이를 가져와서 스케일을 계산합니다. 이 스케일은 그리드의 좌표를 정규화하는 데 사용됩니다.
    #         scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
    #         grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
    #         # 각 제안의 너비와 높이를 계산합니다. 이 값은 바운딩 박스의 크기를 결정
    #         wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
    #         # 그리드의 좌표와 바운딩 박스의 크기를 결합하여 제안을 생성합니다. 각 제안은 4개의 좌표를 가지며, 이는 바운딩 박스의 좌측 상단과 우측 하단 좌표를 의미합니다.
    #         proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
    #         # torch.Size([1, 10880, 4]) -> 바운딩박스 xywh의 정보가 각 피처맵의 크기만큼 존재함 
    #         # 레퍼런스포인트 리스트 생성
    #         proposals.append(proposal)
    #         _cur += (H_ * W_)
    #     # 모든 제안을 결합하여 최종 제안을 생성합니다.
    #     output_proposals = torch.cat(proposals, 1)
    #     # 제안의 유효성을 검사합니다. 제안의 좌표가 [0.01, 0.99] 범위 내에 있는 경우에만 유효한 제안으로 간주합니다
    #     output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    #     # 제안의 좌표를 로그-오즈 변환합니다. 이는 바운딩 박스의 좌표를 [0, 1]이 함수는 인코더의 출력을 기반으로 영역 제안(Region Proposals)을 생성하는 역할을 합니다.
    #     output_proposals = torch.log(output_proposals / (1 - output_proposals))
    #     # 메모리 패딩 마스크를 사용하여 제안의 마스킹을 수행합니다. 
    #     # 메모리 패딩 마스크는 입력 이미지의 유효한 부분을 나타냅니다. 마스크가 True인 부분, 즉 유효하지 않은 부분은 제안에서 무한대(inf)로 채워집니다.
    #     output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    #     # 유효하지 않은 제안을 마스킹합니다. 유효하지 않은 제안은 좌표가 [0.01, 0.99] 범위를 벗어나는 제안을 의미합니다. 이러한 제안은 무한대(inf)로 채워집니다.
    #     output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    #     # 메모리를 복사하여 output_memory를 생성합니다.
        
    #     # 포지셔널인코딩인듯?
    #     output_memory = memory
    #     # 메모리 패딩 마스크를 사용하여 output_memory의 마스킹을 수행합니다. 마스크가 True인 부분은 output_memory에서 0으로 채워집니다
    #     output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    #     # 유효하지 않은 제안에 해당하는 부분을 ㅇ 마스킹합니다. 이 부분은 0으로 채워집니다.
    #     output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    #     # output_memory를 정규화합니다. 이는 네트워크의 학습을 안정화하고 빠르게 만드는 데 도움이 됩니다.
    #     output_memory = self.enc_output_norm(self.enc_output(output_memory))
    #     return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1) # mask_flatten.shape torch.Size([1, 14512])
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder torch.Size([1, 14512, 256])
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, self.visualizer_reference_point)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage :
            # 인코더의 출력과 영역제안
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # hack implementation for two-stage Deformable DETR
            # 각각 클래스 예측과 바운딩 박스 예측을 위한 인코더의 출력
            # enc_outputs_class의 모양확인할 것
            # 이 코드는 영역 후보군들을 다시 인코더에 넣어 예측하는 것으로 변환하는 것 같음
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory) # torch.Size([1, 14512, 91])
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals #torch.Size([1, 14512, 4])
            topk = self.two_stage_num_proposals # 300개의 후보영역 추출
            # topk_proposals는 클래스 점수가 가장 높은 상위 k개의 제안을 선택합니다. 이는 첫 번째 단계에서 생성된 제안 중에서 가장 높은 점수를 가진 제안만을 선택하여 두 번째 단계로 전달합니다.
            # topk의 개수만큼 높은 점수대로 자름
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # topk_coords_unact는 선택된 제안의 좌표를 가져옵니다.
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            # reference_points는 제안의 좌표를 [0, 1] 범위로 정규화합니다. 이는 두 번째 단계에서 사용될 reference point를 생성하는 과정
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # 마지막으로, query_embed와 tgt는 positional encoding을 통해 생성된 query embedding과 target을 분리하는 과정입니다. 이는 두 번째 단계에서 사용될 query와 target을 준비하는 과정입니다.
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)
        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact, topk_coords_unact
            # return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, sampling_locations = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src, sampling_locations


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    # 샘플링 포인트를 시각화 하는 코드
    def visualize_sampling_points(self, reference_points, sampling_locations, layer, spatial_shapes) :
        rp = reference_points.detach().cpu().numpy()
        sp = sampling_locations.detach().cpu().numpy()
        # 가정: Sampling point와 Reference point는 torch.Tensor 형태이고 numpy array로 변환 필요
        for l in range(4):  # level의 수에 따라 조절해야 합니다.
            # fig, ax = plt.subplots(figsize=(6, 6))
            # for h in range(8):  # head의 수에 따라 조절해야 합니다.
            if l == 3 :
                w = spatial_shapes[l][0]
                h = spatial_shapes[l][1]
                sp = sp[:, -w*h:]
                fig, ax = plt.subplots(figsize=(10, 10))
                for k in range(4):  # k의 수에 따라 조절해야 합니다.
                    # 각 level에 대해 reference point에 sampling point를 더하여 실제 위치를 계산합니다.
                    x = sp[0, :, 0, l, k, 0]
                    y = sp[0, :, 0, l, k, 1]
                    print(x.shape)
                    y = y.max() - y
                    ax.scatter(x, y, s=1, color='skyblue')
                    print(f"scatter h={0}, l={l}, k={k}")

                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(y.min(), y.max())
                # ax.legend()

                plt.title(f'Layer_{layer}/lvl_{l}')
                plt.savefig(f"/root/Deformable-DETR/result_sampling/Layer_{layer}_lvl_{l}.jpg")
                print(f"Encdoer Layer {layer}/lvl_{l}'s sampling point save")
                plt.close()
            
        fig, ax = plt.subplots(figsize=(10, 10))
        batch, points, lvl, x_y_ = reference_points.shape
        for l in range(lvl) :
            if l == 3 :
                w = spatial_shapes[l][0]
                h = spatial_shapes[l][1]
                rp = rp[:, -w*h:]
                x = rp[0, :, l, 0]
                y = rp[0, :, l, 1]
                ax.scatter(x, y, s=1, color='skyblue')
                plt.title(f'reference_point_lvl_{l}')
                plt.savefig(f"/root/Deformable-DETR/result_sampling/reference_point_lvl_{l}.jpg")
                plt.close()
                
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None, visualize_rf_point=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output, sampling_locations = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            # 샘플링 포인트 시각화
            if visualize_rf_point :
                self.visualize_sampling_points(reference_points, sampling_locations, _, spatial_shapes)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2, sp = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries, 
        visualize_reference_point=args.visualize_reference_point)


