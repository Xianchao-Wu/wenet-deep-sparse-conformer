#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Multi-Head Attention layer definition."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1-trg, size).
            key (torch.Tensor): Key tensor (#batch, time2-src, size).
            value (torch.Tensor): Value tensor (#batch, time2-src, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        #ipdb.set_trace()
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor,
                          mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        #ipdb.set_trace()
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor = torch.empty(0),) -> torch.Tensor:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.

                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.

                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.

                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        #ipdb.set_trace()
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        #ipdb.set_trace()
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class ProbMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with ProbAttention (AAAI2021 best paper Informer).
    Paper: https://arxiv.org/abs/2012.07436
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, 
                 factor: float=5.0, keep_minlen: float=15.0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.factor = factor
        self.keep_minlen = keep_minlen 
        # NOTE if L_Q or L_K <= keep_minlen, do not do factor * ln(L_Q or L_K)


    def prob_QK(self, Q: torch.Tensor, K: torch.Tensor, 
                sample_k: int, n_top_q: int, 
                index: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Q [B=batch-size, H=head num, L=length, D=dimension]
        _, H, L_K, E = K.shape # K can be pos_emb with B=1, NOTE
        B, _, L_Q, _ = Q.shape
        # TODO if sample_k or n_top_q is not smaller, we can keep everything much 
        # easier!
        if index is None:
            # calculate the sampled Q_K
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
            # sample "sample_k" number of key vectors from K
            #print(L_K, L_Q, sample_k)
            #print(type(L_K), type(L_Q), type(sample_k))

            # NOTE torch.script do not support torch.randint!!!
            ###index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
            ###K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
            K_sample = K_expand

            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

            # find the Top_k(=n_top_q) query with sparisty measurement
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
            M_top = M.topk(n_top_q, sorted=False)[1]

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[torch.arange(B)[:, None, None],
                         torch.arange(H)[None, :, None],
                         M_top, :] # factor*ln(L_q)
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

            # Q_K: [batch, head, q'_len, K.len] attention scores; 
            # M_top: [batch, head, q'_len] index
            return Q_K, M_top
        else:
            # i.e., M_top is given, reuse it to obtain Q_reduce.
            # use the reduced Q to calculate Q_K
            Q_reduce = Q[torch.arange(B)[:, None, None],
                         torch.arange(H)[None, :, None],
                         index, :] # factor*ln(L_q)
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

            # Q_K: [batch, head, q'_len, K.len] attention scores; 
            # M_top: [batch, head, q'_len] index
            return Q_K, index


    def get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        B, H, L_V, D = V.shape
        #if not self.mask_flag:
        #    # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        #else: # use mask
        #    assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
        #    contex = V.cumsum(dim=-2)
        return context # [batch, head, L_Q, dim]


    def update_context(self, context_in: torch.Tensor, V: torch.Tensor, 
                       scores: torch.Tensor, index: torch.Tensor, 
                       L_Q: int, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, H, L_V, D = V.shape

        #if self.mask_flag:
        #    # TODO
        #    attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #    scores.masked_fill_(attn_mask.mask, -np.inf)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).eq(0) # [batch, 1, *, time2]
            # expand from [B,1,1,time2] to [batch,H,time1=L_Q=trg,time2=src.memory]
            attn_mask = attn_mask.expand(B, H, L_Q, attn_mask.shape[-1])

            attn_mask_reduced = attn_mask[torch.arange(B)[:, None, None],
                                          torch.arange(H)[None, :, None],
                                          index, :]
            scores = scores.masked_fill(attn_mask_reduced, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(attn_mask_reduced, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        attn = self.dropout(attn)

        # [batch, head, L_Q, dim]
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        #if self.output_attention:
        #    attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
        #    attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        #    return (context_in, attns)
        #else:
        #    return (context_in, None)
        return context_in


    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor]):
        """Compute 'Scaled Dot Product Attention' with Prob-Attention.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        #ipdb.set_trace()
        # pre prepare
        q, k, v = self.forward_qkv(query, key, value) # out=[batch, head, seq.len, dim]
        B, H, L_Q, D = q.shape
        _, _, L_K, _ = k.shape

        U_part = round(self.factor * math.ceil(math.log(L_K))) # c * ln(L_K)
        u = round(self.factor * math.ceil(math.log(L_Q))) # c * ln(L_Q)
        U_part = int(U_part)
        u = int(u)

        U_part = U_part if U_part < L_K else L_K
        u = u if u <L_Q else L_Q

        U_part = L_K if L_K <= self.keep_minlen else U_part
        u = L_Q if L_Q <= self.keep_minlen else u

        # step 1
        scores_top, index = self.prob_QK(q, k, sample_k=U_part, n_top_q=u, index=None)

        scale = 1./math.sqrt(D)
        scores_top = scores_top * scale

        # step 2
        context = self.get_initial_context(v, L_Q)

        # step 3
        context = self.update_context(context, v, scores_top, index, L_Q, mask)
        # context = [batch, head, L_Q, dim]

        # post process        
        context = context.transpose(1,2).contiguous().view(B, -1, H*D)
        context = self.linear_out(context)

        return context 


class ProbRelPositionMultiHeadedAttention(ProbMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
       together with, Multi-Head Attention layer with ProbAttention (AAAI2021 best paper Informer).
    Paper: https://arxiv.org/abs/2012.07436
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        factor (float) : for replacing L_Q/L_K by factor * ln(L_Q) or factor * ln(L_K)
        keep_minlen (float) : if L_Q or L_K <= keep_minlen, do not use q or k reduce!
    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, 
                 factor: float=5.0, keep_minlen: float=15.0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, factor, keep_minlen)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)


    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        #ipdb.set_trace()
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # after transpose, q=(batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, time1, head, d_k) + (head, d_k) -> transpose(1,2) -> (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, time1, head, d_k) + (head, d_k) -> transpose(1,2) -> (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        
        # new
        B, H, L_Q, D = q_with_bias_u.shape
        _, _, L_K, _ = k.shape

        U_part = round(self.factor * math.ceil(math.log(L_K))) # c * ln(L_K)
        u = round(self.factor * math.ceil(math.log(L_Q))) # c * ln(L_Q)
        U_part = int(U_part)
        u = int(u)

        U_part = U_part if U_part < L_K else L_K
        u = u if u <L_Q else L_Q

        U_part = L_K if L_K <= self.keep_minlen else U_part
        u = L_Q if L_Q <= self.keep_minlen else u

        # step 1, better let index2 = index1 
        scores_top1, index1 = self.prob_QK(q_with_bias_u, k, 
                                           sample_k=U_part, n_top_q=u, index=None)
        scores_top2, _ = self.prob_QK(q_with_bias_v, p, 
                                      sample_k=U_part, n_top_q=u, index=index1)

        scale = 1./math.sqrt(D)
        scores_top = (scores_top1 + scores_top2) * scale

        # step 2
        context = self.get_initial_context(v, L_Q)

        # step 3
        context = self.update_context(context, v, scores_top, index1, L_Q, mask)
        # context = [batch, head, L_Q, dim]

        # post process        
        context = context.transpose(1,2).contiguous().view(B, -1, H*D)
        context = self.linear_out(context)

        return context 

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        ### matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        ### matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        ### scores = (matrix_ac + matrix_bd) / math.sqrt(
        ###    self.d_k)  # (batch, head, time1, time2)

        ### return self.forward_attention(v, scores, mask)

