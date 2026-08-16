import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gpytorch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from quantile_rtg import QuantileRTGHead

class DecisionTransformer(nn.Module):
    def __init__(self, config, input_dim, action_dim, use_mf=False):
        super(DecisionTransformer, self).__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = config.hidden_size
        self.max_seq_length = config.max_seq_length

        # Embeddings
        self.state_embedding = nn.Linear(input_dim, config.hidden_size)
        self.action_embedding = nn.Linear(action_dim, config.hidden_size)
        self.reward_embedding = nn.Linear(1, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_size,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, action_dim)
        )

        # Optional quantile-RTG head (Step 5). Disabled by default -- when False,
        # forward() behaves exactly as before (no causal mask, no quantile output).
        self.quantile_head = QuantileRTGHead(config.hidden_size, M=5)
        self.use_quantile_rtg = False

        # Optional MF-DRO support (Step 6/Prompt 5). Disabled by default -- when
        # False, none of these attributes exist and forward()/get_action_hidden_states
        # behave exactly as before; only forward_mf()/propose_mf() consume them.
        self.use_mf = use_mf
        if use_mf:
            self.btg_embed = nn.Linear(1, config.hidden_size)

            # Mirrors action_head: 2 Linear + ReLU (+ Sigmoid, since this head
            # predicts a fidelity probability, not a raw location).
            self.fidelity_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid()
            )

            # Separate action embedding for MF (accepts d+1 input: location + fidelity)
            self.action_embed_mf = nn.Linear(action_dim + 1, config.hidden_size)

            self.lambda_fid = 1.0
            self._fid_mean_history = []

    def get_action_hidden_states(self, states, actions, rewards, timesteps, attention_mask=None):
        """
        Shared embedding + transformer pass. Returns the transformer's hidden
        state at each action-token position, [batch, seq_len, hidden_size].
        Used both by forward() (for action prediction) and, when
        use_quantile_rtg=True, by the quantile head (training via the
        one-step-shift trick in _train_decision_transformer, inference directly).
        """
        batch_size, seq_length = states.shape[0], states.shape[1]

        # Create embeddings for states, actions, rewards
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        reward_embeddings = self.reward_embedding(rewards.unsqueeze(-1))

        # Create position embeddings
        position_embeddings = self.position_embedding(timesteps)

        # Combine embeddings for sequence input, interleaved per timestep as
        # [r_0, s_0, a_0, r_1, s_1, a_1, ...] -- RTG *first* within each triple.
        # This ordering matters specifically for the causal-masked case (see
        # below): with the previous [s_0, a_0, r_0, ...] ordering, the action
        # token at position 3t+1 came *before* its own timestep's RTG token at
        # 3t+2, so a causal mask (which only allows attending to earlier
        # positions) made the RTG token completely invisible to the action
        # head -- confirmed directly: feeding target_rtg values from -5 to +50
        # into a causal-masked model produced the exact same predicted action
        # every time, while the same sweep on a non-causal-masked model (any
        # non-quantile schema) produced clearly different actions. Putting RTG
        # first means a_t at 3t+2 can always attend to r_t at 3t and s_t at
        # 3t+1, with or without a causal mask.
        #
        # Stacking on dim=2 (not dim=1) is what makes the .reshape(...,
        # 3*seq_length, ...) below actually produce per-timestep interleaving;
        # stacking on dim=1 groups by modality instead ([r_0..r_L, s_0..s_L,
        # a_0..a_L]), which would desync this from both the position-embedding
        # repeat_interleave below and the `[:, 2::3]` action-token extraction
        # further down (both assume per-timestep interleaving).
        sequence = torch.stack([reward_embeddings, state_embeddings, action_embeddings], dim=2)
        sequence = sequence.reshape(batch_size, 3 * seq_length, self.hidden_size)
        sequence = sequence + position_embeddings.repeat_interleave(3, dim=1)

        # Causal mask: only applied when use_quantile_rtg=True. With RTG first
        # in each triple (above), this still gives the quantile head's "predict
        # R_tau from h^a_{tau-1}, not h^R_tau" design real leakage protection:
        # h^a_{tau-1} sits at position 3(tau-1)+2 = 3*tau-1, strictly before
        # r_tau at position 3*tau, so it structurally cannot attend to it.
        # Without this mask, full bidirectional self-attention would let
        # h^a_{tau-1} attend to r_tau (or any future token) regardless of
        # ordering. Gated off by default so the existing (non-quantile)
        # action-prediction path is unaffected: full bidirectional attention,
        # exactly as before (the reordering itself is a no-op there, since
        # unmasked self-attention depends only on the set of (content,
        # position-embedding) pairs attended to, not their array layout).
        causal_mask = None
        if self.use_quantile_rtg:
            seq_len_full = 3 * seq_length
            causal_mask = torch.triu(
                torch.ones(seq_len_full, seq_len_full, dtype=torch.bool, device=sequence.device),
                diagonal=1,
            )

        # Apply transformer
        if attention_mask is not None:
            # Repeat mask for state, action, reward triplets
            key_padding_mask = ~attention_mask.repeat_interleave(3, dim=1)
            transformer_outputs = self.transformer(sequence, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            transformer_outputs = self.transformer(sequence, mask=causal_mask)

        # Extract action hidden states (action token is now the 3rd of each
        # triple: r_t, s_t, a_t at offsets 0, 1, 2 -- so a_t is at index 3t+2).
        return transformer_outputs[:, 2::3]

    def forward(self, states, actions, rewards, timesteps, attention_mask=None, return_quantiles=False):
        h_action = self.get_action_hidden_states(states, actions, rewards, timesteps, attention_mask)

        # Predict next actions
        predicted_actions = self.action_head(h_action)

        if not return_quantiles:
            return predicted_actions

        # Shift by one action-token position so Q_hat[:, tau] is predicted from
        # h^a_{tau-1} (full history up to tau-1), never from h^R_tau itself.
        zero_pad = torch.zeros(h_action.shape[0], 1, h_action.shape[-1], device=h_action.device, dtype=h_action.dtype)
        h_shifted = torch.cat([zero_pad, h_action[:, :-1, :]], dim=1)
        Q_hat = self.quantile_head(h_shifted)
        return predicted_actions, Q_hat

    def forward_mf(self, states, actions_x, actions_ell,
                    rtg, btg, timesteps, attention_mask=None):
        """
        MF-DRO forward pass. Deliberately independent of forward()/
        get_action_hidden_states -- replicates the embedding logic fresh
        rather than calling into them, since the MF sequence has 4 tokens
        per step ([rtg, btg, state, action]) instead of SF's 3
        ([reward, state, action]).

        states:       [B, T, state_dim]
        actions_x:    [B, T, d]
        actions_ell:  [B, T]        int64 {0,1}
        rtg:          [B, T]        float return-to-go
        btg:          [B, T]        float budget-to-go
        timesteps:    [B, T]        int64 (same as SF)
        attention_mask: [B, 4*T] or None (already at token granularity,
            unlike SF's [B,T] mask which get_action_hidden_states itself
            repeat_interleaves by 3 -- here the caller supplies it already
            expanded to 4*T, so it's passed straight through).

        Sequence per step (4 tokens): [rtg, btg, state, action]
        Interleaving: [rtg_0, btg_0, s_0, a_0, rtg_1, btg_1, ...]
        Action is LAST in each group (can attend to rtg and btg).
        Total tokens: 4*T.

        Returns: loss, L_loc, L_fid, x_pred [B,T,d], p_pred [B,T]
        """
        B, T, _ = states.shape
        H = self.hidden_size

        # Embed each token type
        rtg_emb = self.reward_embedding(rtg.unsqueeze(-1))   # [B,T,H]
        btg_emb = self.btg_embed(btg.unsqueeze(-1))          # [B,T,H]
        s_emb = self.state_embedding(states)                 # [B,T,H]

        # Action embedding: concat [x; ell.float()]
        act_inp = torch.cat([
            actions_x,
            actions_ell.float().unsqueeze(-1)
        ], dim=-1)                                            # [B,T,d+1]
        a_emb = self.action_embed_mf(act_inp)                 # [B,T,H]

        # Position embeddings -- 4 tokens per step (NOT 3, unlike SF)
        pos_emb = self.position_embedding(timesteps) \
            .repeat_interleave(4, dim=1)                      # [B,4T,H]

        # Interleave: [rtg, btg, state, action] per step -> [B,4T,H]
        seq = torch.stack(
            [rtg_emb, btg_emb, s_emb, a_emb], dim=2
        ).reshape(B, 4 * T, H)                                # [B,4T,H]
        seq = seq + pos_emb

        # Pass through transformer (same encoder stack as SF, no causal mask)
        if attention_mask is not None:
            h = self.transformer(seq, src_key_padding_mask=~attention_mask)
        else:
            h = self.transformer(seq)                          # [B,4T,H]

        # Action hidden states: positions 3, 7, 11, ... (3 + 4k)
        h_act = h[:, 3::4, :]                                  # [B,T,H]

        # Location head (existing action_head)
        x_pred = self.action_head(h_act)                       # [B,T,d]

        # Fidelity head (new)
        p_pred = self.fidelity_head(h_act).squeeze(-1)         # [B,T]

        # Losses
        L_loc = F.mse_loss(x_pred, actions_x)
        L_fid = F.binary_cross_entropy(
            p_pred, actions_ell.float()
        )
        loss = L_loc + self.lambda_fid * L_fid

        # Monitor fidelity head health
        fid_mean = p_pred.detach().mean().item()
        self._fid_mean_history.append(fid_mean)
        if len(self._fid_mean_history) >= 10:
            recent = self._fid_mean_history[-10:]
            if all(v < 0.05 for v in recent) or \
               all(v > 0.95 for v in recent):
                print(f"WARNING: fidelity head collapsed "
                      f"(mean={fid_mean:.3f}). "
                      "Consider increasing lambda_fid.")

        return loss, L_loc, L_fid, x_pred, p_pred

    def propose_mf(self, state, rtg_target, btg_target, timestep=0):
        """
        Single-step MF inference.
        state:      [state_dim] tensor
        Returns:    x_t [d], ell_t int {0,1}
        """
        self.eval()
        with torch.no_grad():
            s = state.unsqueeze(0).unsqueeze(0)  # [1,1,state_dim]
            r = torch.tensor([[[rtg_target]]], dtype=state.dtype)
            b = torch.tensor([[[btg_target]]], dtype=state.dtype)
            ax = torch.zeros(1, 1, self.action_dim, dtype=state.dtype)
            ae = torch.zeros(1, 1, dtype=torch.long)
            ts = torch.tensor([[timestep]], dtype=torch.long)

            _, _, _, x_pred, p_pred = self.forward_mf(
                s, ax, ae,
                r.squeeze(-1), b.squeeze(-1),
                ts, attention_mask=None
            )
            x_t = x_pred[0, 0].clamp(0.0, 1.0)
            p_val = p_pred[0, 0].item()
            ell_t = 1 if p_val > 0.5 else 0
        self.train()
        self.last_p_pred = p_val  # exposed for diagnostics/verification
        return x_t, ell_t
