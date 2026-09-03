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
        # H102: which loss trains the location head. 'mse' reproduces every
        # prior run bit-for-bit.
        self.loc_loss = str(getattr(config, 'loc_loss', 'mse')).lower()
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
            # H179 (default OFF -> every existing config is bit-identical).
            # h177/h178 measured that raw scalars into Linear(1->H)+LayerNorm
            # SATURATE: over BTG's operating range (~26-30) the trained
            # embedding response is 0.0056, 93x less than RTG's over its own
            # (0.30-1.00), and it stays there through training. Z-scoring the
            # scalar restores the response to 1.88 (336x). These buffers hold a
            # running mean/std used only when standardize_conditioning is set.
            self.standardize_conditioning = bool(
                getattr(config, 'standardize_conditioning', False))
            self.register_buffer('_cond_mu', torch.zeros(2))
            self.register_buffer('_cond_sd', torch.ones(2))
            self.register_buffer('_cond_n', torch.zeros(1))

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

            # Candidate scoring head (used when use_candidate_scoring=True,
            # see forward_mf/propose_mf) -- scores one [hidden_state;
            # candidate_features] pair at a time, batched over K candidates.
            # Does NOT replace action_head, which stays available as the
            # regression path.
            #
            # cand_feature_dim (Change 1c): the per-candidate block is
            # [x_norm(action_dim), mu_H, sigma_H, mu_L, sigma_L, MES_H/c_H,
            # MES_L/c_L, ||x - x_incumbent||] = action_dim + 7 when
            # mf_dro.build_candidate_features supplies it, or bare
            # action_dim coordinates when candidate features are ablated
            # off (config.use_candidate_features=False). Passed in by the
            # caller so this module doesn't import mf_dro (which imports
            # this one -- would be circular).
            self.cand_feature_dim = getattr(config, 'cand_feature_dim', self.action_dim)
            self.score_head = nn.Sequential(
                nn.Linear(config.hidden_size + self.cand_feature_dim, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
            # ITEM 5: state-conditioned LINEAR coefficient score head (behind
            # a flag, default ON). Replaces concat-then-MLP with
            # score = (coef_head(h) * cand).sum(-1) + bias_head(h) -- the
            # state can only act through an explicit per-feature linear
            # weight and a scalar bias, instead of an opaque MLP that could
            # implement (or fail to implement) arbitrary state-dependence
            # invisibly. Reports directly whether the state matters: if
            # coef_head(h) barely varies across the batch, the method HAS
            # reduced to one fixed linear acquisition function regardless of
            # state -- exactly the failure mode found for the old score_head
            # (Probe A/B: argmax pinned to argmax(mu_H) under state
            # perturbation up to 1x batch std). score_head above is kept
            # constructed regardless, so use_linear_score_head=False still
            # works as a direct ablation against it.
            self.use_linear_score_head = getattr(config, 'use_linear_score_head', True)
            self.coef_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, self.cand_feature_dim)
            )
            self.bias_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )

            self.lambda_fid = 1.0
            # Softmax temperature for the candidate-scoring soft target
            # (Change 1d) -- teacher = softmax(teacher_scores / score_temp).
            # Configurable via config.score_temp; 1.0 keeps the teacher's own
            # MES scale.
            self.score_temp = getattr(config, 'score_temp', 1.0)
            self._fid_mean_history = []

            # FIX 2 (post-hoc causal-mask investigation): LayerNorm after
            # each MF embedding, matching standard GPT-style transformer
            # input normalization (DT paper Algorithm 1 / Appendix A).
            # H4 (RTG attention under-allocation). rtg_conditioning:
            #   "token" -- current: 4 tokens [rtg, btg, state, action].
            #   "adaln" -- 2 tokens [state, action]; rtg/btg instead produce
            #              per-feature scale/shift on the state hidden state.
            # Motivation is literature-grounded, not a guess: RADT
            # (arXiv:2402.03923) diagnoses DT's failure to align actual with
            # target return as UNDER-ALLOCATION OF ATTENTION to the RTG
            # tokens, and argues the fix must be structural rather than
            # parametric (i.e. temperature/loss-weight tuning cannot fix it).
            # DDT (arXiv:2601.15953) supplies the mechanism used here:
            # drop RTG from the input and condition via AdaLN-Zero.
            # Our layout is the worse case -- TWO of four tokens are scalar
            # conditioning signals competing for attention, vs one of three
            # in standard DT.
            self.rtg_conditioning = getattr(config, 'rtg_conditioning', 'token')
            # Linear(2 -> 2H), no activation, matching DDT's "single linear
            # layer ... without activation functions". AdaLN-ZERO init: zero
            # weights, bias = [1]*H ++ [0]*H, so gamma=1 / beta=0 at step 0 and
            # the run STARTS equivalent to plain LayerNorm -- the conditioning
            # effect is learned, never imposed by initialization.
            self.adaln_mod = nn.Linear(2, 2 * config.hidden_size)
            nn.init.zeros_(self.adaln_mod.weight)
            with torch.no_grad():
                self.adaln_mod.bias[:config.hidden_size].fill_(1.0)
                self.adaln_mod.bias[config.hidden_size:].fill_(0.0)
            self.adaln_ln = nn.LayerNorm(config.hidden_size, elementwise_affine=False)

            self.state_ln = nn.LayerNorm(config.hidden_size)
            self.action_ln = nn.LayerNorm(config.hidden_size)
            self.reward_ln = nn.LayerNorm(config.hidden_size)
            self.btg_ln = nn.LayerNorm(config.hidden_size)

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

        # Causal mask: ALWAYS applied (previously gated behind
        # use_quantile_rtg=True only). With RTG first in each triple (above),
        # a_t at position 3t+2 can always attend to r_t at 3t and s_t at
        # 3t+1 (both strictly earlier), so masking does not hide a token's
        # own RTG from it -- that failure mode only existed under the OLD
        # [s,a,r] ordering this class no longer uses (see the ordering
        # comment above: with r LAST in a triple, a causal mask made RTG
        # completely invisible to the action head, confirmed empirically by
        # a target_rtg sweep producing the identical predicted action
        # throughout). What changes by turning this on for the *default*
        # (non-quantile) path is that a_t can no longer attend to FUTURE
        # timesteps' tokens (r_{t+1..}, s_{t+1..}, a_{t+1..}) under full
        # bidirectional attention -- previously always on for this path.
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
        # TARGET-LEAKAGE FIX (SF path, same bug as forward_mf's): layout is
        # [r, s, a] per step, so index 2 was the ACTION token (self-visible
        # under the causal mask). Index 1 is the STATE token.
        return transformer_outputs[:, 1::3]

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

    def _std_cond(self, rtg, btg, update=False):
        """H179. Z-score the conditioning scalars using running statistics.

        h177/h178 measured that raw scalars into Linear(1->H)+LayerNorm
        SATURATE: over BTG's operating range (~26-30) the trained embedding
        response is 0.0056 against RTG's 0.5216 over its own (0.30-1.00), a 93x
        gap that persists through training. Z-scoring restores it to ~1.88.

        Returns (rtg, btg) unchanged when standardize_conditioning is False, so
        the default path is bit-identical.
        """
        if not getattr(self, 'standardize_conditioning', False):
            return rtg, btg
        if update and rtg.numel() > 0:
            with torch.no_grad():
                _b = torch.stack([rtg.reshape(-1).mean(), btg.reshape(-1).mean()])
                _s = torch.stack([rtg.reshape(-1).std().clamp_min(1e-6),
                                   btg.reshape(-1).std().clamp_min(1e-6)])
                _m = 0.05 if float(self._cond_n) > 0 else 1.0   # first batch seeds it
                self._cond_mu.mul_(1 - _m).add_(_m * _b.to(self._cond_mu.dtype))
                self._cond_sd.mul_(1 - _m).add_(_m * _s.to(self._cond_sd.dtype))
                self._cond_n += 1
        mu = self._cond_mu.to(rtg.dtype); sd = self._cond_sd.to(rtg.dtype).clamp_min(1e-6)
        return (rtg - mu[0]) / sd[0], (btg - mu[1]) / sd[1]

    def forward_mf(self, states, actions_ell,
                    rtg, btg, timesteps, attention_mask=None,
                    actions_x=None,
                    candidates=None, chosen_idx=None,
                    valid_mask=None, use_candidate_scoring=False,
                    teacher_scores=None, has_soft=None,
                    return_loss_breakdown=False):
        """
        MF-DRO forward pass. Deliberately independent of forward()/
        get_action_hidden_states -- replicates the embedding logic fresh
        rather than calling into them, since the MF sequence has 4 tokens
        per step ([rtg, btg, state, action]) instead of SF's 3
        ([reward, state, action]).

        states:       [B, T, state_dim]
        actions_ell:  [B, T]        int64 {0,1}
        rtg:          [B, T]        float return-to-go
        btg:          [B, T]        float budget-to-go
        timesteps:    [B, T]        int64 (same as SF)
        attention_mask: [B, 4*T] or None (already at token granularity,
            unlike SF's [B,T] mask which get_action_hidden_states itself
            repeat_interleaves by 3 -- here the caller supplies it already
            expanded to 4*T, so it's passed straight through).

        use_candidate_scoring=False (default): ORIGINAL regression path,
        bit-for-bit unchanged from before this flag existed.
            actions_x: [B, T, d], REQUIRED in this mode -- also doubles as
                the action-embedding input (the sequence still needs a
                real filled action token, same as always).
        use_candidate_scoring=True: NEW path. score_head scores each of K
        candidate locations per step; L_loc is cross-entropy over which
        candidate matches the teacher's actual choice, not MSE regression.
            candidates: [B, T, K, d], chosen_idx: [B, T] int64, both
                REQUIRED in this mode. actions_x is NOT required (_train_dt
                doesn't build one in this mode) -- the action-embedding
                token instead gathers the chosen candidate's own location
                (candidates[b,t,chosen_idx[b,t]]), which is "what was
                actually taken" under this mode, exactly analogous to what
                actions_x represents in regression mode.

        valid_mask: [B, T] bool tensor, or None. True = real (non-padded)
            timestep. When given, L_loc/L_fid are averaged over valid
            positions only (padded positions -- e.g. trailing zero-filled
            steps in a BES-shortened rollout -- contribute zero to the
            numerator and are excluded from the denominator, instead of
            silently diluting the loss toward whatever x_pred/p_pred
            predict for a zero-state input). When None, all T positions
            are treated as valid (identical to pre-valid_mask behavior).
            NOTE: valid_mask affects the LOSS only -- padded tokens are
            still embedded and attended over by the transformer below
            (attention_mask is the mechanism for excluding tokens from
            attention, and is a separate, unthreaded concern from this
            change).

        Sequence per step (4 tokens): [rtg, btg, state, action]
        Interleaving: [rtg_0, btg_0, s_0, a_0, rtg_1, btg_1, ...]
        Action is LAST in each group (can attend to rtg and btg).
        Total tokens: 4*T.

        Returns: loss, L_loc, L_fid, x_pred [B,T,d] or None (scoring mode), p_pred [B,T]
        """
        B, T, _ = states.shape
        H = self.hidden_size

        # Embed each token type (FIX 2: LayerNorm after each embedding)
        # H179: standardise the conditioning scalars before embedding.
        # OFF by default -- when off, rtg/btg pass through untouched and every
        # existing configuration is bit-identical.
        rtg, btg = self._std_cond(rtg, btg, update=self.training)
        rtg_emb = self.reward_ln(self.reward_embedding(rtg.unsqueeze(-1)))   # [B,T,H]
        btg_emb = self.btg_ln(self.btg_embed(btg.unsqueeze(-1)))             # [B,T,H]
        s_emb = self.state_ln(self.state_embedding(states))                  # [B,T,H]

        # Action embedding: concat [x; ell.float()]. In candidate-scoring
        # mode actions_x is None (_train_dt doesn't build one -- see its own
        # docstring) but the action token still needs a REAL location to
        # embed, not a missing/placeholder one: gather it from the
        # candidate the teacher actually chose (candidates[b,t,chosen_idx
        # [b,t]]), which is exactly "the location this step's action_embed_mf
        # token should represent" under either mode.
        if use_candidate_scoring:
            idx_exp = chosen_idx.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, candidates.shape[-1])
            # [..., :action_dim] -- candidates carry the d normalized
            # coordinates FIRST, then Change 1c's per-candidate GP/MES
            # feature columns. The action token embeds a LOCATION, so only
            # the coordinate columns belong here (action_embed_mf's input
            # width is action_dim+1, unchanged by the feature block).
            embed_x = candidates.gather(2, idx_exp).squeeze(2)[..., :self.action_dim]
        else:
            embed_x = actions_x
        act_inp = torch.cat([
            embed_x,
            actions_ell.float().unsqueeze(-1)
        ], dim=-1)                                            # [B,T,d+1]
        a_emb = self.action_ln(self.action_embed_mf(act_inp))  # [B,T,H] (FIX 2)

        # H4: tokens-per-step depends on rtg_conditioning (4 vs 2).
        _adaln = (self.rtg_conditioning == 'adaln')
        _tps = 2 if _adaln else 4
        pos_emb = self.position_embedding(timesteps) \
            .repeat_interleave(_tps, dim=1)                   # [B,_tps*T,H]

        if _adaln:
            # [state, action] only -- rtg/btg are removed from the sequence
            # entirely and re-enter via AdaLN on the readout below.
            seq = torch.stack([s_emb, a_emb], dim=2).reshape(B, 2 * T, H)
        else:
            # Interleave: [rtg, btg, state, action] per step -> [B,4T,H]
            seq = torch.stack(
                [rtg_emb, btg_emb, s_emb, a_emb], dim=2
            ).reshape(B, 4 * T, H)                            # [B,4T,H]
        seq = seq + pos_emb

        # FIX 1: causal mask over the full 4T training sequence. RTG is
        # FIRST within each 4-token group (rtg,btg,s,a), so a_t at 4t+3 can
        # always attend to rtg_t/btg_t/s_t at 4t/4t+1/4t+2 (all strictly
        # earlier) under this mask -- matches the RTG-first-ordering lesson
        # already established for the SF path above (get_action_hidden_
        # states' docstring): a causal mask only breaks RTG visibility if
        # RTG is ordered AFTER the action token it corresponds to, which it
        # is not here. This does NOT change propose_mf's single-timestep
        # inference call (nothing there to mask -- see propose_mf's own
        # docstring) -- it only changes what TRAINING (this multi-timestep
        # pass) can attend to, preventing a_t from also seeing rtg_{t+1..T}
        # (future timesteps' targets) during training.
        seq_len = _tps * T
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq.device),
            diagonal=1,
        )

        # Pass through transformer (causally masked, FIX 1)
        if attention_mask is not None:
            h = self.transformer(seq, mask=causal_mask, src_key_padding_mask=~attention_mask)
        else:
            h = self.transformer(seq, mask=causal_mask)         # [B,4T,H]

        # Action hidden states: positions 3, 7, 11, ... (3 + 4k)
        # TARGET-LEAKAGE FIX: read out from the STATE token (index 2 of the
        # 4-token [rtg, btg, s, a] group), NOT the action token (index 3).
        # The causal mask is triu(diagonal=1), so position i attends to
        # itself -- meaning the action token at 4t+3 could see its OWN
        # a_emb, which embeds the TRUE actions_x/actions_ell. Every head
        # reading from that position was therefore predicting its own
        # target. Verified from the mask directly: row 3 is blocked from
        # nothing (sees itself), while row 2 IS blocked from position 3 --
        # so the state token cannot see the action token, the zero-action
        # placeholder propose_mf uses at inference is inert, and train/
        # inference no longer differ in their most predictive input.
        # Matches the DT paper (predict a_t from the state token).
        # State token is index 2 of 4 under token conditioning, index 0 of 2
        # under adaln (where only [state, action] remain).
        h_act = h[:, (0 if _adaln else 2)::_tps, :]            # [B,T,H]
        if _adaln:
            # AdaLN-Zero: gamma,beta = Linear([rtg, btg]); applied to the
            # (affine-free) LayerNorm of the state hidden state.
            _cond = torch.stack([rtg, btg], dim=-1)            # [B,T,2]
            _gb = self.adaln_mod(_cond)                        # [B,T,2H]
            _gamma, _beta = _gb[..., :H], _gb[..., H:]
            h_act = _gamma * self.adaln_ln(h_act) + _beta

        # Fidelity head -- IDENTICAL in both modes (fidelity_head's own last
        # layer is already nn.Sigmoid(), so no extra torch.sigmoid() here).
        p_pred = self.fidelity_head(h_act).squeeze(-1)         # [B,T]

        if not use_candidate_scoring:
            # ── ORIGINAL REGRESSION PATH (unchanged) ──
            x_pred = self.action_head(h_act)                   # [B,T,d]
            if valid_mask is not None:
                vm = valid_mask.float()   # [B, T]
                _lf = F.l1_loss if getattr(self, 'loc_loss', 'mse') == 'l1' else F.mse_loss
                L_loc = (
                    _lf(x_pred, actions_x, reduction='none')
                    .mean(dim=-1)   # [B, T]
                    * vm
                ).sum() / vm.sum().clamp_min(1)
            else:
                _lf = F.l1_loss if getattr(self, 'loc_loss', 'mse') == 'l1' else F.mse_loss
                L_loc = _lf(x_pred, actions_x)
            x_pred_out = x_pred
        else:
            # ── CANDIDATE SCORING PATH ──
            K = candidates.shape[2]
            if self.use_linear_score_head:
                # ITEM 5: state-conditioned LINEAR coefficients. w,b depend
                # on h_act (the state token) ONLY, broadcast over K -- the
                # state cannot express anything beyond a per-feature linear
                # weight + scalar bias on the candidate features.
                w = self.coef_head(h_act)                       # [B,T,F]
                b = self.bias_head(h_act)                        # [B,T,1]
                scores = (w.unsqueeze(2) * candidates).sum(-1) + b   # [B,T,K]
                if not getattr(self, '_w_diag_done', False):
                    self._w_diag_done = True
                    with torch.no_grad():
                        vm0 = (valid_mask if valid_mask is not None
                               else torch.ones(B, T, dtype=torch.bool, device=w.device))
                        w_valid = w[vm0]                          # [N, F]
                        var_per_coef = w_valid.var(dim=0)
                        mean_abs_per_coef = w_valid.abs().mean(dim=0)
                        print(f"[W-DIAG] across-batch VAR(w) per coefficient "
                              f"(F={w_valid.shape[-1]}, N={w_valid.shape[0]}): "
                              f"{[round(v,6) for v in var_per_coef.tolist()]}", flush=True)
                        print(f"[W-DIAG] mean|w| per coefficient: "
                              f"{[round(v,4) for v in mean_abs_per_coef.tolist()]}", flush=True)
                        print(f"[W-DIAG] ratio VAR(w)/mean|w|^2 per coefficient (relative spread): "
                              f"{[round((var_per_coef[i]/(mean_abs_per_coef[i]**2+1e-12)).item(),4) for i in range(w_valid.shape[-1])]}",
                              flush=True)
            else:
                h_exp = h_act.unsqueeze(2).expand(B, T, K, H)
                scores = self.score_head(
                    torch.cat([h_exp, candidates], dim=-1)
                ).squeeze(-1)                                      # [B,T,K]

            scores_flat = scores.reshape(B * T, K)
            chosen_flat = chosen_idx.reshape(B * T)
            vm_flat = (valid_mask.reshape(B * T) if valid_mask is not None
                       else torch.ones(B * T, dtype=torch.bool, device=scores.device))

            if teacher_scores is not None:
                # Change 1d + CRITICAL-2 FIX: SOFT distillation target,
                # KL(teacher || student) over the K candidates.
                #
                # teacher_scores is cost-normalized MES in NATS-PER-COST, a
                # scale-free quantity whose across-candidate spread is tiny
                # (measured directly on Hartmann_6D: softmax(s/1.0) over
                # K=20 gave teacher entropy 2.9957 vs log(20)=2.9957 --
                # exactly uniform to 4 decimals, so KL against it was ~0 and
                # the score head learned nothing, while the tiny L_loc looked
                # like convergence). This is the same absolute-threshold-on-a-
                # scale-free-quantity trap already documented in
                # simulate_mf_trajectory's bes_delta docstring. Tuning
                # score_temp would repeat that mistake one level down, so
                # instead the scores are STANDARDIZED WITHIN each candidate
                # set before the softmax -- which is scale-invariant by
                # construction (post-standardization entropy: 1.3492 = 45% of
                # log(K)).
                #
                # has_soft marks steps whose teacher_scores is a real MES
                # vector. Steps from rollout_policy "thompson"/"random" have
                # no candidate-pool scores at all; standardizing their
                # placeholder would fabricate a soft distribution, so they
                # fall back to hard cross-entropy on chosen_idx.
                t_flat = teacher_scores.reshape(B * T, K)
                soft_flat = (has_soft.reshape(B * T) if has_soft is not None
                             else torch.ones(B * T, dtype=torch.bool, device=scores.device))
                sel_soft = vm_flat & soft_flat
                sel_hard = vm_flat & (~soft_flat)
                terms, n_terms = [], 0
                if sel_soft.any():
                    st = t_flat[sel_soft]
                    st = (st - st.mean(-1, keepdim=True)) / (st.std(-1, keepdim=True) + 1e-8)
                    teacher_p = F.softmax(st / self.score_temp, dim=-1)
                    student_logp = F.log_softmax(scores_flat[sel_soft], dim=-1)
                    terms.append(F.kl_div(student_logp, teacher_p, reduction='batchmean')
                                 * sel_soft.sum())
                    n_terms += sel_soft.sum()
                if sel_hard.any():
                    terms.append(F.cross_entropy(
                        scores_flat[sel_hard], chosen_flat[sel_hard].long()
                    ) * sel_hard.sum())
                    n_terms += sel_hard.sum()
                L_loc = (sum(terms) / n_terms.clamp_min(1)) if terms else scores_flat.sum() * 0.0
            else:
                L_loc = F.cross_entropy(
                    scores_flat[vm_flat], chosen_flat[vm_flat].long()
                )
            x_pred_out = None   # not used in scoring mode

        # Fidelity loss -- IDENTICAL in both modes.
        if valid_mask is not None:
            vm = valid_mask.float()   # [B, T]
            L_fid = (
                F.binary_cross_entropy(
                    p_pred, actions_ell.float(), reduction='none'
                )   # [B, T]
                * vm
            ).sum() / vm.sum().clamp_min(1)
        else:
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

        return loss, L_loc, L_fid, x_pred_out, p_pred

    def propose_mf(self, state, rtg_target, btg_target, timestep=0,
                    use_candidate_scoring=False, candidate_features=None,
                    fidelity_sampling=True, hist=None):
        """
        Single-step MF inference.
        state:      [state_dim] tensor
        use_candidate_scoring=False: ORIGINAL regression path, bit-for-bit
            unchanged from before this flag existed -- x_t comes from
            action_head, exactly as always.
        use_candidate_scoring=True: scores K candidates with score_head and
            returns the argmax's location. Rescaling to the benchmark's
            actual domain bounds happens at the SAME call site as the
            regression path's output always did
            (DirectMFRegretOptimization._propose_next_query, the single
            exit point for real x_t) -- both modes return [0,1]^d-normalized
            coordinates here, so no special-casing needed downstream.

        candidate_features (Change 1b/1c): [K, cand_feature_dim] tensor,
            REQUIRED when use_candidate_scoring=True. Built by the caller
            via mf_dro.build_candidate_features from a fresh uniform draw
            over the domain -- the SAME function and the SAME uniform
            candidate distribution simulate_mf_trajectory uses to build its
            training candidate sets (its roi_candidates pool is itself
            uniform over bounds), so train and inference score structurally
            identical inputs. The caller supplies these rather than this
            method drawing its own, because computing them needs the KO
            model and cost constants, which live on the optimizer side.
            Its first action_dim columns must be the [0,1]^d-normalized
            coordinates (build_candidate_features guarantees this).

        BEHAVIOR BEFORE THIS CHANGE (per spec's "report what propose_mf
        currently does"): in candidate-scoring mode this method drew its own
        `X_cand = torch.rand(200, action_dim)` internally and scored bare
        coordinates (`torch.cat([h_exp, X_cand])`), with no GP/MES features
        and no caller involvement. The distribution (uniform over the unit
        cube) already matched training's roi_candidates, so 1b's requirement
        was in fact already satisfied on the distribution front; what
        changes here is that the candidates now carry the per-candidate
        feature block (1c) and are built by the shared helper, so the two
        paths cannot diverge.

        Returns:    x_t [d], ell_t int {0,1}
        """
        # ISSUE-5 FIX: save the INCOMING mode and restore it on exit, rather
        # than unconditionally calling self.train() at the end. Previously
        # this method always left the module in train() mode, silently
        # undoing any caller-side .eval() guard the moment it returned --
        # so the call-site comment in _propose_next_query claiming the
        # guarantee held "even if propose_mf's internals change" was false.
        _was_training = self.training
        self.eval()
        with torch.no_grad():
            # Inlines forward_mf's embedding+transformer pipeline directly
            # (rather than calling forward_mf) so the candidate-scoring
            # branch can score 200 candidates without forcing them through
            # forward_mf's [B,T,K,d] batch-shaped interface. The dummy
            # action placeholder (ax/ae below) mirrors forward_mf's own
            # action-embedding requirement -- at inference the action isn't
            # known yet (it's what we're computing), so a zero placeholder
            # fills the 4th token slot before the transformer runs, exactly
            # as this method already did when it called forward_mf.
            # SLIDING-WINDOW INFERENCE (hist is None => original T=1 path,
            # bit-for-bit unchanged). hist is a list of dicts with keys
            # state/rtg/btg for the K-1 PRECEDING real queries, oldest first;
            # the current (state, rtg_target, btg_target) is appended last so
            # the readout position is the final state token. Positions run
            # 0..T-1, staying inside the trained range (0..rollout_length-1).
            if hist:
                _st = [h['state'] for h in hist] + [state]
                _rt = [float(h['rtg']) for h in hist] + [float(rtg_target)]
                _bt = [float(h['btg']) for h in hist] + [float(btg_target)]
                T = len(_st)
                s = torch.stack(_st).unsqueeze(0)                 # [1,T,state_dim]
                r = torch.tensor(_rt, dtype=state.dtype).view(1, T, 1)
                b = torch.tensor(_bt, dtype=state.dtype).view(1, T, 1)
                # h196: feed the REAL past actions, as DT Algorithm 1 does
                # (a + [action]) and as training does. The CURRENT step keeps a
                # zero placeholder -- its action is what we are predicting, and
                # the paper's own `a` is short by one at prediction time.
                # Falls back to zeros for any history entry lacking 'ax', so
                # older callers and h27-era behaviour remain reproducible.
                ax = torch.zeros(1, T, self.action_dim, dtype=state.dtype)
                ae = torch.zeros(1, T, dtype=torch.long)
                for _i, _h in enumerate(hist):          # history slots only; last stays zero
                    _hx = _h.get('ax')
                    if _hx is not None:
                        ax[0, _i] = _hx.reshape(-1)[:self.action_dim].to(ax.dtype)
                        ae[0, _i] = int(_h.get('ae', 0))
                ts = torch.arange(T, dtype=torch.long).unsqueeze(0)
            else:
                T = 1
                s = state.unsqueeze(0).unsqueeze(0)  # [1,1,state_dim]
                r = torch.tensor([[[rtg_target]]], dtype=state.dtype)
                b = torch.tensor([[[btg_target]]], dtype=state.dtype)
                ax = torch.zeros(1, 1, self.action_dim, dtype=state.dtype)
                ae = torch.zeros(1, 1, dtype=torch.long)
                ts = torch.tensor([[timestep]], dtype=torch.long)

            H = self.hidden_size
            # FIX 2: same LayerNorm modules as forward_mf -- required for
            # train/inference consistency (skipping these here while
            # forward_mf applies them during training would silently make
            # propose_mf feed the transformer differently-scaled embeddings
            # than what it was trained on).
            # H179: same standardisation as training (update=False at inference).
            r, b = self._std_cond(r, b, update=False)
            rtg_emb = self.reward_ln(self.reward_embedding(r))   # r already [1,1,1] -> [1,1,H]
            btg_emb = self.btg_ln(self.btg_embed(b))             # b already [1,1,1] -> [1,1,H]
            s_emb = self.state_ln(self.state_embedding(s))                          # [1,1,H]
            act_inp = torch.cat([ax, ae.float().unsqueeze(-1)], dim=-1)  # [1,1,d+1]
            a_emb = self.action_ln(self.action_embed_mf(act_inp))                   # [1,1,H]
            # H4: must mirror forward_mf EXACTLY -- same token count, same
            # readout index, same AdaLN -- or the train/inference mismatch
            # this method was just fixed for comes straight back.
            _adaln = (self.rtg_conditioning == 'adaln')
            _tps = 2 if _adaln else 4
            pos_emb = self.position_embedding(ts).repeat_interleave(_tps, dim=1)
            if _adaln:
                seq = torch.stack([s_emb, a_emb], dim=2).reshape(1, _tps * T, H)
            else:
                seq = torch.stack([rtg_emb, btg_emb, s_emb, a_emb], dim=2).reshape(1, _tps * T, H)
            seq = seq + pos_emb
            # FIX 4: apply the SAME causal mask training uses. Without it
            # inference was BIDIRECTIONAL while training was causal -- a
            # train/inference mismatch in the attention pattern itself.
            _L = _tps * T
            _cm = torch.triu(torch.ones(_L, _L, dtype=torch.bool, device=seq.device),
                              diagonal=1)
            h_full = self.transformer(seq, mask=_cm)  # [1,_L,H]
            # FIX 3: STATE token, matching forward_mf's h_act. With a sliding
            # window the readout is the LAST state token (the current step).
            h = h_full[0, (0 if _adaln else 2)::_tps, :][-1]   # [H]
            if _adaln:
                _cond = torch.tensor([[rtg_target, btg_target]],
                                      dtype=h.dtype, device=h.device)
                _gb = self.adaln_mod(_cond)[0]
                h = _gb[:H] * self.adaln_ln(h.unsqueeze(0))[0] + _gb[H:]

            if not use_candidate_scoring:
                # ── ORIGINAL: regression head (unchanged) ──
                x_t = self.action_head(h).clamp(0.0, 1.0)
            else:
                # ── Score the caller-supplied candidate features ──
                if candidate_features is None:
                    raise ValueError(
                        "propose_mf(use_candidate_scoring=True) requires "
                        "candidate_features -- build them with "
                        "mf_dro.build_candidate_features so training and "
                        "inference score identical feature blocks."
                    )
                cf = candidate_features.to(dtype=h.dtype, device=h.device)
                K = cf.shape[0]
                if self.use_linear_score_head:
                    # ITEM 5: same state-conditioned linear scoring as
                    # forward_mf's training path -- w,b computed ONCE from
                    # this single state, dotted with each candidate's own
                    # feature vector.
                    w = self.coef_head(h)                        # [F]
                    b = self.bias_head(h)                         # [1]
                    scores = (cf * w.unsqueeze(0)).sum(-1) + b
                else:
                    h_exp = h.unsqueeze(0).expand(K, -1)
                    scores = self.score_head(
                        torch.cat([h_exp, cf], dim=-1)
                    ).squeeze(-1)
                # First action_dim columns are the normalized coordinates
                # (build_candidate_features' own layout).
                x_t = cf[scores.argmax(), :self.action_dim]

            # Fidelity: computed once, IDENTICAL in both modes (fidelity_head
            # already ends in nn.Sigmoid(), no extra torch.sigmoid() here --
            # and computed once, not once per use, to avoid two independent
            # forward passes silently having to agree).
            p_val = self.fidelity_head(h).item()
            # THRESHOLD-BUG FIX (config flag, default True): p_val>0.5
            # requires the head to believe HF is MORE LIKELY THAN NOT before
            # ever selecting it -- but the measured tau=0 HF label rate is
            # 0.371, so a PERFECTLY CALIBRATED head outputs p_val~0.371 at
            # exactly the position real inference always uses, and the
            # threshold then selects LF on every single iteration by
            # construction, independent of anything the head actually
            # learned. fidelity_sampling=True draws ell_t ~ Bernoulli(p_val)
            # instead, matching what the head was trained to predict (a
            # probability) rather than imposing an unrelated >50% bar.
            # fidelity_sampling=False keeps the original threshold, for
            # direct ablation.
            if fidelity_sampling:
                ell_t = 1 if torch.rand(1).item() < p_val else 0
            else:
                ell_t = 1 if p_val > 0.5 else 0
        self.train(_was_training)
        self.last_p_pred = p_val  # exposed for diagnostics/verification
        return x_t, ell_t
