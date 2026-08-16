"""
Quantile RTG head for the Decision Transformer: predicts M quantiles of the
achievable-return distribution from the *action* hidden state one step back
(h^a_{tau-1}), never from the RTG token itself (h^R_tau), to avoid trivially
echoing the RTG input at inference. See Step 5 for wiring this into
src/model/decisionTransformer.py.
"""
import torch
import torch.nn as nn

DEFAULT_ALPHA_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


class QuantileRTGHead(nn.Module):
    """
    Input:  h_shifted [batch, seq_len, embed_dim]  (action hidden, shifted)
    Output: Q_hat     [batch, seq_len, M]
    Architecture: embed_dim -> 64 (ReLU) -> M (no activation)
    """
    def __init__(self, embed_dim, M=5, alpha_levels=DEFAULT_ALPHA_LEVELS):
        super().__init__()
        if len(alpha_levels) != M:
            raise ValueError(f"len(alpha_levels)={len(alpha_levels)} must equal M={M}")
        self.M = M
        self.register_buffer('alpha_levels', torch.tensor(alpha_levels, dtype=torch.get_default_dtype()))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, M),
        )

    def forward(self, h_shifted: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_shifted)


def compute_pinball_loss(Q_hat: torch.Tensor, R_true: torch.Tensor, alpha_levels) -> torch.Tensor:
    """
    Q_hat:  [batch, seq_len, M]
    R_true: [batch, seq_len]
    Returns: scalar mean pinball loss
    """
    if torch.is_tensor(alpha_levels):
        alpha = alpha_levels.to(device=Q_hat.device, dtype=Q_hat.dtype)
    else:
        alpha = torch.tensor(alpha_levels, device=Q_hat.device, dtype=Q_hat.dtype)
    alpha = alpha.view(*([1] * (Q_hat.ndim - 1)), -1) # broadcast over [batch, seq_len, M]

    diff = R_true.unsqueeze(-1) - Q_hat # [batch, seq_len, M]
    loss = torch.where(diff >= 0, alpha * diff, (alpha - 1.0) * diff)
    return loss.mean()


def interpolate_quantile(Q_hat_single: torch.Tensor, alpha_inference: float, alpha_levels) -> torch.Tensor:
    """
    Q_hat_single:    [M] predicted quantiles at one step
    alpha_inference: float in (0,1)
    Returns: scalar rtg_target
    """
    alpha_levels = list(alpha_levels)
    M = len(alpha_levels)

    if alpha_inference <= alpha_levels[0]:
        return Q_hat_single[0]
    if alpha_inference >= alpha_levels[-1]:
        return Q_hat_single[-1]

    j_low = max(j for j in range(M) if alpha_levels[j] <= alpha_inference)
    j_high = min(j for j in range(M) if alpha_levels[j] >= alpha_inference)

    if j_low == j_high:
        return Q_hat_single[j_low]

    weight = (alpha_inference - alpha_levels[j_low]) / (alpha_levels[j_high] - alpha_levels[j_low])
    return (1.0 - weight) * Q_hat_single[j_low] + weight * Q_hat_single[j_high]


if __name__ == '__main__':
    torch.manual_seed(0)

    # --- Test 1: shape ---
    head = QuantileRTGHead(embed_dim=128, M=5)
    Q_hat = head(torch.randn(4, 8, 128))
    print(f"Test 1 - Q_hat.shape: {tuple(Q_hat.shape)} (EXPECT: (4, 8, 5))")

    # --- Test 2: pinball loss value ---
    Q_hat_zeros = torch.zeros(2, 4, 5)
    R_true_ones = torch.ones(2, 4)
    loss = compute_pinball_loss(Q_hat_zeros, R_true_ones, DEFAULT_ALPHA_LEVELS)
    print(f"Test 2 - pinball loss: {loss.item():.6f} (EXPECT: approx 0.5)")

    # --- Test 3: no leakage ---
    # Minimal causal Decision-Transformer-like scaffold, used only to validate that
    # QuantileRTGHead's shifted-action-hidden-state input is free of RTG leakage.
    # A causal mask is required for this guarantee to hold at all -- see the note
    # printed below and the message to the user about src/model/decisionTransformer.py
    # (which currently has no causal mask, so this property does not yet hold there).
    class _ToyCausalDT(nn.Module):
        def __init__(self, state_dim, action_dim, embed_dim=32, n_heads=4, n_layers=2, max_len=20):
            super().__init__()
            self.embed_dim = embed_dim
            self.state_embedding = nn.Linear(state_dim, embed_dim)
            self.action_embedding = nn.Linear(action_dim, embed_dim)
            self.reward_embedding = nn.Linear(1, embed_dim)
            self.position_embedding = nn.Embedding(max_len, embed_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=4 * embed_dim,
                dropout=0.0, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        def forward(self, states, actions, rewards, timesteps):
            B, L = states.shape[0], states.shape[1]
            se = self.state_embedding(states)
            ae = self.action_embedding(actions)
            re = self.reward_embedding(rewards.unsqueeze(-1))
            seq = torch.stack([se, ae, re], dim=2).reshape(B, 3 * L, self.embed_dim)
            seq = seq + self.position_embedding(timesteps).repeat_interleave(3, dim=1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(3 * L).to(seq.device)
            out = self.transformer(seq, mask=causal_mask)
            return out[:, 1::3] # [B, L, D] action hidden states, one per timestep

    state_dim, action_dim, embed_dim = 4, 2, 32
    toy_dt = _ToyCausalDT(state_dim, action_dim, embed_dim=embed_dim)
    toy_dt.eval()
    quantile_head = QuantileRTGHead(embed_dim=embed_dim, M=5)
    quantile_head.eval()

    B, L = 2, 6
    states = torch.randn(1, L, state_dim).expand(B, L, state_dim)
    actions = torch.randn(1, L, action_dim).expand(B, L, action_dim)
    timesteps = torch.arange(L).unsqueeze(0).expand(B, L)

    rewards_A = torch.randn(B, L)
    rewards_B = rewards_A.clone()
    rewards_B[:, -1] = rewards_A[:, -1] + 100.0 # perturb only the LAST reward token

    with torch.no_grad():
        h_action_A = toy_dt(states, actions, rewards_A, timesteps)
        h_action_B = toy_dt(states, actions, rewards_B, timesteps)
        zero_pad = torch.zeros(B, 1, embed_dim)
        h_shifted_A = torch.cat([zero_pad, h_action_A[:, :-1, :]], dim=1)
        h_shifted_B = torch.cat([zero_pad, h_action_B[:, :-1, :]], dim=1)
        Q_hat_A = quantile_head(h_shifted_A)
        Q_hat_B = quantile_head(h_shifted_B)

    identical = torch.allclose(Q_hat_A, Q_hat_B, atol=1e-5)
    print(f"Test 3 - Q_hat_A == Q_hat_B within 1e-5 despite different final-step reward: {identical} (EXPECT: True)")
    print("  (uses a causal-masked toy transformer; src/model/decisionTransformer.py has")
    print("   no causal mask today, so this guarantee does not yet hold there -- see note below)")

    # --- Test 4: calibration ---
    torch.manual_seed(1)
    cal_embed_dim = 8
    cal_head = QuantileRTGHead(embed_dim=cal_embed_dim, M=5)
    optimizer = torch.optim.Adam(cal_head.parameters(), lr=0.05)

    h_fixed = torch.randn(1, cal_embed_dim)
    true_mean, true_std = 2.0, 1.0
    train_batch_size = 256

    for step in range(500):
        R_true = true_mean + true_std * torch.randn(train_batch_size)
        h_batch = h_fixed.expand(train_batch_size, cal_embed_dim).unsqueeze(1) # [B, 1, D]
        Q_hat_train = cal_head(h_batch) # [B, 1, M]
        loss = compute_pinball_loss(Q_hat_train, R_true.unsqueeze(1), DEFAULT_ALPHA_LEVELS)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Q_hat_final = cal_head(h_fixed.unsqueeze(1)).squeeze() # [M]
    eval_samples = true_mean + true_std * torch.randn(1000)
    print("Test 4 - empirical coverage vs nominal alpha (EXPECT: approx equal):")
    for j, alpha_j in enumerate(DEFAULT_ALPHA_LEVELS):
        coverage = (eval_samples < Q_hat_final[j]).float().mean().item()
        print(f"  alpha={alpha_j:.2f}  Q_hat={Q_hat_final[j].item():.4f}  empirical_coverage={coverage:.3f}")
