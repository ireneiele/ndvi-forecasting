import torch
import torch.nn as nn

from .layers import PositionalEncoding


class AgriMatNetQuantile(nn.Module):
    """
    Variante quantile di AgriMatNet: produce un vettore di quantili per ciascun timestep futuro.
    """

    def __init__(
        self,
        input_dim,
        quantiles,
        d_model=128,
        num_layers=2,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        if not quantiles:
            raise ValueError("La lista dei quantili non può essere vuota.")
        if sorted(quantiles) != list(quantiles):
            raise ValueError("I quantili devono essere forniti in ordine crescente.")
        if len(set(quantiles)) != len(quantiles):
            raise ValueError("I quantili devono essere tutti distinti.")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile non valido: {q}. Devono essere compresi tra 0 e 1.")

        self.input_dim = input_dim
        self.quantiles = list(quantiles)
        self.d_model = d_model

        self.history_proj = nn.Linear(input_dim, d_model)
        self.future_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="relu",
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.future_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.history_pos = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.future_pos = PositionalEncoding(d_model=d_model, dropout=dropout)

        fusion_dim = d_model * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.quantiles)), # Unica differenza rispetto al modello puntuale adesso la testa da un numero di neuroni pari al numero di quantili 
        )

    @staticmethod
    def _clean_sequence(sequence, mask):
        zeros = torch.zeros_like(sequence)
        return torch.where(mask, zeros, sequence)

    def encode_history(self, history, history_mask, history_pad_mask):
        tokens_to_mask = history_pad_mask | history_mask.all(dim=-1)
        clean = self._clean_sequence(history, history_mask)
        emb = self.history_proj(clean)
        emb = self.history_pos(emb)
        encoded = self.history_encoder(emb, src_key_padding_mask=tokens_to_mask)
        keep = ~tokens_to_mask
        weights = keep.float()
        summed = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        summary = summed / counts.unsqueeze(-1)
        return encoded, summary, tokens_to_mask

    def encode_future(self, future, future_mask, future_pad_mask):
        tokens_to_mask = future_pad_mask | future_mask.all(dim=-1)
        clean = self._clean_sequence(future, future_mask)
        emb = self.future_proj(clean)
        emb = self.future_pos(emb)
        encoded = self.future_encoder(emb, src_key_padding_mask=tokens_to_mask)
        return encoded, tokens_to_mask

    def forward(self, batch):
        history = batch["history"]
        future = batch["future"]
        history_mask = batch["history_mask"]
        future_mask = batch["future_mask"]
        history_pad_mask = batch["history_pad_mask"]
        future_pad_mask = batch["future_pad_mask"]
        future_target_positions = batch["future_target_positions"]

        _, history_summary, _ = self.encode_history(history, history_mask, history_pad_mask)
        future_encoded, _ = self.encode_future(future, future_mask, future_pad_mask)

        B, forecast_window = future_target_positions.shape
        gather_index = future_target_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        selected_future = torch.gather(future_encoded, 1, gather_index)

        history_context = history_summary.unsqueeze(1).expand(-1, forecast_window, -1)
        fusion = torch.cat([selected_future, history_context], dim=-1)
        preds = self.head(fusion)
        return preds


def quantile_loss(preds, targets, mask, quantiles, weights=None):
    """
    Calcola la pinball loss su una batteria di quantili.
    preds: (B, T, Q)
    targets: (B, T)
    mask: (B, T) True sui valori da ignorare
    quantiles: lista di quantili in (0,1)
    weights: opzionale (B, T) con pesi non negativi applicati prima del masking
    """
    if preds.size(-1) != len(quantiles):
        raise ValueError("Dimensione finale di preds incoerente con il numero di quantili.")

    keep = ~mask  # mask True per i NaN → li ignoriamo
    weight_tensor = keep.float()
    if weights is not None:
        if weights.shape != keep.shape:
            raise ValueError("weights deve avere la stessa shape di targets/mask.")
        weight_tensor = weight_tensor * weights

    targets = targets.unsqueeze(-1)  # (B, T, 1)
    diff = targets - preds  # (B, T, Q)

    losses = []
    for idx, tau in enumerate(quantiles):
        diff_tau = diff[..., idx]
        positive = (diff_tau >= 0).float()
        loss_tau = torch.abs(diff_tau) * (tau * positive + (1 - tau) * (1 - positive))
        losses.append(loss_tau)

    loss_stack = torch.stack(losses, dim=-1)  # (B, T, Q)
    loss_stack = loss_stack * weight_tensor.unsqueeze(-1)
    denom = weight_tensor.sum().clamp(min=1e-8).float()
    return loss_stack.sum() / (denom * len(quantiles))
