import torch
from typing import Optional, Sequence, Tuple, Dict

from .fiedler_sign import DEFAULT_PIPELINE, SignContext, orient_eigenvector


def _lexicographic_sign(
    W_top: torch.Tensor, v2: torch.Tensor, lex_round_decimals: int
) -> Tuple[torch.Tensor, int]:
    """Historical sign rule: pick the orientation whose reordered W is lexicographically smaller.

    Returns ``(order, sign)`` where ``sign`` is +1 if the ascending-Fiedler order was chosen.
    """
    order_plus = torch.argsort(v2)
    order_minus = torch.argsort(-v2)
    W_plus = W_top[order_plus][:, order_plus]
    W_minus = W_top[order_minus][:, order_minus]
    scale = 10 ** lex_round_decimals
    W_plus_flat = torch.round(W_plus.flatten() * scale)
    W_minus_flat = torch.round(W_minus.flatten() * scale)
    if W_plus_flat.tolist() < W_minus_flat.tolist():
        return order_plus, 1
    return order_minus, -1


def canonicalize_topology(
    W_top: torch.Tensor,
    eigengap_tol: float = 1e-6,
    tie_tol: float = 1e-8,
    lex_round_decimals: int = 8,
    eps: float = 1e-6,
    sign_rule: str = "fiedler_cascade",
    sign_pipeline: Optional[Sequence[str]] = None,
    sign_context: Optional[SignContext] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Spectral-seriation ordering of a topology matrix by its Fiedler vector.

    ``sign_rule`` selects how the Fiedler vector's global sign is resolved:

    * ``"fiedler_cascade"`` (default): use :func:`vino.transforms.fiedler_sign.orient_eigenvector`
      with ``sign_pipeline`` (defaulting to the four-stage cascade) and ``sign_context``.
    * ``"lexicographic_topology"``: the historical rule that compares the reordered ``W_top``.

    ``meta`` records ``sign_method`` (the deciding method, or ``"unresolved"`` / the historical
    rule name) and ``sign_value`` (+1 / -1).
    """
    n = W_top.size(0)
    meta = {
        "canonicalization_unstable": False,
        "fiedler_eigengap": 0.0,
        "num_fiedler_ties": 0,
        "sign_rule": sign_rule,
        "sign_method": "trivial",
        "sign_value": 1,
    }

    if n <= 2:
        return torch.arange(n), meta

    # Symmetric normalized Laplacian
    deg = W_top.sum(dim=1)
    d_inv_sqrt = torch.pow(deg.clamp(min=eps), -0.5)

    D_inv_sqrt = torch.diag(d_inv_sqrt)
    L_sym = torch.eye(n) - D_inv_sqrt @ W_top @ D_inv_sqrt

    # Eigendecomposition
    # torch.linalg.eigh expects symmetric matrix
    L_sym = (L_sym + L_sym.T) / 2
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
    except Exception:
        meta["canonicalization_unstable"] = True
        return torch.arange(n), meta

    v2 = eigenvectors[:, 1]
    # The Fiedler vector (eigenvectors[:, 1], eigenvalue lambda_1) is only well defined
    # when lambda_1 is separated from BOTH neighbors. Checking only lambda_2 - lambda_1
    # misses a degenerate smallest eigenvalue (e.g. disconnected graphs, where
    # lambda_0 == lambda_1 == 0). Use the minimum of the lower and upper gaps.
    lower_gap = float((eigenvalues[1] - eigenvalues[0]).item())
    upper_gap = float((eigenvalues[2] - eigenvalues[1]).item())
    meta["fiedler_eigengap"] = min(lower_gap, upper_gap)

    if meta["fiedler_eigengap"] < eigengap_tol:
        meta["canonicalization_unstable"] = True

    # Check for near-ties in Fiedler vector
    v2_diff = torch.abs(v2.unsqueeze(0) - v2.unsqueeze(1))
    v2_diff.fill_diagonal_(float("inf"))
    if (v2_diff < tie_tol).any():
        meta["num_fiedler_ties"] = int((v2_diff < tie_tol).sum().item() // 2)
        meta["canonicalization_unstable"] = True

    if sign_rule == "lexicographic_topology":
        order, sign = _lexicographic_sign(W_top, v2, lex_round_decimals)
        meta["sign_method"] = "lexicographic_topology"
        meta["sign_value"] = int(sign)
        return order, meta

    if sign_rule != "fiedler_cascade":
        raise ValueError(
            f"Unknown sign_rule {sign_rule!r}; use 'fiedler_cascade' or 'lexicographic_topology'"
        )

    pipeline = tuple(sign_pipeline) if sign_pipeline else DEFAULT_PIPELINE
    v2_np = v2.detach().cpu().numpy()
    _, sign, method = orient_eigenvector(v2_np, sign_context, pipeline)
    meta["sign_method"] = method
    meta["sign_value"] = int(sign)
    order = torch.argsort(v2 * float(sign))
    return order, meta
