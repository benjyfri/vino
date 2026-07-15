# Method: Graph-to-Image Foundation-Model Transfer

This project implements a proof-of-concept for representing graphs as 3-channel matrix images for use with pretrained vision foundation models like DINOv3.

## 1. Graph Image Construction

We construct a 3-channel image for each graph. Let $n$ be the number of nodes.

### Channel 1: Topology (APSP Heat Kernel)
Given the undirected adjacency matrix, we compute all-pairs shortest paths $D$. 
For connected pairs, the heat kernel is:
$W_{top}[i,j] = \exp\left(-\left(\frac{D_{norm}[i,j]}{\sigma}\right)^{power}\right)$
where $D_{norm} = D / \max_{finite}(D)$. Disconnected pairs have value 0.

### Canonicalization (Fiedler Vector)
We compute the symmetric normalized Laplacian $L_{sym}$ from $W_{top}$.
Nodes are sorted by the Fiedler vector (eigenvector of the second smallest eigenvalue).
Sign ambiguity is resolved by lexicographically comparing the sorted $W_{top}$.
This permutation is applied to all channels.

This is spectral seriation, not a complete canonical labeling. It is deterministic under a
simple Fiedler eigenvalue, distinct Fiedler coordinates, and a stable eigensolver, after resolving
the global sign. Repeated eigenvalues, tied coordinates, disconnected graphs, and automorphisms
can remain ambiguous. The cache records eigengap/tie diagnostics, and
`vino.benchmarks.robustness.permutation_stability` measures empirical relabeling drift.

### Channel 2: Node Covariance (ALL-IN style)
Node features $X$ are projected using a random matrix $C_{node}$.
We compute representations at propagation steps $p=0,1,2$:
$R_p = A_{prop}^p (X C_{node})$
$K_{node} = \sum_p \alpha_p \frac{R_{p,c} R_{p,c}^T}{h_{node}}$
where $R_{p,c}$ is centered over nodes.

### Channel 3: Edge Covariance
Edge features are projected and aggregated incident to each node to form $R_{edge\_node}$.
Similar covariance operations are applied:
$K_{edge} = \sum_p \beta_p \text{NodeCov}(A_{prop}^p R_{edge\_node})$

## 2. Padding and Masking
Images are normalized to $[0,1]$, stacked into a $3 \times n \times n$ tensor, and zero-padded to $3 \times N_{max} \times N_{max}$ ($N_{max}=256$).

Cache schema version 2 may instead store the cropped tensor and dynamically pad a batch. Cache
identity hashes the dataset/image configuration plus schema version. Graphs exceeding `N_max`
and malformed records fail preprocessing; the default accepted failure fraction is zero.

## 3. Architecture
The 3-channel images are passed through an input adapter (e.g., $1 \times 1$ convolution) and then a frozen or lightly finetuned DINOv3 backbone. A task head predicts graph-level properties.

Cached valid regions are cropped and resized or patch-aligned before the model. Invalid regions
are zeroed before and after a learned stem. Pooling is explicit (`auto`, `cls`, or `mean`).

## Evaluation protocol

BBBP uses deterministic Bemis-Murcko scaffold groups by default; molhiv uses the official OGB
split and evaluator. Split graph IDs and SHA-256 checksums are frozen in cache manifests. Model
selection uses validation data only and undefined selection metrics are errors rather than silent
fallbacks. See `configs/experiments/PAPER_PROTOCOL.md` for the locked paper experiment plan.

## Limitations
- Canonicalization is unstable for graphs with exact automorphisms or repeated eigenvalues.
- Large graphs ($>256$ nodes) are skipped or require truncation.
- Designed as a fast proof-of-concept.
