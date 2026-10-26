# UniRestorer++ architecture entries

See the [README](../README.md) for environment setup, entry commands, data requirements and the current training-workflow gaps. This page describes the architecture interfaces and surrogate-gradient design.

`basicsr/models/archs/unirestorerpp_arch.py` exposes both versions:

| Registry type | Experts | Routing |
| --- | --- | --- |
| `UniRestorerPP` | Original LoRA version; legacy name retained | Once in each wrapped Transformer block |
| `UniRestorerPPLoRA` | Explicit name inheriting the same LoRA implementation | Same as `UniRestorerPP`; identical state-dict keys |
| `UniRestorerPPFullModel` | Independent, complete native Restormer models | Once at the image input |

The new implementation is in `unirestorerpp/full_expert_network.py`. It reuses the existing `HierarchyIndex`, `DenseDegradationPrior`, `DensePriorRouter` and unchanged native `restormer_arch.py`; it introduces no LoRA modules or per-block routers. Upstream Restormer attribution remains in that native source, and the BasicSR registry attribution remains in the existing registry builder. No third-party implementation or weights were downloaded for this addition.

## Full-model forward

1. Compute dense `E`, `E_d`, `E_g` from the original, unpadded RGB image `X`, as in the original architecture.
2. Compute a separate learnable shallow routing feature `H = GELU(Conv3x3(X))`. The convolution has stride 1, padding 1, no bias, and `router_feature_channels` output channels (defaults to `dim`). This new stem is independent of every restoration model. It is an explicit implementation choice for the new input router, not a claimed transfer of old block weights.
3. Invoke one existing router with `(H, E_d, E_g)`. Its prior features are bilinearly resized to H as before. DER selects one leaf; GER selects one level; their ancestor selects one node. Level 0 selects the root/base. The router keeps its existing training noise and selected-probability gate. There is no routing inside the full experts.
4. Pad X once on the right/bottom to multiples of 8: reflect when legal, otherwise replicate. The complete base and every active complete expert see this same full, unmasked padded image. Each native Restormer already includes its input residual. Crop complete outputs back to the original dimensions before selection; do not add X again.
5. Hard masks already have the original HxW resolution. No mask interpolation is needed, so masks remain boolean, disjoint and cover every pixel including the root. A node that is never selected is not evaluated. Every active expert still processes the whole image.

The tree uses the existing perfect binary indexing. Root is heap node 0 and expert ID -1; non-root `experts[i]` is heap node `i+1`. A depth D tree has `2**(D+1)-1` complete restoration models including the base. The base is always evaluated to preserve the old gate's baseline, even if no pixel selects root. Memory and compute can therefore grow substantially with depth and the number of active nodes.

## Hard values and surrogate gradients

Let `Y0=base(X)`, `Yi=expert_i(X)`, `Mi` be the hard boolean masks and `g=(pD/detach(pD))*(pG/detach(pG))`. The inherited gradient reference is:

```text
Y_reference = Y0 + sum_i Mi * g * (Yi - Y0)
```

`g` is exactly one in forward and retains gradients of the two selected probabilities. The new implementation uses `torch.where` and a small custom autograd function: forward copies the selected complete output exactly, avoiding floating-point cancellation in `Y0+(Yi-Y0)`. Backward differentiates the reference formula above. It is a deliberate surrogate for discrete routing, not the mathematical derivative of argmax; finite-difference gradcheck of the gate would be inappropriate. Relative to each selected full output, the gradients are `dY/dYi=g`, `dY/dY0=1-g`, and `dY/dg=Yi-Y0`, summed across image channels for the gate. The masks have no gradient. Root-only pixels have the base gradient and no direct reconstruction gradient to DER/GER, as before.

If an entire batch selects only root while base is frozen and X does not require gradients, the reconstruction output has no trainable autograd path: calling a reconstruction-only `loss.backward()` raises a no-grad error. This inherited edge case is deliberately preserved; this arch addition does not add a zero-valued training connection or change any loss.

Noise remains enabled only when both `model.training` and `router_noise` are true, with DER standard deviation `1/num_leaves` and GER `1/num_levels`. `model.eval()` disables this noise.

The default base and DR encoder are frozen and stay in eval during `model.train()`; full experts, the new stem, the router and both phi heads remain trainable. `freeze_base=False` and `freeze_dr=False` explicitly unfreeze the corresponding component. No `no_grad` wraps the base: gradients to the image input remain available. Full experts initially copy base values but share no Parameters or buffers. Equal initial outputs imply zero initial reconstruction gradient through the gate until experts diverge. This addition does not prescribe expert pretraining or alter existing losses/training stages.

## Construction and checkpoints

From the repository root, with the existing requirements available:

```python
import torch
import yaml
from basicsr.models.archs import build_network

with open('options/arch_unirestorerpp_full_model.yml', encoding='utf-8') as stream:
    net = build_network(yaml.safe_load(stream)).eval()
with torch.no_grad():
    restored, aux = net(torch.randn(1, 3, 17, 19), return_aux=True)
assert restored.shape == (1, 3, 17, 19)
assert aux['hard_masks'].sum(dim=1).eq(1).all()
```

The YAML is architecture-only, uses random initial weights unless an explicit local native `base_checkpoint` is supplied, and is not a trained restoration example. `base_checkpoint_key` selects an envelope such as `params`; the selected native state loads strictly into the base and initializes each independent expert with the same values. `native_base_state_dict()` exports native Restormer keys. Save/load the complete new model using its ordinary `state_dict()` and the same constructor configuration. Individual complete expert weights can be loaded explicitly with `net.experts[i].load_state_dict(native_state, strict=True)`.

Legacy LoRA checkpoints/configurations continue using `UniRestorerPP` or `UniRestorerPPLoRA`. They are not full-model checkpoints and do not initialize this new router stem or its one-shot feature branches automatically. Existing LoRA training/assembly helpers are not extended by this arch-only addition. `return_aux=True` on the new class returns one `RoutingDecision` at `aux['routing']`, boolean root/non-root `hard_masks`, prior features and padding metadata; it does not imitate the old per-block routing dictionary. `parameter_counts()` reports only complete restoration Parameters (`base`, `experts`, `total`), excluding the prior/stem/router.

## CPU checks

Run from the repository root (no weights, images, training, or GPU needed):

```bash
python -B -m unittest discover -s tests -p test_full_model_routing_kernel.py -v
python -B -m unittest discover -s tests -p test_unirestorerpp_full_expert.py -v
```

The first suite checks the actual routing core and extracted exact-selection autograd class with PyTorch alone; it does not instantiate the full architecture. The second includes real small Restormers and the real dense encoder, plus controlled output/router fixtures for exact masks and gradient comparisons. It requires both torchvision and einops. A missing dependency is a blocked integration run, not a passing or skipped test. Tests are synthetic CPU unit checks and do not establish trained restoration quality, CUDA/AMP behavior, or compatibility with the unextended training pipeline.

Last local verification (2026-10-26): all 8 kernel tests passed against the target repository. The 13 full-architecture tests remain unexecuted because import fails on missing `einops`; the limited check of existing Python/conda environments found no ready torch+einops environment. Registry/legacy-alias behavior, complete-network forward/backward, independent model Parameters and save/load have static implementation/test coverage, but are not runtime-verified by the kernel suite. No dependencies were installed and no training or GPU tasks were launched during this verification.
