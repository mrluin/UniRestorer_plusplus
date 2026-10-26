# UniRestorer++: Scalable Universal Image Restoration with Multi-Granularity Degradation Modeling

[Jingbo Lin](https://scholar.google.com/citations?user=zPycW30AAAAJ&hl=zh-CN&oi=ao), [Zhilu Zhang](https://scholar.google.com/citations?user=8pIq2N0AAAAJ&hl=zh-CN&oi=ao),
[Fan Li](),
[Renjing Pei](),
[Wenbo Li](https://scholar.google.com/citations?user=foGn_TIAAAAJ&hl=zh-CN&oi=ao),
[Hang Xu](https://scholar.google.com/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao),
[Hongzhi Zhang](https://scholar.google.com/citations?user=Ysk4WBwAAAAJ&hl=zh-CN&oi=ao), and [Wangmeng Zuo](https://scholar.google.com/citations?user=rUOpCEYAAAAJ&hl=zh-CN&oi=ao)




## Environment

[requirements.txt](requirements.txt) pins PyTorch 2.5.1, torchvision 0.20.1, einops 0.8.1 and the core image/YAML utilities. In an environment with the appropriate PyTorch build, install the declared dependencies with:

```bash
python -m pip install -r requirements.txt
```

Caption-bank generation additionally needs `open_clip_torch` and an explicit local ViT-B-32 checkpoint; that package is not included in `requirements.txt`. Dataset files and pretrained weights are not bundled. The supplied stage/evaluation runners use CPU; CUDA, AMP and distributed runs have not been validated.


## WorkFlow

Each script reads its own top-level `CONFIG`, with paths relative to the repository root. These tools do **not** accept `--opt`. Fill the required recipe values (including run settings, optimizer/scheduler, and DR initialization/BN/loss/mixing settings) and provide the input artifacts first. Some optional values such as `resume` may remain `None`.

The commands below are the correct module entry points; they are not a claim that the workflow has run successfully.

| Step | Purpose | Command | Current status |
| --- | --- | --- | --- |
| 1 | Prepare frozen text caption bank | `python -B -m tools.prepare_caption_bank` | Implementation present; needs local teacher, captions and `open_clip_torch` |
| 2 | Train dense degradation encoder D | `python -B -m tools.train_dr` | Implementation present; requires a filled training recipe and bank |
| 3 | Extract dense features and image GAP vectors | `python -B -m tools.extract_dr` | Implementation present; requires a trained D checkpoint |
| 4 | Build the binary degradation tree | `python -B -m tools.hierarchical_clustering` | Implementation present; requires completed feature artifacts |
| 5 | Train the complete Restormer base | `python -B -m tools.train_base` | Entry exists; not connected |
| 6 | Train one LoRA expert per non-root cluster | `python -B -m tools.train_cluster_lora` | Entry exists; not connected |
| 7 | Assemble base, LoRA experts and D | `python -B -m tools.assemble_cluster_lora` | Entry exists; not connected |
| 8 | Initialize DER using true-leaf teacher CE | `python -B -m tools.train_der` | Entry exists; not connected |
| 9 | Jointly train routers, phi heads and LoRA | `python -B -m tools.train_joint` | Entry exists; not connected |


## Data and evaluation

The stage configurations refer to user-supplied files:

| Input | Required content |
| --- | --- |
| `data/captions_input.csv` | `caption_id,caption` |
| `data/dr_samples.csv` | `image_path,caption_id`; image paths relative to `data/inputs` |
| `data/restoration_pairs.csv` | Unique `sample_id,lq_path,gt_path`; paired RGB images under `data/inputs` and `data/targets`, with matching sizes |
| Hierarchy artifacts | Completed extraction/clustering outputs, including sample mapping and labels; pair IDs must match hierarchy sample IDs, and `lq_path` must match the hierarchy relative path |


## Acknowledgement

Our code is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Restormer](https://github.com/swz30/Restormer) and [LoRA-IR](https://github.com/shallowdream204/LoRA-IR). We thank to these works for their awesome backbones and code repos.

## Contact

If you have any questions, please contact [jblincs1996@gmail.com](mailto:jblincs1996@gmail.com).
