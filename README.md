# UniRestorer++: Scalable Universal Image Restoration with Multi-Granularity Degradation Modeling

[Jingbo Lin](https://scholar.google.com/citations?user=zPycW30AAAAJ&hl=zh-CN&oi=ao), [Zhilu Zhang](https://scholar.google.com/citations?user=8pIq2N0AAAAJ&hl=zh-CN&oi=ao), 
[Fan Li](),
[Renjing Pei](),
[Wenbo Li](https://scholar.google.com/citations?user=foGn_TIAAAAJ&hl=zh-CN&oi=ao), 
[Hang Xu](https://scholar.google.com/citations?user=J_8TX6sAAAAJ&hl=zh-CN&oi=ao),
[Hongzhi Zhang](https://scholar.google.com/citations?user=Ysk4WBwAAAAJ&hl=zh-CN&oi=ao), and [Wangmeng Zuo](https://scholar.google.com/citations?user=rUOpCEYAAAAJ&hl=zh-CN&oi=ao)

## Requirements

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Training & Evaluation

1. Prepare the frozen text caption bank.

   ```bash
   python tools.prepare_caption_bank.py
   ```

2. Train the dense degradation representation encoder.

   ```bash
   python tools.train_dr.py
   ```

3. Extract dense degradation features and per-image GAP vectors.

   ```bash
   python tools.extract_dr.py
   ```

4. Build the hierarchical degradation clusters.

   ```bash
   python tools.hierarchical_clustering.py
   ```

5. Train the Restormer base on all paired training data.

   ```bash
   python tools.train_base.py
   ```

6. Train one LoRA expert per non-root cluster on the frozen shared base.

   ```bash
   python tools.train_cluster_lora.py
   ```

7. Assemble the base, cluster LoRA experts and DR encoder into a restoration checkpoint.

   ```bash
   python tools.assemble_cluster_lora.py
   ```

8. Initialize DER with CE on true-leaf expert features; variance has no CE gradient.

   ```bash
   python tools.train_der.py
   ```

9. Jointly train DER, GER, both phi heads and LoRA with restoration, DG and level-balance losses.

   ```bash
   python tools.train_joint.py
   ```

## Acknowledgement

Our code is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Restormer](https://github.com/swz30/Restormer) and [LoRA-IR](https://github.com/shallowdream204/LoRA-IR). We thank to these works for their awesome backbones and code repos.

## Contact
If you have any questions, please contact [jblincs1996@gmail.com](jblincs1996@gmail.com).