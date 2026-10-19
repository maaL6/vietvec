# VietVec

Word2Vec for Vietnamese, written in PyTorch (Skip-gram + Negative Sampling) for NLP class. Trained on Vietnamese Wikipedia: 160,644 words, 300-dim vectors.

Results: `máy_tính` (computer) sits next to `phần_cứng` (hardware) and `phần_mềm` (software); `việt_nam : hà_nội = nhật_bản : tokyo`; country → capital analogies are 59% correct.

```bash
pip install -r requirements.txt && python train.py         # train (needs data in data/processed/)
python scripts/visualize_pairs.py && open web/index.html    # explore 20 word pairs in 2D/3D
```

The corpus and model are several GB, so they are not in git. MIT license.
