# Othello-GPT Cloud Run Instructions

## 1. Transfer to Cloud
Upload the entire `clean_audit` directory to your cloud instance (e.g., via `scp` or `rsync`).

```bash
# Example
scp -r clean_audit user@gpu-instance:~/
```

## 2. Run Training
SSH into the machine and run the setup script:

```bash
cd clean_audit
bash run_cloud.sh
```

## 3. What This Does
1. Installs minimal dependencies (`torch`, `numpy`, `tqdm`)
2. Checks for A100 GPU
3. Runs training with **Batch Size 2048** and **TF32** optimization
4. Saves checkpoints to `clean_audit/data/`

## 4. Expected Performance
- **A100 Time**: ~15-20 minutes for 25K steps (vs ~4 hours on CPU/small GPU)
- **Accuracy**: Should hit >99%
