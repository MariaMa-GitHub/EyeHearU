# Sign-Speak vs EyeHearU benchmark

Compare Sign-Speak's commercial ASL recognition API against our I3D backend
on the **same** validation clips.

## Setup

```bash
pip install boto3 requests
```

1. Copy `.env.example` → `.env` and fill in your keys.
2. Run **Step 1** to pick 10 val clips from S3 and download them:
   ```bash
   python run_benchmark.py pick
   ```
3. Run **Step 2** to call Sign-Speak API on those clips:
   ```bash
   python run_benchmark.py sign-speak
   ```
4. Run **Step 3** to call our own backend API on the same clips:
   ```bash
   python run_benchmark.py ours
   ```
5. Run **Step 4** to compare:
   ```bash
   python run_benchmark.py compare
   ```

Or run everything at once:
```bash
python run_benchmark.py all
```

Results are saved in `results/`.
