# Testing NaVILA Inference (Single GPU, Quantized)

This guide runs the NaVILA checkpoint on a single video or folder of frames plus a navigation instruction, and prints the model's next action. It does **not** need Habitat or VLN-CE (for the full benchmark, see the Evaluation section of the main README).

## 1. Hardware & quantization

The checkpoint ([a8cheng/navila-llama3-8b-8f](https://huggingface.co/a8cheng/navila-llama3-8b-8f)) is a Llama-3-8B LLM plus a SigLIP vision tower.

| Mode | Flag | Approx. VRAM (8 frames) | Notes |
|------|------|-------------------------|-------|
| fp16 | _(none)_ | ~17 GB | Needs a 24 GB GPU |
| 8-bit | `--load-8bit` | ~10–11 GB | Tight on 12 GB cards; may OOM |
| 4-bit NF4 | `--load-4bit` | ~6–7 GB | **Recommended on ≤12 GB GPUs** (e.g. RTX 4070) |

Only the LLM is quantized (bitsandbytes); the vision tower and projector stay in fp16. Expect a small accuracy drop compared to fp16.

You also need ~16 GB of disk for the checkpoint and ~15 GB for the conda env.

## 2. Environment (conda)

From the repo root:

```bash
./environment_setup.sh navila
conda activate navila
```

The script creates a Python 3.10 env and installs:
- torch 2.3.0
- the FlashAttention 2.5.8 (cu122) wheel
- this repo in editable mode
- transformers 4.37.2

It also copies `llava/train/transformers_replace/*` over the installed transformers. That patch is required: the model depends on the modified Llama code. If you ever reinstall transformers, re-run the two `cp` lines at the end of `environment_setup.sh`.

Sanity check:

```bash
python -c "import torch, bitsandbytes, flash_attn; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Notes:
- A newer NVIDIA driver (e.g. CUDA 13.x) runs these CUDA 12 wheels fine; Ada GPUs (sm_89) are supported by FlashAttention 2.
- If bitsandbytes 0.41.0 can't find its CUDA library, or errors on 4-bit, upgrade it: `pip install bitsandbytes==0.43.1`.

## 3. Download the checkpoint

```bash
huggingface-cli download a8cheng/navila-llama3-8b-8f --local-dir checkpoints/navila-llama3-8b-8f
```

## 4. Run inference

```bash
python llava/eval/run_navigation.py \
    --model-path checkpoints/navila-llama3-8b-8f \
    --video-file path/to/clip.mp4 \
    --query "Walk down the hallway, turn left into the kitchen, and stop next to the table." \
    --load-4bit
```

Input options. In every case the model sees 8 frames: 7 sampled uniformly from the history plus the latest frame as the "current observation".
- `--video-file clip.mp4`: frames are sampled from the video. A short (5–20 s), forward-facing phone video of a hallway or room works well.
- `--video-file frames_dir/`: every image in the folder, sorted by filename. Use zero-padded names (`0001.jpg`, `0002.jpg`, …) so they sort in order. The folder is then sampled down to 8 frames, and padded with black frames if it has fewer.
- `--image-file a.jpg,b.jpg,...`: an explicit comma-separated list, treated the same way as a folder.

Expected output is a single high-level action, for example:

```
The next action is turn left 30 degrees.
```

or `move forward 75 cm` / `stop`. The first run spends most of its time loading and quantizing the weights (about a minute); generation itself takes a few seconds.

Watch memory in a second terminal with `watch -n 1 nvidia-smi`.

## 5. Troubleshooting

- **`torch.cuda.OutOfMemoryError`**: use `--load-4bit`, and close other GPU-heavy apps such as browsers with hardware acceleration.
- **`KeyError: 'torch_dtype'` when quantizing**: you're running an old `llava/model/builder.py`. The fix on this branch always sets `torch_dtype`.
- **Errors from inside `transformers/models/llama/` or `modeling_utils.py`**: check that `pip show transformers` reports 4.37.2 and that the patch was applied. Re-run the `cp` lines from `environment_setup.sh`.
- **`ModuleNotFoundError: llava`**: run from the repo root after `pip install -e .`.
- **Output is gibberish or unrelated to navigation**: check that `--num-video-frames` is 8 (the default) and that frames are in temporal order.
