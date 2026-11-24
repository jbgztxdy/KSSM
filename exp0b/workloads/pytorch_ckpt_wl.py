import os, sys, time, random, threading
import torch
import torch.nn as nn

base = sys.argv[1]
dur = int(sys.argv[2]) if len(sys.argv) > 2 else 65
os.makedirs(base, exist_ok=True)
dir_staging = os.path.join(base, "staging")
dir_steps   = os.path.join(base, "step_ckpts")
dir_best    = os.path.join(base, "best_ckpts")
dir_data    = os.path.join(base, "dataset")
for d in (dir_staging, dir_steps, dir_best, dir_data):
    os.makedirs(d, exist_ok=True)

# Pre-create dataset shards (large one-shot streaming files) & hot vocab/index file
vocab_path = os.path.join(dir_data, "vocab_hot.bin")
with open(vocab_path, "wb") as f:
    f.write(os.urandom(512 * 1024))
shard_paths = []
for i in range(16):
    sp = os.path.join(dir_data, f"train_shard_{i:02d}.bin")
    with open(sp, "wb") as f:
        f.write(os.urandom(8 * 1024 * 1024)) # 8 MB per shard
    shard_paths.append(sp)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(512, 1024), nn.GELU(),
    nn.Linear(1024, 1024), nn.GELU(),
    nn.Linear(1024, 512),
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

t_start = time.time()
rolling_ckpts = []
step = 0

while time.time() - t_start < dur:
    step += 1
    # 1. Read hot vocab/index repeatedly (Hot Reusable)
    with open(vocab_path, "rb") as f:
        for _ in range(4):
            f.seek(random.randint(0, 64) * 4096)
            _ = f.read(16384)

    # 2. Stream a training shard sequentially (One-Shot Scan)
    sp = shard_paths[(step - 1) % len(shard_paths)]
    with open(sp, "rb") as f:
        while True:
            blk = f.read(262144)
            if not blk:
                break

    # 3. Real GPU 3 forward/backward step
    x = torch.randn(64, 512, device=device)
    y = model(x)
    loss = (y ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    # 4. Short-lived (<10s) staging / optimizer scratch dump (e.g. grad bucket / temp metric)
    tmp_p = os.path.join(dir_staging, f"grad_stage_{step}.pt")
    torch.save({"step": step, "buf": torch.randn(512, 512)}, tmp_p)
    # Read back for verification & delete quickly (2-5s lifetime)
    if step > 2:
        old_tmp = os.path.join(dir_staging, f"grad_stage_{step - 2}.pt")
        if os.path.exists(old_tmp):
            with open(old_tmp, "rb") as f: _ = f.read(65536)
            os.unlink(old_tmp)

    # 5. Rolling step checkpoint (medium-lived: evicted after ~16-22s via save_total_limit)
    step_p = os.path.join(dir_steps, f"ckpt_step_{step:04d}.pt")
    torch.save({"step": step, "model": model.state_dict()}, step_p)
    rolling_ckpts.append((time.time(), step_p))
    now = time.time()
    while rolling_ckpts and (now - rolling_ckpts[0][0]) >= 16.0:
        _, old_ckpt = rolling_ckpts.pop(0)
        if os.path.exists(old_ckpt):
            os.unlink(old_ckpt)

    # 6. Durable best checkpoint every 5 steps (long-lived: same size as step_ckpt,
    #    preceded by validation eval read + fsync!)
    if step % 5 == 0:
        # Realistic eval phase: re-read vocab + validation slice before writing best_ckpt
        with open(vocab_path, "rb") as f:
            _ = f.read()
        best_p = os.path.join(dir_best, f"best_epoch_{step:04d}.pt")
        tmp_best = best_p + ".tmp"
        torch.save({"step": step, "model": model.state_dict()}, tmp_best)
        with open(tmp_best, "rb+") as f:
            os.fsync(f.fileno())
        os.rename(tmp_best, best_p)

    time.sleep(0.6)

print(f"PyTorch workload completed ({step} steps on {device})")
