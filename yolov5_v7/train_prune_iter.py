"""
Iterative pruning + fine-tuning for YOLOv5 using torch_pruning
- numpy<2 환경 전제 (pip install "numpy<2" --force-reinstall)
- DependencyGraph를 이용한 수동 pruning plan으로
  YOLOv5 C3/Concat/shortcut 구조까지 채널 감축 가능
"""

import argparse, os, sys, subprocess
import torch
import torch.nn as nn
import torch_pruning as tp
from utils.torch_utils import select_device


# -------------------- NumPy 버전 가드 --------------------
def check_numpy():
    import numpy as _np
    ver = tuple(map(int, _np.__version__.split(".")[:2]))
    if ver >= (2, 0):
        print(f"[ERROR] NumPy {_np.__version__} detected. Please install numpy<2")
        print("        Run: pip install 'numpy<2' --force-reinstall")
        sys.exit(1)


# -------------------- L1 기반 채널 선택 (tp v2용) --------------------
def l1_prune_indices(weight: torch.Tensor, amount: float):
    """
    weight: (out_channels, in_channels, kH, kW)
    amount: 0~1, 제거 비율
    return: 제거할 out_channel 인덱스 리스트 (작은 L1 norm 우선)
    """
    oc = weight.shape[0]
    prune_num = int(oc * float(amount))
    if prune_num < 1:
        return []
    norms = weight.abs().mean(dim=(1, 2, 3))  # 각 out_channel의 평균 L1
    idxs = norms.argsort()[:prune_num].tolist()
    return idxs


# -------------------- Validation Helper --------------------
def run_validation(data_yaml, weights, imgsz=640, device_str="0"):
    """YOLOv5 val.py subprocess 실행 후 mAP50 반환"""
    cmd = [
        sys.executable, "val.py",
        "--data", data_yaml,
        "--weights", weights,
        "--imgsz", str(imgsz),
        "--device", str(device_str),   # 문자열 그대로 전달
        "--batch-size", "16",
        "--iou-thres", "0.6",
        "--conf-thres", "0.001",
        "--verbose", "False",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    best = 0.0
    for line in out.stdout.splitlines():
        parts = line.strip().split()
        # 일반적으로: Class Images Instances P R mAP50 mAP50-95
        if len(parts) >= 6 and parts[0] == "all":
            try:
                val = float(parts[5])  # mAP50
                best = max(best, val)
            except Exception:
                pass
    return best


# -------------------- Checkpoint Helper --------------------
def save_checkpoint(model, path, cfg_yaml):
    """cfg yaml 포함해 YOLOv5 호환 체크포인트 저장"""
    ckpt = {
        "model": model.state_dict(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "epoch": -1,
        "best_fitness": 0.0,
        "training_results": None,
        "wandb_id": None,
        "yaml": cfg_yaml,
    }
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


# -------------------- Debug Utils --------------------
def describe_ignored_layers(ignored):
    print(f"[DEBUG] Ignored layers ({len(ignored)}):")
    for i, m in enumerate(ignored[:20]):
        print(f"  - {i:02d}: {m.__class__.__name__}")
    if len(ignored) > 20:
        print(f"  ... and {len(ignored)-20} more")


def list_prunable_convs(model, ignored_set):
    convs = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m not in ignored_set:
            convs.append((name, m.out_channels))
    if not convs:
        print("[WARN] No prunable Conv2d found (after ignored filtering).")
    else:
        print(f"[DEBUG] Prunable Conv2d count: {len(convs)}")
        for name, ch in convs[:30]:
            print(f"  - {name} (out_channels={ch})")
        if len(convs) > 30:
            print(f"  ... and {len(convs)-30} more")
    return convs


# -------------------- Main Pruning Routine --------------------
def iterative_prune(
    weights,
    data,
    imgsz=640,
    target_prune_rate=0.5,
    iterative_steps=4,
    max_map_drop=0.2,
    fine_tune_epochs=10,
    device="0",
    cfg_yaml="models/yolov5s.yaml",
    per_step_override=None
):
    check_numpy()

    # subprocess용 문자열 보존 + torch용 device 객체 분리
    device_str = str(device)
    torch_device = select_device(device_str)

    os.makedirs("runs/prune", exist_ok=True)

    print("Loading model…")
    ckpt = torch.load(weights, map_location=torch_device)
    model = ckpt["ema"] if ckpt.get("ema") else ckpt["model"]
    model = model.float().to(torch_device).eval()

    example_inputs = torch.randn(1, 3, imgsz, imgsz).to(torch_device)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"[Original] MACs: {base_macs/1e9:.3f}G, Params: {base_params/1e6:.3f}M")

    init_map = run_validation(data, weights, imgsz, device_str)
    print(f"[Original] mAP50: {init_map:.4f}")

    # per-step ratio
    if per_step_override is not None:
        step_ratio = float(per_step_override)
    else:
        step_ratio = 1 - (1 - float(target_prune_rate)) ** (1.0 / int(iterative_steps))
    step_ratio = max(1e-4, min(0.5, step_ratio))
    print(f"Per-step pruning ratio: {step_ratio:.3f}")

    # ---- Ignore layers: Detect + Concat + Upsample ----
    from models.yolo import Detect
    from models.common import Concat
    ignored = []
    for m in model.modules():
        if isinstance(m, (Detect, Concat, nn.Upsample)):
            ignored.append(m)
    describe_ignored_layers(ignored)
    ignored_set = set(ignored)

    list_prunable_convs(model, ignored_set)

    # -------------------- Iterative Steps --------------------
    for step in range(iterative_steps):
        print(f"\n{'='*25} STEP {step+1}/{iterative_steps} {'='*25}")
        model.train()

        pre_macs, pre_params = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"Before prune: {pre_macs/1e9:.3f}G MACs, {pre_params/1e6:.3f}M params")

        # ---- DependencyGraph 기반 수동 pruning ----
        DG = tp.DependencyGraph().build_dependency(model, example_inputs)
        total_reduced = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and m not in ignored_set:
                # (선택) 너무 작은 레이어는 스킵: out_channels < 8 등
                if getattr(m, "out_channels", 0) <= 8:
                    continue
                idxs = l1_prune_indices(m.weight, amount=step_ratio)
                if idxs:
                    try:
                        plan = DG.get_pruning_plan(m, tp.prune_conv, idxs)
                        print(f"[Plan] {name}: prune {len(idxs)} channels")
                        plan.exec()
                        total_reduced += len(idxs)
                    except Exception as e:
                        print(f"[SKIP] {name} plan failed: {e}")

        if total_reduced == 0:
            print("[WARN] No channels pruned in this step — dependency constraints may block pruning.")
        model.eval()

        post_macs, post_params = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"After prune : {post_macs/1e9:.3f}G MACs, {post_params/1e6:.3f}M params")

        # ---- Save & Validate ----
        step_path = f"runs/prune/step_{step+1}_pruned.pt"
        save_checkpoint(model, step_path, cfg_yaml)
        pruned_map = run_validation(data, step_path, imgsz, device_str)
        print(f"After prune mAP50: {pruned_map:.4f}")

        # ---- Fine-tuning ----
        if fine_tune_epochs > 0:
            print(f"Fine-tuning {fine_tune_epochs} epochs…")
            ft_cmd = [
                sys.executable, "train.py",
                "--data", data,
                "--weights", step_path,
                "--epochs", str(fine_tune_epochs),
                "--imgsz", str(imgsz),
                "--device", device_str,
                "--project", "runs/prune",
                "--name", f"finetune_step_{step+1}",
                "--exist-ok",
            ]
            subprocess.run(ft_cmd, check=False)

        if init_map > 0 and pruned_map > 0 and (init_map - pruned_map) > max_map_drop:
            print("Early stop: mAP drop exceeded threshold.")
            break

        list_prunable_convs(model, ignored_set)

    # -------------------- Save Final --------------------
    final_path = "runs/prune/final_pruned.pt"
    save_checkpoint(model, final_path, cfg_yaml)
    final_macs, final_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("\n========== PRUNING COMPLETED ==========")
    print(f"Original : {base_macs/1e9:.3f}G MACs, {base_params/1e6:.3f}M params")
    print(f"Final    : {final_macs/1e9:.3f}G MACs, {final_params/1e6:.3f}M params")
    print(f"Reduction: MACs {(1-final_macs/base_macs)*100:.1f}%, "
          f"Params {(1-final_params/base_params)*100:.1f}%")
    print(f"Final model saved at {final_path}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--target-prune-rate", type=float, default=0.5)
    parser.add_argument("--iterative-steps", type=int, default=4)
    parser.add_argument("--max-map-drop", type=float, default=0.2)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cfg-yaml", default="models/yolov5s.yaml",
                        help="original model yaml used for training")
    parser.add_argument("--per-step-override", type=float, default=None,
                        help="set fixed per-step ratio (e.g., 0.1)")
    opt = parser.parse_args()

    iterative_prune(
        weights=opt.weights,
        data=opt.data,
        imgsz=opt.imgsz,
        target_prune_rate=opt.target_prune_rate,
        iterative_steps=opt.iterative_steps,
        max_map_drop=opt.max_map_drop,
        fine_tune_epochs=opt.fine_tune_epochs,
        device=opt.device,
        cfg_yaml=opt.cfg_yaml,
        per_step_override=opt.per_step_override,
    )
