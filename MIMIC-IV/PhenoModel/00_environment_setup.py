import argparse
import importlib
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT)) 

REQUIRED = {
    "torch":             ("torch", "PyTorch"),
    "torchvision":       ("torchvision", "TorchVision"),
    "transformers":      ("transformers", "Transformers"),
    "pandas":            ("pandas", "pandas"),
    "pyarrow":           ("pyarrow", "pyarrow"),
    "Pillow":            ("PIL", "Pillow"),
    "tqdm":              ("tqdm", "tqdm"),
    "scikit-learn":      ("sklearn", "scikit-learn"),
    "pyyaml":            ("yaml", "PyYAML"),
}

def _print_header(msg: str) -> None:
    bar = "=" * (len(msg) + 2)
    print(f"\n{bar}\n {msg}\n{bar}")

def check_or_install(pip_name: str, import_name: str, label: str, install: bool) -> Tuple[bool, Optional[str]]:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[ok] {label:>14}   import={import_name:<15}  version={ver}")
        return True, ver
    except Exception:
        print(f"[miss] {label:>14}   import={import_name:<15}  (pip: {pip_name})")
        if not install:
            return False, None
        print(f"[act] Installing {pip_name} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"[ok] Installed {label}  version={ver}")
            return True, ver
        except subprocess.CalledProcessError as e:
            print(f"[err] pip install failed for {pip_name}: {e}")
        except Exception as e:
            print(f"[err] import still failing after install for {pip_name}: {e}")
        return False, None

def verify_environment(install_missing: bool) -> None:
    _print_header("Verifying Python packages")
    ok_all = True
    for pip_name, (import_name, label) in REQUIRED.items():
        ok, _ = check_or_install(pip_name, import_name, label, install_missing)
        ok_all &= ok
    if not ok_all:
        print("\n[warn] Some requirements are missing. Re-run with --install-missing or install manually.")
    else:
        print("\n[ok] All required packages available.")

def cuda_amp_checks(seed: int) -> None:
    _print_header("CUDA / AMP checks")
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            cc = torch.cuda.get_device_capability(idx)
            print(f"GPU[{idx}]: {name}  compute_capability={cc}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}  version: {torch.backends.cudnn.version()}")
        else:
            dev = torch.device("cpu")
            print("Running CPU-only.")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        dtype = None
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if hasattr(torch, "bfloat16") else None

        if dtype is not None:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Testing autocast with dtype={dtype} on device_type={device_type} ...")
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=True):
                a = torch.randn(256, 256, device=dev)
                b = torch.randn(256, 256, device=dev)
                c = a @ b
            print(f"[ok] autocast matmul succeeded, c.shape={tuple(c.shape)}")
        else:
            print("[info] Skipping AMP test (no fp16/bf16 available).")
    except Exception as e:
        print(f"[err] CUDA/AMP check failed: {e}")

def ensure_project_tree(make_dirs: bool) -> None:
    _print_header("Project directories")
    data_dir = REPO_ROOT / "data" / "MIMIC-IV"
    ckpt_dir = REPO_ROOT / "checkpoints"
    if make_dirs:
        data_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ok] ensured {data_dir}")
        print(f"[ok] ensured {ckpt_dir}")
    else:
        print(f"[skip] directory creation disabled -- found? data={data_dir.exists()}  ckpts={ckpt_dir.exists()}")

def load_and_show_cfg() -> None:
    _print_header("Loading config from Model/env_config.py")
    cfg = None
    try:
        env_cfg = importlib.import_module("Model.env_config")
        cfg = env_cfg.load_cfg(yaml_path=None, overrides=None)
        try:
            from dataclasses import asdict
            cfg_json = json.dumps(asdict(cfg), indent=2)
        except Exception:
            cfg_json = str(cfg)
        print("\n[CFG dump]")
        print(cfg_json)

        print("\n[key settings]")
        print(f"- task_name: {getattr(cfg, 'task_name', 'N/A')}")
        print(f"- d (hidden dim): {getattr(cfg, 'd', 'N/A')}")
        print(f"- structured_n_feats: {getattr(cfg, 'structured_n_feats', 'N/A')}  "
              f"structured_seq_len: {getattr(cfg, 'structured_seq_len', 'N/A')}")
        print(f"- text_model_name: {getattr(cfg, 'text_model_name', 'N/A')}")
        print(f"- image_model_name: {getattr(cfg, 'image_model_name', 'N/A')}")
        print(f"- routing_backend: {getattr(cfg, 'routing_backend', 'N/A')}  use_gates: {getattr(cfg, 'use_gates', 'N/A')}")
    except ModuleNotFoundError as e:
        print("[err] Could not import Model.env_config. Make sure you're running at the repo root.")
        print(f"      {e}")
    except Exception as e:
        print(f"[err] Loading config failed: {e}")

def smoke_test_src_imports() -> None:
    _print_header("Smoke test: src model imports")
    tried = []
    ok = False
    for mod_name, cls_names in [
        ("src.capsule_model_pheno", ["CapsModelPheno", "CapsuleModelPheno"]),
        ("src.capsule_model",       ["CapsModel"]),
    ]:
        tried.append(mod_name)
        try:
            mod = importlib.import_module(mod_name)
            found = [c for c in cls_names if hasattr(mod, c)]
            if found:
                print(f"[ok] imported {mod_name} – found class(es): {', '.join(found)}")
                ok = True
                break
            else:
                print(f"[warn] imported {mod_name} but expected class not found ({', '.join(cls_names)})")
        except ModuleNotFoundError:
            print(f"[miss] {mod_name} (not present yet)")
        except Exception as e:
            print(f"[err] import failed for {mod_name}: {e}")
    if not ok:
        print(f"[info] Tried: {', '.join(tried)}. This is fine if you haven’t added the phenotyping model yet.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-missing", action="store_true",
                        help="Attempt to pip install any missing packages into the current environment.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNGs (default: 42).")
    parser.add_argument("--no-make-dirs", action="store_true", help="Do not create ./data/MIMIC-IV or ./checkpoints.")
    args = parser.parse_args()

    verify_environment(install_missing=args.install_missing)
    cuda_amp_checks(seed=args.seed)
    ensure_project_tree(make_dirs=not args.no_make_dirs)

    try:
        env_cfg = importlib.import_module("Model.env_config")
        if hasattr(env_cfg, "set_global_seed"):
            env_cfg.set_global_seed(args.seed)
            print(f"[ok] set_global_seed({args.seed}) via Model.env_config")
        if hasattr(env_cfg, "set_deterministic"):
            env_cfg.set_deterministic(False)
    except Exception:
        pass  

    load_and_show_cfg()
    smoke_test_src_imports()

    print("\n[done] Environment looks good. You’re ready to run data/build & training scripts.")

if __name__ == "__main__":
    main()
