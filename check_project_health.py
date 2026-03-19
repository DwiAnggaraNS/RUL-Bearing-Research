import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


def check_imports(modules):
    results = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            results.append((mod, True, "ok"))
        except Exception as exc:
            results.append((mod, False, str(exc)))
    return results


def check_gpu():
    try:
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return True, f"CUDA available (device {idx}: {name})"
        return False, "CUDA not available"
    except Exception as exc:
        return False, f"torch/cuda check failed: {exc}"


def check_files(dataset_path: Path):
    expected = [
        # Logs
        "raw_dual_model_training_log.csv",
        "raw_single_model_training_log.csv",
        "fft_dual_model_training_log.csv",
        "fft_single_model_training_log.csv",
        "raw_dual_gru_training_log.csv",
        "raw_single_gru_training_log.csv",
        "fft_dual_gru_training_log.csv",
        "fft_single_gru_training_log.csv",
        # Prediction results
        "raw_dual_lstm_results.csv",
        "raw_single_lstm_results.csv",
        "fft_dual_lstm_results.csv",
        "fft_single_lstm_results.csv",
        "raw_dual_gru_results.csv",
        "raw_single_gru_results.csv",
        "fft_dual_gru_results.csv",
        "fft_single_gru_results.csv",
        # Evaluation metrics exported by training notebooks
        "raw_dual_lstm_eval_metrics.csv",
        "raw_single_lstm_eval_metrics.csv",
        "fft_dual_lstm_eval_metrics.csv",
        "fft_single_lstm_eval_metrics.csv",
        "raw_dual_gru_eval_metrics.csv",
        "raw_single_gru_eval_metrics.csv",
        "fft_dual_gru_eval_metrics.csv",
        "fft_single_gru_eval_metrics.csv",
    ]

    status = []
    for name in expected:
        path = dataset_path / name
        status.append((name, path.exists()))
    return status


def main():
    parser = argparse.ArgumentParser(description="Check project environment and dataset outputs")
    parser.add_argument(
        "--dataset-path",
        default="/home/praktikan/projects/github/DwiAnggara/ProyekRisetBearing",
        help="Path to the datasets/output folder used by training and summary notebooks",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    print("=" * 80)
    print("PROJECT HEALTH CHECK")
    print("=" * 80)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"Working Dir : {os.getcwd()}")
    print(f"Dataset Path: {dataset_path}")
    print()

    # 1) Dependency checks
    modules = [
        "numpy",
        "pandas",
        "dask",
        "pyarrow",
        "torch",
        "matplotlib",
        "seaborn",
        "plotly",
        "sklearn",
        "tqdm",
    ]
    print("[1] Python package imports")
    import_results = check_imports(modules)
    for mod, ok, msg in import_results:
        print(f"- {mod:<12}: {'OK' if ok else 'FAIL'}{'' if ok else f' ({msg})'}")
    print()

    # 2) GPU check
    print("[2] GPU (CUDA) availability")
    gpu_ok, gpu_msg = check_gpu()
    print(f"- torch.cuda : {'OK' if gpu_ok else 'INFO'} ({gpu_msg})")
    print()

    # 3) Dataset folder check
    print("[3] Dataset/output folder")
    if dataset_path.exists() and dataset_path.is_dir():
        print("- folder     : OK")
    else:
        print("- folder     : FAIL (path does not exist or is not a directory)")
        print("=" * 80)
        return
    print()

    # 4) Expected output files check
    print("[4] Training outputs expected by summary")
    file_status = check_files(dataset_path)
    missing = []
    for name, ok in file_status:
        print(f"- {name:<40}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(name)
    print()

    print("[5] Final status")
    import_fail = [m for m, ok, _ in import_results if not ok]
    if import_fail:
        print(f"- Missing Python packages: {', '.join(import_fail)}")
    else:
        print("- Python packages: OK")

    if missing:
        print(f"- Missing training outputs: {len(missing)} file(s)")
    else:
        print("- Training outputs for summary: OK")

    print("=" * 80)


if __name__ == "__main__":
    main()
