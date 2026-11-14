"""Command-line interface for the benchmark (GPU modules are loaded lazily)."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="GeoGuessr-50k benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="benchmark project root")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("inventory", help="audit original dataset")
    commands.add_parser("split", help="create and lock fixed real splits")
    verify = commands.add_parser("verify", help="verify a published stage")
    commands.add_parser("report", help="recompute a report from locked test predictions")
    commands.add_parser("augment-freeze", help="validate and freeze the augmented dataset")
    commands.add_parser("augment", help="generate candidates until review or frozen membership")
    commands.add_parser("preflight", help="probe and freeze the CUDA environment")
    commands.add_parser("init", help="freeze shared initial weights for each seed")
    commands.add_parser("train-all", help="run the six paired training jobs")
    commands.add_parser("freeze-checkpoints", help="lock six best checkpoints")
    commands.add_parser("test-all", help="evaluate the six frozen checkpoints on the real test split")
    single = commands.add_parser("_train-run")
    single.add_argument("--condition", choices=("original", "augmented"), required=True)
    single.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    review_export = commands.add_parser("review-export", help="prepare an offline diffusion review bundle")
    review_export.add_argument("--output", type=Path, required=True)
    review_import = commands.add_parser("review-import", help="import hash-bound review decisions")
    review_import.add_argument("--input", type=Path, required=True)
    from .artifacts import STAGES
    verify.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from .inventory import build_inventory
            build_inventory(args.root)
        elif args.command == "split":
            from .split import build_split
            build_split(args.root)
        elif args.command == "verify":
            from .artifacts import verify_stage
            verify_stage(args.root, args.stage)
        elif args.command == "report":
            from .reporting import build_report
            build_report(args.root)
        elif args.command == "augment-freeze":
            from .augmentation.workflow import freeze_augmented
            freeze_augmented(args.root)
        elif args.command == "augment":
            from .augmentation.workflow import generate_offline
            state = generate_offline(args.root)
            print(f"augment: {state}")
            return 0
        elif args.command == "preflight":
            from .preflight import run_preflight
            run_preflight(args.root)
        elif args.command == "init":
            from .experiment import initialize_pairs
            initialize_pairs(args.root)
        elif args.command == "train-all":
            from .experiment import train_all
            train_all(args.root)
        elif args.command == "freeze-checkpoints":
            from .experiment import freeze_checkpoints
            freeze_checkpoints(args.root)
        elif args.command == "test-all":
            from .evaluate import evaluate_test_once
            evaluate_test_once(args.root)
        elif args.command == "_train-run":
            import os
            if os.environ.get("PYTHONHASHSEED") != str(args.seed) or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise ValueError("training subprocess must be launched with the frozen seed and CUDA settings")
            from .training import train_run
            train_run(args.root, args.condition, args.seed)
        elif args.command == "review-export":
            from .augmentation.review import export_review_queue
            export_review_queue(args.root, args.output)
        else:
            from .augmentation.review import import_review_decisions
            import_review_decisions(args.root, args.input)
        print(f"{args.command}: success")
        return 0
    except (OSError, ValueError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
