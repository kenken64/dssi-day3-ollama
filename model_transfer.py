#!/usr/bin/env python3
"""
Model Transfer Tool - Export and Import models for offline transfer between machines.

Packages the base model (from cache), fine-tuned adapter weights, and dataset
into a single zip file that can be transferred to another machine.

Usage:
    Export:  python model_transfer.py export [--output model_package.zip]
    Import:  python model_transfer.py import model_package.zip
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path


# Default paths
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_OUTPUT_ZIP = "qwen2.5-coder-0.5b-text-to-sql.zip"

# Known cache locations to search
MODELSCOPE_CACHE = Path.home() / ".cache" / "modelscope" / "hub" / "models"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def find_model_in_cache(model_name: str) -> tuple[Path | None, str]:
    """Find the model in ModelScope or HuggingFace cache directories."""
    org, name = model_name.split("/")

    # Check ModelScope cache (uses sanitized names with ___ for special chars)
    ms_name = name.replace(".", "___").replace("-", "-")
    ms_path = MODELSCOPE_CACHE / org / ms_name
    if ms_path.exists():
        return ms_path, "modelscope"

    # Also try exact name
    ms_path_exact = MODELSCOPE_CACHE / org / name
    if ms_path_exact.exists():
        return ms_path_exact, "modelscope"

    # Check HuggingFace cache
    hf_dir_name = f"models--{org}--{name}"
    hf_path = HF_CACHE / hf_dir_name
    if hf_path.exists():
        # HF cache uses snapshots - find the actual model files
        snapshots = hf_path / "snapshots"
        if snapshots.exists():
            # Use the most recent snapshot
            snapshot_dirs = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshot_dirs:
                return snapshot_dirs[0], "huggingface"

    return None, ""


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def add_directory_to_zip(zf: zipfile.ZipFile, dir_path: Path, archive_prefix: str):
    """Add all files in a directory to the zip under a given prefix."""
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file():
            arcname = f"{archive_prefix}/{file_path.relative_to(dir_path)}"
            print(f"  Adding: {arcname} ({format_size(file_path.stat().st_size)})")
            zf.write(file_path, arcname)


def do_export(args):
    """Export model, adapter, and dataset to a zip file."""
    project_dir = Path(args.project_dir)
    output_zip = Path(args.output)
    model_name = args.base_model

    print(f"=== Model Transfer - Export ===")
    print(f"Base model: {model_name}")
    print(f"Project dir: {project_dir}")
    print(f"Output: {output_zip}")
    print()

    # 1. Find base model in cache
    cache_path, cache_source = find_model_in_cache(model_name)
    if cache_path is None:
        print(f"ERROR: Base model '{model_name}' not found in cache.")
        print(f"  Checked: {MODELSCOPE_CACHE}")
        print(f"  Checked: {HF_CACHE}")
        print("  Run the training script first to download the model.")
        sys.exit(1)

    print(f"Found base model in {cache_source} cache: {cache_path}")
    print(f"  Size: {format_size(get_dir_size(cache_path))}")

    # 2. Check for fine-tuned adapter
    adapter_dir = project_dir / "text-to-sql-final"
    has_adapter = adapter_dir.exists() and (adapter_dir / "adapter_model.safetensors").exists()
    if has_adapter:
        print(f"Found fine-tuned adapter: {adapter_dir}")
        print(f"  Size: {format_size(get_dir_size(adapter_dir))}")
    else:
        print("No fine-tuned adapter found (text-to-sql-final/) - skipping")

    # 3. Check for Modelfile (Ollama config)
    modelfile_path = project_dir / "Modelfile"
    has_modelfile = modelfile_path.exists()
    if has_modelfile:
        print(f"Found Modelfile: {modelfile_path}")

    # 4. Check for GGUF export (if exists)
    gguf_files = list(project_dir.glob("*.gguf")) + list((project_dir / "text-to-sql-final").glob("*.gguf") if adapter_dir.exists() else [])
    if gguf_files:
        print(f"Found GGUF files: {[f.name for f in gguf_files]}")

    print()
    print("Packing zip file...")

    # Create manifest
    manifest = {
        "model_name": model_name,
        "cache_source": cache_source,
        "has_adapter": has_adapter,
        "has_modelfile": has_modelfile,
        "gguf_files": [f.name for f in gguf_files],
        "export_machine": os.uname().nodename,
    }

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        # Write manifest
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Add base model
        print(f"\n[1/4] Adding base model...")
        add_directory_to_zip(zf, cache_path, "base_model")

        # Add adapter if exists
        if has_adapter:
            print(f"\n[2/4] Adding fine-tuned adapter...")
            add_directory_to_zip(zf, adapter_dir, "adapter")
        else:
            print(f"\n[2/4] Skipping adapter (not found)")

        # Add Modelfile
        if has_modelfile:
            print(f"\n[3/4] Adding Modelfile...")
            zf.write(modelfile_path, "Modelfile")
        else:
            print(f"\n[3/4] Skipping Modelfile (not found)")

        # Add GGUF files
        if gguf_files:
            print(f"\n[4/4] Adding GGUF files...")
            for gf in gguf_files:
                print(f"  Adding: {gf.name} ({format_size(gf.stat().st_size)})")
                zf.write(gf, f"gguf/{gf.name}")
        else:
            print(f"\n[4/4] Skipping GGUF (not found)")

    final_size = output_zip.stat().st_size
    print(f"\n=== Export complete ===")
    print(f"Output: {output_zip}")
    print(f"Size: {format_size(final_size)}")
    print(f"\nTransfer this file to the target machine and run:")
    print(f"  python model_transfer.py import {output_zip}")


def do_import(args):
    """Import model, adapter, and dataset from a zip file."""
    input_zip = Path(args.zipfile)
    if not input_zip.exists():
        print(f"ERROR: File not found: {input_zip}")
        sys.exit(1)

    project_dir = Path(args.project_dir)
    print(f"=== Model Transfer - Import ===")
    print(f"Input: {input_zip} ({format_size(input_zip.stat().st_size)})")
    print(f"Project dir: {project_dir}")
    print()

    with zipfile.ZipFile(input_zip, "r") as zf:
        # Read manifest
        manifest = json.loads(zf.read("manifest.json"))
        model_name = manifest["model_name"]
        cache_source = manifest.get("cache_source", "modelscope")

        print(f"Model: {model_name}")
        print(f"Original cache: {cache_source}")
        print(f"Has adapter: {manifest['has_adapter']}")
        print(f"Has Modelfile: {manifest['has_modelfile']}")
        print(f"GGUF files: {manifest.get('gguf_files', [])}")
        print(f"Exported from: {manifest.get('export_machine', 'unknown')}")
        print()

        all_entries = zf.namelist()

        # 1. Restore base model to cache
        print("[1/4] Restoring base model to cache...")
        base_entries = [e for e in all_entries if e.startswith("base_model/")]

        if cache_source == "modelscope":
            org, name = model_name.split("/")
            ms_name = name.replace(".", "___").replace("-", "-")
            target_dir = MODELSCOPE_CACHE / org / ms_name
        else:
            org, name = model_name.split("/")
            # For HF, we restore into a simple snapshot dir
            hf_dir = HF_CACHE / f"models--{org}--{name}"
            target_dir = hf_dir / "snapshots" / "imported"

        target_dir.mkdir(parents=True, exist_ok=True)
        for entry in base_entries:
            if entry == "base_model/":
                continue
            rel_path = entry[len("base_model/"):]
            dest = target_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Extracting: {rel_path}")
            with zf.open(entry) as src, open(dest, "wb") as dst:
                dst.write(src.read())

        print(f"  Restored to: {target_dir}")

        # 2. Restore adapter
        if manifest["has_adapter"]:
            print("\n[2/4] Restoring fine-tuned adapter...")
            adapter_dir = project_dir / "text-to-sql-final"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            adapter_entries = [e for e in all_entries if e.startswith("adapter/")]
            for entry in adapter_entries:
                if entry == "adapter/":
                    continue
                rel_path = entry[len("adapter/"):]
                dest = adapter_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Extracting: {rel_path}")
                with zf.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            print(f"  Restored to: {adapter_dir}")
        else:
            print("\n[2/4] No adapter to restore")

        # 3. Restore Modelfile
        if manifest["has_modelfile"]:
            print("\n[3/4] Restoring Modelfile...")
            dest = project_dir / "Modelfile"
            with zf.open("Modelfile") as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  Restored to: {dest}")
        else:
            print("\n[3/4] No Modelfile to restore")

        # 4. Restore GGUF files
        gguf_files = manifest.get("gguf_files", [])
        if gguf_files:
            print("\n[4/4] Restoring GGUF files...")
            gguf_entries = [e for e in all_entries if e.startswith("gguf/")]
            for entry in gguf_entries:
                if entry == "gguf/":
                    continue
                rel_path = entry[len("gguf/"):]
                dest = project_dir / rel_path
                print(f"  Extracting: {rel_path}")
                with zf.open(entry) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
            print(f"  Restored to: {project_dir}")
        else:
            print("\n[4/4] No GGUF files to restore")

    print(f"\n=== Import complete ===")
    print(f"Base model restored to: {target_dir}")
    if manifest["has_adapter"]:
        print(f"Adapter restored to: {project_dir / 'text-to-sql-final'}")
    print(f"\nYou can now run the training script or deploy with Ollama.")
    if cache_source == "huggingface":
        print(f"\nNote: For HuggingFace cache, you may also need to set:")
        print(f"  export TRANSFORMERS_OFFLINE=1")
        print(f"  to prevent re-downloading.")


def main():
    parser = argparse.ArgumentParser(
        description="Export/Import model cache for offline transfer between machines"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to zip")
    export_parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT_ZIP,
        help=f"Output zip filename (default: {DEFAULT_OUTPUT_ZIP})"
    )
    export_parser.add_argument(
        "--base-model", default=DEFAULT_BASE_MODEL,
        help=f"Base model name (default: {DEFAULT_BASE_MODEL})"
    )
    export_parser.add_argument(
        "--project-dir", default=".",
        help="Project directory containing adapter/Modelfile (default: current dir)"
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import model from zip")
    import_parser.add_argument("zipfile", help="Path to the zip file to import")
    import_parser.add_argument(
        "--project-dir", default=".",
        help="Project directory to restore adapter/Modelfile into (default: current dir)"
    )

    args = parser.parse_args()

    if args.command == "export":
        do_export(args)
    elif args.command == "import":
        do_import(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
