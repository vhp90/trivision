import os
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request


DEFAULT_EXTRA_PACKAGES = [
    "trimesh",
    "pygltflib",
    "plyfile",
    "moderngl",
    "huggingface_hub",
    "kornia",
    "kornia-rs",
]


def _run(cmd):
    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}")
    subprocess.check_call(cmd)


def _download_requirements(url, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return out_path


def setup_trivision(
    token,
    machine="a100",
    drive_root="/content/drive/MyDrive/TriVision/cache",
    extra_packages=None,
):
    token = str(token).strip()
    if not token or set(token) == {"*"}:
        raise ValueError(
            "Set your missinglink.build token before running setup_trivision()."
        )

    machine = str(machine).strip().lower()
    extra_packages = list(extra_packages or DEFAULT_EXTRA_PACKAGES)

    drive_root = pathlib.Path(drive_root)
    pip_cache = drive_root / "pip_cache"
    wheelhouse = drive_root / "wheelhouse" / machine
    requirements_path = wheelhouse / f"{machine}.txt"
    encoded_token = urllib.parse.quote(token, safe="")
    private_requirements_url = f"https://{encoded_token}@missinglink.build/{machine}.txt"

    pip_cache.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)

    print(f"Using pip cache: {pip_cache}")
    print(f"Using wheelhouse: {wheelhouse}")

    try:
        _download_requirements(private_requirements_url, requirements_path)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach missinglink.build for the private wheel index. "
            "Double-check the token value and retry. "
            f"Original error: {exc}"
        ) from exc

    # Keep all remote wheels in Drive so future Colab sessions can install locally.
    _run([
        sys.executable, "-m", "pip", "download",
        "--dest", str(wheelhouse),
        "--requirement", str(requirements_path),
        "--prefer-binary",
    ])
    _run([
        sys.executable, "-m", "pip", "download",
        "--dest", str(wheelhouse),
        "--prefer-binary",
        *extra_packages,
    ])

    # Install from Drive-backed wheelhouse first; pip cache remains on Drive as fallback.
    _run([
        sys.executable, "-m", "pip", "install",
        "--find-links", str(wheelhouse),
        "--requirement", str(requirements_path),
    ])
    _run([
        sys.executable, "-m", "pip", "install",
        "--find-links", str(wheelhouse),
        *extra_packages,
    ])

    return {
        "requirements_path": str(requirements_path),
        "pip_cache": str(pip_cache),
        "wheelhouse": str(wheelhouse),
    }
