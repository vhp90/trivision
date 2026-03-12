import os
import pathlib
import shlex
import subprocess
import sys
import base64
import urllib.parse
import urllib.error
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
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"+ {printable}")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PIP_PROGRESS_BAR", "on")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
    finally:
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _download_requirements(url, out_path, token=None):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url)
    if token:
        auth = base64.b64encode(f"{token}:".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(request) as resp:
        data = resp.read()
    out_path.write_bytes(data)
    return out_path


def _has_cached_wheels(wheelhouse):
    wheelhouse = pathlib.Path(wheelhouse)
    return wheelhouse.exists() and any(wheelhouse.glob("*.whl"))


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
    private_requirements_url = f"https://missinglink.build/{machine}.txt"

    pip_cache.mkdir(parents=True, exist_ok=True)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)

    print(f"Using pip cache: {pip_cache}")
    print(f"Using wheelhouse: {wheelhouse}")

    try:
        _download_requirements(private_requirements_url, requirements_path, token=token)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Failed to reach missinglink.build for the private wheel index. "
            "Double-check the token value and retry. "
            f"Original error: {exc}"
        ) from exc

    if _has_cached_wheels(wheelhouse):
        print("Using cached wheelhouse from Drive; skipping wheel download.")
    else:
        print("No cached wheels found yet; downloading the first-run wheel set to Drive.")
        # Keep all remote wheels in Drive so future Colab sessions can install locally.
        _run([
            sys.executable, "-u", "-m", "pip", "download",
            "--dest", str(wheelhouse),
            "--requirement", str(requirements_path),
            "--prefer-binary",
            "-v",
        ])
        _run([
            sys.executable, "-u", "-m", "pip", "download",
            "--dest", str(wheelhouse),
            "--prefer-binary",
            "-v",
            *extra_packages,
        ])

    # Install from Drive-backed wheelhouse first; pip cache remains on Drive as fallback.
    _run([
        sys.executable, "-u", "-m", "pip", "install",
        "--find-links", str(wheelhouse),
        "--requirement", str(requirements_path),
        "-v",
    ])
    _run([
        sys.executable, "-u", "-m", "pip", "install",
        "--find-links", str(wheelhouse),
        "-v",
        *extra_packages,
    ])

    return {
        "requirements_path": str(requirements_path),
        "pip_cache": str(pip_cache),
        "wheelhouse": str(wheelhouse),
    }
