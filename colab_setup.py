import os
import pathlib
import shlex
import subprocess
import sys
import urllib.parse


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
    private_requirements_url = f"https://missinglink.build/{machine}.txt"
    encoded_token = urllib.parse.quote(token, safe="")
    authenticated_requirements_url = f"https://{encoded_token}@missinglink.build/{machine}.txt"

    pip_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PIP_CACHE_DIR"] = str(pip_cache)

    print(f"Using pip cache: {pip_cache}")
    print("Installing MissingLink wheel set with Drive-backed pip cache.")
    _run([
        sys.executable, "-u", "-m", "pip", "install",
        "--cache-dir", str(pip_cache),
        "--no-deps",
        "--requirement", authenticated_requirements_url,
        "-v",
    ])
    _run([
        sys.executable, "-u", "-m", "pip", "install",
        "--cache-dir", str(pip_cache),
        "-v",
        *extra_packages,
    ])

    return {
        "pip_cache": str(pip_cache),
        "requirements_url": private_requirements_url,
    }
