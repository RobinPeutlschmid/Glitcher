#!/usr/bin/env python3
"""
build_portable.py - package Glitcher as a portable app (no Python needed to run it).

Run it on the system you want the app for (PyInstaller cannot cross-build: build on Windows
for Windows, on a Mac for macOS), from the folder with glitch.py, glitch_gui.py and
estimate_pose.py, using the Python that already runs the GUI:

    python build_portable.py              # full build, pose estimation included (~600 MB)
    python build_portable.py --no-pose    # without MediaPipe: much smaller, Pose tab can't estimate
    python build_portable.py --onefile    # one file instead of a folder (starts slower)

The native engine (engine/, Rust): its library is bundled, so the app renders with it. If
engine/target/release has no library yet and Rust's cargo is installed, it is built first;
--no-engine leaves it out (the app then renders with the Python renderer, ~10x slower).

Result: dist/Glitcher/ (or Glitcher.app on macOS) plus a .zip named with the version, system
and CPU, e.g. Glitcher-1.3.0-windows-x64.zip ("-nopose" is added for --no-pose builds). Copy or
unzip that anywhere and start Glitcher. The same program also works as the command-line tools:

    Glitcher --glitch frames/ -o out/ ...        (same options as glitch.py)
    Glitcher --estimate-pose frames/ -o pose.csv (same options as estimate_pose.py)
    Glitcher --estimate-pose --check             (exit code 0 = pose estimation works)

The pose model: put pose_landmarker_heavy.task (or full / lite) next to this script, or have
it in ~/.cache/estimate_pose/ from an earlier run, and it is bundled. Without it the app
downloads the model the first time the Pose tab estimates, or finds one you place next to
the app.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "Glitcher"


CV2_PLACEHOLDER = '''"""PyInstaller runtime hook written by build_portable.py: a placeholder "cv2" module.

MediaPipe's vision package imports OpenCV at import time, but only its drawing helpers use it,
and Glitcher never calls them. The packaged app leaves OpenCV out; this stands in for it so the
import succeeds. Anything that really asks OpenCV for something gets a clear error.
"""
import sys
import types


def _missing(name):
    raise AttributeError(f"cv2.{name}: OpenCV is not included in the packaged Glitcher")


if "cv2" not in sys.modules:
    _m = types.ModuleType("cv2", "placeholder: OpenCV is not included in the packaged Glitcher")
    _m.__getattr__ = _missing
    sys.modules["cv2"] = _m
'''


def arch() -> str:
    """CPU architecture for the zip name: x64 or arm64 (what the built app runs on)."""
    m = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(m, m or "unknown")


def engine_library(rebuild: bool = False):
    """The native engine's library, built with cargo if it's missing (or rebuild); None on error."""
    sys.path.insert(0, str(HERE))
    import glitch_native
    lib = HERE / "engine" / "target" / "release" / glitch_native.LIB_NAME
    if rebuild or not lib.is_file():
        cargo = shutil.which("cargo")
        if cargo is None:
            print(f"error: {lib} is missing and Rust's cargo isn't installed (https://rustup.rs), "
                  "or build with --no-engine", file=sys.stderr)
            return None
        print("building the native engine (cargo build --release) ...")
        subprocess.check_call([cargo, "build", "--release", "--lib"], cwd=HERE / "engine")
    try:                                      # the library must match glitch_native.py
        glitch_native.Engine(lib)
    except (OSError, RuntimeError) as ex:
        print(f"error: the native engine {lib} doesn't load: {ex}\n(rebuild it: "
              "python build_portable.py --rebuild-engine)", file=sys.stderr)
        return None
    return lib


def main() -> int:
    ap = argparse.ArgumentParser(description="Package the glitch editor with PyInstaller.")
    ap.add_argument("--no-pose", action="store_true",
                    help="leave out MediaPipe (pose estimation); much smaller")
    ap.add_argument("--onefile", action="store_true",
                    help="a single executable instead of a folder (slower start, unpacks each run)")
    ap.add_argument("--no-zip", action="store_true", help="don't zip the result")
    ap.add_argument("--no-engine", action="store_true",
                    help="leave out the native engine (render with the Python renderer)")
    ap.add_argument("--rebuild-engine", action="store_true",
                    help="build the native engine (cargo build --release) even if a library exists")
    args = ap.parse_args()

    for f in ("glitch.py", "glitch_gui.py", "estimate_pose.py", "glitch_numba.py"):
        if not (HERE / f).is_file():
            print(f"error: {f} must be next to build_portable.py", file=sys.stderr)
            return 2
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("installing PyInstaller into this Python ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    if not args.no_pose:
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            print("error: mediapipe is not installed in this Python (pip install mediapipe), "
                  "or build with --no-pose", file=sys.stderr)
            return 2

    sep = os.pathsep                          # --add-data separator: ';' on Windows, ':' elsewhere
    plugin = HERE / "resolve" / "Glitcher.py"
    if not plugin.is_file():
        print(f"error: {plugin} is missing (the DaVinci Resolve plugin)", file=sys.stderr)
        return 2
    cmd = [sys.executable, "-m", "PyInstaller", str(HERE / "glitch_gui.py"),
           "--name", NAME, "--noconfirm", "--windowed",
           "--onefile" if args.onefile else "--onedir",
           "--paths", str(HERE),
           "--hidden-import", "estimate_pose",
           "--hidden-import", "PIL._tkinter_finder",   # Pillow <-> Tk bridge, missed otherwise
           "--distpath", str(HERE / "dist"), "--workpath", str(HERE / "build"),
           "--specpath", str(HERE / "build"),
           "--add-data", f"{plugin}{sep}resolve"]      # Glitcher --install-resolve copies it
    if not args.no_engine:
        lib = engine_library(args.rebuild_engine)
        if lib is None:
            return 2
        cmd += ["--hidden-import", "glitch_native", "--add-binary", f"{lib}{sep}."]
        print(f"bundling the native engine {lib}")
    if args.no_pose:
        cmd += ["--exclude-module", "mediapipe"]
    else:
        cmd += ["--collect-all", "mediapipe", "--hidden-import", "mediapipe.tasks.c"]
        # MediaPipe imports OpenCV (cv2) only for drawing helpers Glitcher never calls. Leave the
        # real OpenCV out (~110 MB, and its self-loader breaks inside a macOS .app: "recursion
        # is detected during loading of cv2") and give MediaPipe a placeholder module instead.
        (HERE / "build").mkdir(exist_ok=True)
        stub = HERE / "build" / "pyi_rth_cv2_placeholder.py"
        stub.write_text(CV2_PLACEHOLDER, encoding="utf-8")
        cmd += ["--exclude-module", "cv2", "--runtime-hook", str(stub)]
        models = [p for v in ("heavy", "full", "lite")
                  for p in (HERE / f"pose_landmarker_{v}.task",
                            Path.home() / ".cache" / "estimate_pose" / f"pose_landmarker_{v}.task")
                  if p.is_file()]
        if models:
            cmd += ["--add-data", f"{models[0]}{sep}."]
            print(f"bundling model {models[0]}")
        else:
            print("note: no pose model found to bundle - the app will download it on first use")
    try:
        import miniaudio  # noqa: F401
        cmd += ["--collect-all", "miniaudio"]            # MP3 audio + playback in the Audio tab
    except ImportError:
        print("note: miniaudio not installed - the app will read WAV audio only "
              "(pip install miniaudio for MP3 and in-app playback)")
    try:
        import OpenEXR  # noqa: F401
        cmd += ["--hidden-import", "OpenEXR"]
    except ImportError:
        print("note: OpenEXR not installed - the app will read/write PNG only")
    try:
        import numba  # noqa: F401
        # the compiled kernels: as a module, and as a plain .py file next to the app, which is
        # what glitch.py loads them from when frozen (Numba keys its disk cache to a source file)
        cmd += ["--hidden-import", "glitch_numba", "--add-data", f"{HERE / 'glitch_numba.py'}{sep}."]
    except ImportError:
        print("note: Numba not installed - the app renders with numpy (same frames, ~1.3x slower)")

    print(" ".join(cmd))
    subprocess.check_call(cmd)

    out = HERE / "dist" / (f"{NAME}.app" if sys.platform == "darwin" else
                           (f"{NAME}.exe" if args.onefile and os.name == "nt" else NAME))
    print(f"\nbuilt: {out}")
    if not args.no_zip and out.exists():
        tag = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux") + "-" + arch()
        if args.no_pose:
            tag += "-nopose"
        sys.path.insert(0, str(HERE))
        import glitch
        version = glitch.__version__
        if sys.platform == "darwin":          # ditto keeps the .app's symlinks and permissions
            z = str(HERE / "dist" / f"{NAME}-{version}-{tag}.zip")
            subprocess.check_call(["ditto", "-c", "-k", "--keepParent", str(out), z])
        else:
            z = shutil.make_archive(str(HERE / "dist" / f"{NAME}-{version}-{tag}"), "zip",
                                    root_dir=out.parent, base_dir=out.name)
        print(f"zipped: {z}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
