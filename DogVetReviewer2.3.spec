# DogVetReviewer2.3.spec
# Works with PyInstaller 6.x and your 3.12 venv.

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT
from PyInstaller.building.datastruct import Tree

# Use CWD (PyInstaller does not define __file__ inside .spec)
PROJECT_DIR = Path(os.getcwd()).resolve()

script = str(PROJECT_DIR / "DogVetReviewer2.3.py")  # <-- must match the .py filename exactly

# Optional runtime folders/files (only added if they exist)
datas = []
if (PROJECT_DIR / "badwords.txt").exists():
    datas.append((str(PROJECT_DIR / "badwords.txt"), "."))

trees = []
if (PROJECT_DIR / "vlc_runtime").is_dir():
    trees.append(Tree(str(PROJECT_DIR / "vlc_runtime"), prefix="vlc_runtime"))
if (PROJECT_DIR / "ffmpeg").is_dir():
    trees.append(Tree(str(PROJECT_DIR / "ffmpeg"), prefix="ffmpeg"))

# If you want to ship local VLC from Program Files, add it too (optional):
# vlc_pf = Path(r"C:\Program Files\VideoLAN\VLC")
# if vlc_pf.is_dir():
#     trees.append(Tree(str(vlc_pf), prefix="vlc_runtime"))

hiddenimports = [
    # speech stack bits that PyInstaller sometimes misses
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
]

# If you *do not* use onnxruntime or torch, exclude them to slim the build
excludes = [
    "onnx", "onnxruntime", "onnxruntime.quantization",
    "torch", "torchvision", "torchaudio",
]

a = Analysis(
    [script],
    pathex=[str(PROJECT_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

# Add directory trees (vlc_runtime, ffmpeg) after Analysis creation
for t in trees:
    a.datas += t

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="DogVetReviewer",
    console=False,                             # hides console window
    icon=r"C:\Dev\DogVetReviewer\icon.ico",    # your icon path
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DogVetReviewer",
)
