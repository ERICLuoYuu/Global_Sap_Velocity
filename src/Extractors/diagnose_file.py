import h5py
import shutil
import tempfile
import os

# Copy to temp
src = r".\data\raw\grided\stand_age\2026113115224222_BGIForestAgeMPIBGC1.0.0.nc"
dst = os.path.join(tempfile.gettempdir(), "test.nc")
shutil.copy2(src, dst)
print(f"Copied to: {dst}")

# Try h5py directly
try:
    f = h5py.File(dst, 'r')
    print("SUCCESS! Keys:", list(f.keys()))
    f.close()
except Exception as e:
    print(f"h5py failed: {e}")

# Check HDF5 library version
print(f"\nh5py version: {h5py.__version__}")
print(f"HDF5 version: {h5py.version.hdf5_version}")