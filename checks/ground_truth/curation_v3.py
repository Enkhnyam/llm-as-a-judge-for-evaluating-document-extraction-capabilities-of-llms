"""The frozen curated dataset used for scoring."""
import hashlib
from _setup import *

before = curated(original)
after = curated(frozen)
checksum = hashlib.sha256(data_path(frozen).read_bytes()).hexdigest()

print("original", len(before), "experiments")
print("frozen  ", len(after), "experiments  (the difference was curated in by hand)")
print("papers  ", after.doi.nunique())
print("sha256  ", checksum)
