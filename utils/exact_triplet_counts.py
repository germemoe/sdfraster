"""Exact duplicate-coordinate counts for Delaunay input validation.

The native library hashes float64 coordinate triplets bit-exactly; any unsupported input
falls back to the original NumPy lexsort, so results are identical either way.
"""
import ctypes,hashlib
from pathlib import Path
import numpy as np
from .delaunay_isolation import is_external_termination

def original_sort_counts(array):
 order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
 sorted_points = array[order]
 starts = np.flatnonzero(np.any(sorted_points[1:] != sorted_points[:-1], axis=1)) + 1
 counts = np.diff(np.concatenate(([0], starts, [len(array)])))
 return len(counts),int(np.count_nonzero(counts>1))

class ExactTripletCounts:
 def __init__(self,module_pin):
  self.lib=ctypes.CDLL(module_pin['path']);self.lib.exact_triplet_counts_abi.restype=ctypes.c_char_p
  assert self.lib.exact_triplet_counts_abi()==b'E164_EXACT_NATIVE_F64_TRIPLET_COUNTS_V1'
  self.call=self.lib.exact_triplet_counts;self.call.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)];self.call.restype=ctypes.c_int
  self.fixture=self.lib.exact_triplet_counts_fixture;self.fixture.argtypes=self.call.argtypes+[ctypes.c_uint,ctypes.c_uint];self.fixture.restype=ctypes.c_int
  self.lib.exact_triplet_slot_bytes.restype=ctypes.c_uint64;self.slot_bytes=int(self.lib.exact_triplet_slot_bytes())
  self.lib.exact_triplet_fp_snapshot.argtypes=[ctypes.POINTER(ctypes.c_uint64)];self.lib.exact_triplet_fp_snapshot.restype=ctypes.c_int
  self.native_attempts=0;self.hash_calls=0;self.fallback_calls=0;self.reasons={};self.last_backend=None;self.last_table_bytes=0
 def fallback(self,array,reason):
  self.fallback_calls+=1;self.reasons[reason]=self.reasons.get(reason,0)+1;self.last_backend='ORIGINAL_NUMPY_SORT:'+reason
  return original_sort_counts(array)
 def __call__(self,points,array):
  # A normal C-contiguous ndarray view is allowed; ndarray subclasses are not.
  if not(type(points)is np.ndarray and points.ndim==2 and points.shape[1:]==(3,)
   and points.dtype==np.dtype('float64')and points.dtype.isnative
   and points.flags.c_contiguous and points.flags.aligned and len(points)>=4
   and points is array):return self.fallback(array,'input_contract')
  out=(ctypes.c_uint64*3)();self.native_attempts+=1
  try:code=int(self.call(array.ctypes.data,len(array),array.nbytes,out))
  except Exception as error:
   if is_external_termination(error):raise
   return self.fallback(array,'native_call_exception')
  if code:return self.fallback(array,{1:'native_unsupported',2:'native_allocation',3:'native_exception',4:'subnormal',5:'nonfinite'}.get(code,'unknown_native_status'))
  self.hash_calls+=1;self.last_backend='EXACT_INTEGER_TRIPLET_HASH';self.last_table_bytes=int(out[2])*self.slot_bytes
  return int(out[0]),int(out[1])
 def fp_snapshot(self):
  out=(ctypes.c_uint64*3)();assert self.lib.exact_triplet_fp_snapshot(out)==0
  return tuple(int(v)for v in out)

class _NumpyTripletCounts:
 """Fallback with identical results when the native library is not built."""
 last_backend='ORIGINAL_NUMPY_SORT:no_native_library'
 def __call__(self,points,array):
  return original_sort_counts(array)


def load_exact_triplet_counts():
 path=Path(__file__).resolve().with_name('exact_triplet_counts_linux_x86_64.so')
 if not path.is_file():
  return _NumpyTripletCounts()
 return ExactTripletCounts(dict(path=str(path)))
