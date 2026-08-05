"""Show TinyJit's ignore, capture, and replay calls."""

from tinygrad import Device, Tensor, TinyJit
from tinygrad.helpers import JIT


@TinyJit
def add_one(x: Tensor) -> Tensor:
  return (x + 1).realize()


for call_number in range(3):
  phase = ("ignore", "capture", "replay")[call_number] if JIT else "ordinary Python (JIT=0)"
  x = Tensor([float(call_number), float(call_number + 1)], device=Device.DEFAULT).realize()
  out = add_one(x)
  expected = [float(call_number + 1), float(call_number + 2)]
  actual = out.tolist()
  print(
    f"call={call_number + 1} phase={phase} next_count={add_one.cnt} "
    f"captured={add_one.captured is not None} result={actual}"
  )
  assert actual == expected

if JIT:
  assert add_one.captured is not None
  print("captured call bodies:", [call.src[0].op.name for call in add_one.captured._linear.src])
else:
  assert add_one.captured is None
  print("capture disabled: True")
