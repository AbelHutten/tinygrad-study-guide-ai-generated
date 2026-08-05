"""Observe Tensor wrappers, lazy UOps, and autograd without GPU hardware."""

from collections import Counter

from tinygrad import Tensor, dtypes


def op_counts(root):
  return Counter(node.op.name for node in root.toposort())


def main():
  x = Tensor([1.0, 2.0, 3.0], device="PYTHON", dtype=dtypes.float32)

  # Tensor objects are ordinary Python wrappers. Rebuilding the same expression
  # makes a new wrapper, but live, structurally identical UOps are interned.
  twice_a, twice_b = x * 2, x * 2
  assert twice_a is not twice_b
  assert twice_a.uop is twice_b.uop
  print("different Tensor wrappers:", twice_a is not twice_b)
  print("same interned UOp:       ", twice_a.uop is twice_b.uop)

  loss = (x * x + 2 * x).sum()
  assert not loss.uop.is_realized
  print("\nforward shape/dtype:     ", loss.shape, loss.dtype)
  print("forward op counts:       ", dict(sorted(op_counts(loss.uop).items())))
  print("loss realized before read:", loss.uop.is_realized)

  # gradient returns another lazy Tensor graph. It does not write x.grad.
  dx, = loss.gradient(x)
  assert x.grad is None
  print("\ngradient shape/dtype:    ", dx.shape, dx.dtype)
  print("gradient op counts:      ", dict(sorted(op_counts(dx.uop).items())))
  print("x.grad after gradient(): ", x.grad)

  # backward uses the same graph transform, then attaches gradient wrappers to
  # live Tensor objects in the forward graph. Reading the value realizes it.
  loss.backward()
  assert x.grad is not None
  print("x.grad after backward(): ", "attached")
  print("gradient realized before read:", x.grad.uop.is_realized)
  values = x.grad.tolist()
  print("gradient values:         ", values)
  print("tolist() crossed the execution boundary")
  assert values == [4.0, 6.0, 8.0]


if __name__ == "__main__":
  main()
