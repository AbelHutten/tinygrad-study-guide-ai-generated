"""Observe Tensor wrappers, lazy UOps, and autograd without GPU hardware."""

from collections import Counter

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops


def op_counts(root):
  return Counter(node.op.name for node in root.toposort())


def print_graph(root):
  nodes = list(root.toposort())
  node_id = {node: f"N{i}" for i, node in enumerate(nodes)}
  for node in nodes:
    sources = ",".join(node_id[src] for src in node.src)
    detail = f" arg={node.arg}" if node.op is Ops.CONST else ""
    print(f"  {node_id[node]} {node.op.name} shape={node.shape} <- [{sources}]{detail}")


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
  print("forward graph:")
  print_graph(loss.uop)
  print("loss realized before read:", loss.uop.is_realized)

  # gradient returns another lazy Tensor graph. It does not write x.grad.
  dx, = loss.gradient(x)
  assert x.grad is None
  print("\ngradient shape/dtype:    ", dx.shape, dx.dtype)
  print("gradient op counts:      ", dict(sorted(op_counts(dx.uop).items())))
  print("gradient graph:")
  print_graph(dx.uop)
  print("x.grad after gradient(): ", x.grad)
  print("returned gradient realized:", dx.uop.is_realized)
  dx_values = dx.tolist()
  print("returned gradient values:", dx_values)
  assert dx_values == [4.0, 6.0, 8.0]

  # backward uses the same graph transform, then attaches gradient wrappers to
  # live Tensor objects in the forward graph. Reading the value realizes it.
  loss.backward()
  assert x.grad is not None
  print("x.grad after backward(): ", "attached")
  print("gradient realized before read:", x.grad.uop.is_realized)
  values = x.grad.tolist()
  print("gradient values:         ", values)
  assert values == [4.0, 6.0, 8.0]

  # backward accumulates into an existing .grad. Optimizer.zero_grad performs
  # the same reset as assigning None here for each parameter.
  loss.backward()
  accumulated = x.grad.tolist()
  print("gradient after two backward calls:", accumulated)
  assert accumulated == [8.0, 12.0, 16.0]
  x.grad = None
  print("gradient after explicit reset:", x.grad)

  # A non-scalar output needs an explicit cotangent/seed. This computes a
  # vector-Jacobian product, not a full Jacobian matrix.
  vector = x * x
  seed = Tensor([1.0, 10.0, 100.0], device="PYTHON", dtype=dtypes.float32)
  seeded_dx, = vector.gradient(x, gradient=seed)
  seeded_values = seeded_dx.tolist()
  print("seeded vector gradient:  ", seeded_values)
  assert seeded_values == [2.0, 40.0, 600.0]

  # DETACH preserves forward values but blocks the reverse traversal through
  # one branch.
  detached_loss = (x * x + (2 * x).detach()).sum()
  detached_dx, = detached_loss.gradient(x)
  detached_values = detached_dx.tolist()
  print("gradient with detached 2*x branch:", detached_values)
  assert detached_values == [2.0, 4.0, 6.0]

  # Broadcasting repeats logical values in the forward pass. The reverse pass
  # must sum repeated contributions back to each source shape.
  xb = Tensor([[1.0], [2.0]], device="PYTHON", dtype=dtypes.float32)
  w = Tensor([[10.0, 20.0, 30.0]], device="PYTHON", dtype=dtypes.float32)
  broadcast_loss = (xb * w).sum()
  dxb, dw = broadcast_loss.gradient(xb, w)
  dxb_values, dw_values = dxb.tolist(), dw.tolist()
  print("broadcast gradient shapes:", dxb.shape, dw.shape)
  print("broadcast gradient dxb:   ", dxb_values)
  print("broadcast gradient dw:    ", dw_values)
  assert dxb.shape == (2, 1) and dw.shape == (1, 3)
  assert dxb_values == [[60.0], [60.0]]
  assert dw_values == [[3.0, 3.0, 3.0]]


if __name__ == "__main__":
  main()
