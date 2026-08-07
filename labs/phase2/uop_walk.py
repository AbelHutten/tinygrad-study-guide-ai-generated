"""Inspect the raw UOp DAG for (x*x + 2*x).sum()."""

import gc
import weakref

from tinygrad import dtypes
from tinygrad.uop import Ops
from tinygrad.uop.ops import UOp, consumer_map_from_toposort


def payload(node: UOp) -> str:
  """Render only the stable, relevant part of each operation's payload."""
  if node.op is Ops.PARAM:
    return f"slot={node.arg.slot}, name={node.arg.name!r}, device={node.arg.device!r}"
  if node.op is Ops.CONST:
    return repr(node.arg)
  return repr(node.arg) if node.arg is not None else "-"


def print_graph(root: UOp) -> tuple[list[UOp], dict[UOp, int]]:
  order = list(root.toposort())
  number = {node: i for i, node in enumerate(order)}

  print("id  op       dtype             shape  src       arg")
  print("--  -------- ----------------- ------ --------- ----------------------------------------")
  for node in order:
    sources = "[" + ",".join(f"N{number[source]}" for source in node.src) + "]"
    print(f"N{number[node]}  {node.op.name:<8} {str(node.dtype):<17} {str(node.shape):<6} "
          f"{sources:<9} {payload(node)}")
  return order, number


def main():
  # This is Chapter 4's loss expressed directly in tinygrad's UOp language.
  x = UOp.param(0, dtypes.float32, shape=(3,), device="PYTHON", name="x")
  square = x * x
  scaled = 2 * x
  summed = square + scaled
  loss = summed.sum()

  order, number = print_graph(loss)
  expected_ops = [Ops.CONST, Ops.PARAM, Ops.MUL, Ops.CONST, Ops.MUL, Ops.ADD, Ops.REDUCE]
  assert [node.op for node in order] == expected_ops
  assert order[-1] is loss and loss.shape == ()
  assert len(square.src) == 2 and all(source is x for source in square.src)
  assert len(loss.backward_slice) == 6
  assert all(number[source] < number[node] for node in order for source in node.src)

  # Construction is interned while the equivalent graph is still alive.
  rebuilt_square = x * x
  rebuilt_loss = (x * x + 2 * x).sum()
  assert rebuilt_square is square
  assert rebuilt_loss is loss

  square_positions = [i for i, source in enumerate(square.src) if source is x]
  edge_positions = [f"N{number[node]}[{i}]" for node in order for i, source in enumerate(node.src) if source is x]
  consumers = consumer_map_from_toposort(order)
  consumer_numbers = [f"N{number[node]}" for node in consumers[x]]
  assert square_positions == [0, 1]
  assert edge_positions == ["N2[0]", "N2[1]", "N4[1]"]
  assert consumer_numbers == ["N2", "N4"]

  print("\nsame square rebuilt:    ", rebuilt_square is square)
  print("same loss rebuilt:      ", rebuilt_loss is loss)
  print("x positions in square:  ", square_positions)
  print("all x source positions: ", edge_positions)
  print("x consumer nodes:       ", consumer_numbers)
  print("backward slice nodes:   ", len(loss.backward_slice))
  print("toposort nodes:         ", len(order))

  # tag participates in interning, while the separate .key digest omits tag.
  tag = ("phase2", "demo")
  tagged = loss.rtag(tag)
  assert tagged is not loss
  assert loss.rtag(tag) is tagged
  assert tagged.key == loss.key
  try:
    loss.replace(arg=[Ops.ADD, 1])
  except TypeError:
    list_arg_rejected = True
  else:
    list_arg_rejected = False
  try:
    loss.rtag(["not", "hashable"])
  except TypeError:
    list_tag_rejected = True
  else:
    list_tag_rejected = False
  assert list_arg_rejected and list_tag_rejected

  print("\ntag changes identity:    ", tagged is not loss)
  print("same tagged form reused: ", loss.rtag(tag) is tagged)
  print("tag omitted from .key:   ", tagged.key == loss.key)
  print("list arg rejected:       ", list_arg_rejected)
  print("list tag rejected:       ", list_tag_rejected)

  # Replacing an interior node does not mutate or redirect its consumers.
  tripled = square.replace(src=(x, UOp.const(3.0)))
  updated_sum = summed.replace(src=(tripled, scaled))
  updated_loss = loss.replace(src=(updated_sum,))
  assert tripled is x * 3
  assert tripled is not square and updated_loss is not loss
  assert len(square.src) == 2 and all(source is x for source in square.src)
  assert loss.src[0] is summed and updated_loss.src[0] is updated_sum
  assert updated_sum.src[1] is scaled
  assert updated_loss.shape == loss.shape == ()

  print("\nnew root uses replacement:             ", updated_loss.src[0] is updated_sum)
  print("replacement is interned x * 3:         ", tripled is x * 3)
  print("unchanged scaled branch is shared:     ", updated_sum.src[1] is scaled)
  print("old root still uses original sum:      ", loss.src[0] is summed)
  print("old and new root shapes:               ", loss.shape, updated_loss.shape)

  # The cache is weak: it reuses live forms but is not permanent storage.
  temporary = UOp.const(987654.25, dtypes.float32)
  temporary_again = UOp.const(987654.25, dtypes.float32)
  temporary_ref = weakref.ref(temporary)
  assert temporary_again is temporary and temporary_ref() is temporary
  del temporary, temporary_again
  gc.collect()
  assert temporary_ref() is None
  new_temporary = UOp.const(987654.25, dtypes.float32)
  assert new_temporary is not None

  print("\nweak-cache object alive after deleting references:", temporary_ref() is not None)
  print("equivalent node can be constructed again:          ", new_temporary is not None)


if __name__ == "__main__":
  main()
