"""Build a small UOp DAG and inspect identity and topological order."""

from tinygrad import dtypes
from tinygrad.uop.ops import UOp


def short_arg(value, limit=42):
  text = repr(value)
  return text if len(text) <= limit else text[:limit-3] + "..."


def main():
  x = UOp.variable("x", -8, 8, dtypes.float32)
  shared = x * 2
  root = shared + shared

  # Reconstructing the same live graph returns the interned nodes.
  reconstructed = (x * 2) + (x * 2)
  assert root is reconstructed
  assert root.src[0] is root.src[1] is shared

  order = list(root.toposort())
  number = {node: i for i, node in enumerate(order)}
  assert order[-1] is root
  assert all(number[src] < number[node] for node in order for src in node.src)

  print("local  op        dtype              shape       src       arg")
  print("-----  --------  -----------------  ----------  --------  ------------------------------------------")
  for node in order:
    srcs = "[" + ",".join(f"N{number[src]}" for src in node.src) + "]"
    print(f"N{number[node]:<4}  {node.op.name:<8}  {str(node.dtype):<17}  {str(node.shape):<10}  "
          f"{srcs:<8}  {short_arg(node.arg)}")

  print("\nroot sources share identity:", root.src[0] is root.src[1])
  print("reconstructed root is root: ", reconstructed is root)

  # tag is part of the interning key. It has no universal interpretation: a
  # pass decides what its own tag means.
  tagged = root.rtag("phase2-demo")
  assert tagged is not root
  assert root.rtag("phase2-demo") is tagged
  assert tagged.src == root.src
  print("tag changes identity:         ", tagged is not root)
  print("same tagged form is interned: ", root.rtag("phase2-demo") is tagged)


if __name__ == "__main__":
  main()
