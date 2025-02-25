import numpy as np

sizes = (1, 2, 3)
print(sizes)

# for y in sizes:
#     print(y)
# for y in sizes[1:]:
#     print(np.random.randn(y, 1))

for x, y in zip(sizes[-1:], sizes[1:]):
    print(f"{x}, {y}")
    # np.random.randn(y, 1)

