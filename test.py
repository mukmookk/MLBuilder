import os

print([os.path.join("./cifar-10-batches-py", 'data_batch_{}'.format(i)).replace('\\', '/') for i in range(1, 6)])