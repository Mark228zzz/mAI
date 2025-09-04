import random


class DataLoader:
    def __init__(self, data, target, batch_size=1, shuffle=True):
        assert len(data) == len(target), "data and target must be the same length"

        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self.indices):
            raise StopIteration

        # Get batch indices
        batch_idx = self.indices[self.current:self.current+self.batch_size]
        self.current += self.batch_size

        # Slice data + targets
        batch_data = [self.data[i] for i in batch_idx]
        batch_target = [self.target[i] for i in batch_idx]

        return batch_data, batch_target

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
