from torch import cat, split
from torch.nn import Module, Conv1d, BatchNorm1d, SiLU, Sequential

__all__ = ["ConvDownSampler", "Chunker"]

class ConvDownSampler(Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        ds_rate //= 2
        chan = embed_dim // ds_rate
        blocks = [Conv1d(in_chans, chan, 5, 2, 2), BatchNorm1d(chan), SiLU()]

        while ds_rate > 1:
            blocks += [
                Conv1d(chan, 2 * chan, 5, 2, 2),
                BatchNorm1d(2 * chan),
                SiLU(),
            ]
            ds_rate //= 2
            chan = 2 * chan

        blocks += [
            Conv1d(
                chan,
                embed_dim,
                1,
            )
        ]
        self.blocks = Sequential(*blocks)

    def forward(self, X):
        return self.blocks(X)


class Chunker(Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        self.embed = Conv1d(in_chans, embed_dim // ds_rate, 7, padding=3)
        self.project = Conv1d((embed_dim // ds_rate) * ds_rate, embed_dim, 1)
        self.ds_rate = ds_rate

    def forward(self, X):
        X = self.embed(X)
        X = cat(
            [cat(split(x_i, 1, -1), 1) for x_i in split(X, self.ds_rate, -1)],
            -1,
        )
        X = self.project(X)

        return X