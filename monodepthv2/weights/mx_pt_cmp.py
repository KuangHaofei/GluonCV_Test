import mxnet as mx


if __name__ == '__main__':
    x = mx.random.randn(1, 1, 3, 3)
    mask = mx.nd.zeros_like(x)
    mask[:, :, 1:, 1:] = 1

    print(x)
    print(mask)

    print(mx.nd.contrib.boolean_mask(x.reshape(-1), mask.reshape(-1)))

