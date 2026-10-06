import torch


def test():
    a = torch.zeros(16, 3, 200, 200)
    print(a.shape)
    print(a.dtype)
    print(a.device)
    b = a.to("cuda")
    print(b.device)
    print(torch.cuda.is_available())


test()
