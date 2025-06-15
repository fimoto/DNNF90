import time, sys, torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
W = int(sys.argv[1]); nep = int(sys.argv[2])
import numpy as np
d = np.loadtxt('train.dat')
x = torch.tensor(d[:,0:1]); y = torch.tensor(d[:,1:2])
dims = [1, W, W, W, W, W, 1]
layers = []
for i in range(6):
    layers.append(torch.nn.Linear(dims[i], dims[i+1]))
    if i < 5: layers.append(torch.nn.Tanh())
net = torch.nn.Sequential(*layers)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
# warmup
loss = ((net(x)-y)**2).mean(); loss.backward(); opt.zero_grad()
t0 = time.time()
for ep in range(nep):
    opt.zero_grad()
    loss = 0.5*((net(x)-y)**2).mean()
    loss.backward()
    opt.step()
t1 = time.time()
print(f"W={W} epochs={nep} torch_time={t1-t0:.3f}s  loss={loss.item():.3e}")
