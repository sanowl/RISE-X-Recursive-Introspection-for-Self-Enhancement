import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, gym
from typing import NamedTuple, List
from torch.distributions import Categorical
from collections import deque

class Exp(NamedTuple): s: np.ndarray; a: int; r: float; ns: np.ndarray; d: bool; i: float

class PER:
    def __init__(self, cap: int, α: float = 0.6, β: float = 0.4, βi: float = 0.001):
        self.cap, self.α, self.β, self.βi, self.max_p = cap, α, β, βi, 1.0
        self.tree, self.data = np.zeros(2*cap-1), np.empty(cap, dtype=object)
        self.size = self.ptr = 0

    def add(self, e: Exp):
        idx = self.ptr + self.cap - 1
        self.data[self.ptr] = e
        self.update(idx, self.max_p)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update(self, idx: int, p: float):
        change, self.tree[idx] = p - self.tree[idx], p
        while idx := (idx - 1) // 2: self.tree[idx] += change
        self.max_p = max(p, self.max_p)

    def sample(self, n: int):
        idxs = [self.get(np.random.uniform(i*seg, (i+1)*seg)) for i, seg in enumerate(np.linspace(0, self.tree[0], n+1)[:-1])]
        batch = [self.data[i] for i in idxs]
        probs = self.tree[idxs + self.cap - 1] / self.tree[0]
        self.β = min(1., self.β + self.βi)
        weights = (self.size * probs) ** -self.β / max((self.size * probs) ** -self.β)
        return batch, idxs, weights

    def get(self, v: float):
        idx = 0
        while idx < self.cap - 1:
            left = 2 * idx + 1
            right = left + 1  # Define right here
            idx = left if v <= self.tree[left] else right
            v -= self.tree[left] * (idx == right)
        return idx - self.cap + 1

class NoisyLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, σ: float = 0.5):
        super().__init__()
        self.μw, self.σw = nn.Parameter(torch.empty(out_f, in_f)), nn.Parameter(torch.empty(out_f, in_f))
        self.μb, self.σb = nn.Parameter(torch.empty(out_f)), nn.Parameter(torch.empty(out_f))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        μ_range = 1 / np.sqrt(self.μw.size(1))
        self.μw.data.uniform_(-μ_range, μ_range)
        self.σw.data.fill_(self.σ / np.sqrt(self.σw.size(1)))
        self.μb.data.uniform_(-μ_range, μ_range)
        self.σb.data.fill_(self.σ / np.sqrt(self.σb.size(0)))

    def reset_noise(self):
        εi, εj = [self._scale_noise(s) for s in self.μw.size()]
        self.εw, self.εb = εj.outer(εi), εj

    def _scale_noise(self, size: int):
        return torch.randn(size).sign().mul(torch.rand(size).sqrt())

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.μw + self.σw * self.εw, self.μb + self.σb * self.εb)

class Net(nn.Module):
    def __init__(self, s: int, a: int, h: List[int], d: float = 0.1):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(s, h[0]), nn.ReLU(), nn.LayerNorm(h[0]), nn.Dropout(d),
            *sum([[NoisyLinear(i, o), nn.ReLU(), nn.LayerNorm(o), nn.Dropout(d)]
                  for i, o in zip(h, h[1:])], []))
        self.v, self.a = NoisyLinear(h[-1], 1), NoisyLinear(h[-1], a)

    def forward(self, x: torch.Tensor):
        f = self.feat(x)
        return self.v(f) + self.a(f) - self.a(f).mean(1, keepdim=True)

class RISE:
    def __init__(self, s: int, a: int, h: List[int], γ: float = 0.99, τ: float = 5e-3, lr: float = 3e-4, α: float = 0.2, n: int = 3):
        self.q, self.tq = Net(s, a, h), Net(s, a, h)
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr)
        self.γ, self.τ, self.α, self.a, self.n = γ, τ, α, a, n

    def act(self, s: np.ndarray, ε: float = 0.) -> int:
        return np.random.randint(self.a) if np.random.rand() < ε else self.q(torch.FloatTensor(s)).argmax().item()

    def learn(self, b: List[Exp], w: np.ndarray):
        s, a, r, ns, d, _ = map(torch.tensor, zip(*b))
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        nq = self.tq(ns).max(1)[0]
        eq = (r + self.γ**self.n * nq * (1-d)).detach()
        loss = (torch.FloatTensor(w) * F.mse_loss(q, eq, reduction='none')).mean()
        p = Categorical(F.softmax(self.q(s), dim=1))
        loss -= self.α * p.entropy().mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        for tp, p in zip(self.tq.parameters(), self.q.parameters()):
            tp.data.copy_(self.τ * p.data + (1 - self.τ) * tp.data)
        return loss.item(), p.entropy().mean().item(), ((q - eq).abs().detach().numpy() + 1e-6)

class Trainer:
    def __init__(self, env: gym.Env, agent: RISE, bs: int = 64, n: int = 1000, spi: int = 1000, ue: int = 4, sε: float = 1., eε: float = 0.01, εd: float = 0.995):
        self.env, self.agent, self.bs, self.n, self.spi = env, agent, bs, n, spi
        self.ue, self.ε, self.eε, self.εd = ue, sε, eε, εd
        self.buf, self.n_step_buf = PER(1_000_000), deque(maxlen=agent.n)

    def n_step_learn(self):
        if len(self.n_step_buf) < self.agent.n: return
        r = sum([e.r * self.agent.γ**i for i, e in enumerate(self.n_step_buf)])
        self.buf.add(Exp(self.n_step_buf[0].s, self.n_step_buf[0].a, r, self.n_step_buf[-1].ns, self.n_step_buf[-1].d, 1))

    def train(self):
        for i in range(self.n):
            s, d, r, ts = self.env.reset(), False, 0, 0
            while not d and ts < self.spi:
                a = self.agent.act(s, self.ε)
                ns, rw, d, _ = self.env.step(a)
                self.n_step_buf.append(Exp(s, a, rw, ns, d, 1))
                self.n_step_learn()
                s, r, ts = ns, r + rw, ts + 1
                if len(self.buf.data) > self.bs and ts % self.ue == 0:
                    b, idx, w = self.buf.sample(self.bs)
                    l, e, prios = self.agent.learn(b, w)
                    self.buf.update(idx, prios)
            self.ε = max(self.eε, self.ε * self.εd)
            print(f"Iter {i+1}/{self.n}, R: {r:.2f}, ε: {self.ε:.3f}")

    def eval(self, n: int = 100) -> float:
        return np.mean([sum(self.env.step(self.agent.act(s))[1] for s in [self.env.reset()] for _ in iter(lambda: self.env.step(self.agent.act(s))[2], True)) for _ in range(n)])

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = RISE(env.observation_space.shape[0], env.action_space.n, [64, 64])
    trainer = Trainer(env, agent)
    trainer.train()
    print(f"Avg R: {trainer.eval():.2f}")
