import os
import time
from collections import deque
from itertools import chain

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from common.util import scale_ob, Trajectories

import math

import torch.nn as nn
from torch.optim import Adam
from gym import spaces

import torch
import torch.distributions

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((44, 44)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255),
    # transforms.Lambda(lambda x: x.numpy()),
])

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class Categorical(object):
    """Categorical policy distribution"""
    def __init__(self, logits):
        self._dist = torch.distributions.Categorical(logits=logits)

    def entropy(self):
        return self._dist.entropy()

    def log_prob(self, b_a):
        return self._dist.log_prob(b_a.long())

    def sample(self):
        return self._dist.sample()

    @classmethod
    def __repr__(cls):
        return 'Categorical'


class DiagGaussian(object):
    """DiagGaussian policy distribution"""
    def __init__(self, mean_logstd):
        size = mean_logstd.size(1) // 2
        mean, logstd = torch.split(mean_logstd, size, 1)
        self._dist = torch.distributions.Normal(mean, logstd.exp())

    def entropy(self):
        return self._dist.entropy().sum(-1)

    def log_prob(self, b_a):
        return self._dist.log_prob(b_a.float()).sum(-1)

    def sample(self):
        return self._dist.sample()

    @classmethod
    def __repr__(cls):
        return 'DiagGaussian'


def atari(env, **kwargs):
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n
    network = CNN(in_dim, policy_dim)
    optimizer = Adam(network.parameters(), 2.5e-4, eps=1e-5)
    params = dict(
        dist=Categorical,
        network=network,
        optimizer=optimizer,
        gamma=0.99,
        grad_norm=0.5,
        timesteps_per_batch=128,
        ent_coef=.01,
        vf_coef=0.5,
        gae_lam=0.95,
        nminibatches=4,
        opt_iter=4,
        cliprange=0.1,
        ob_scale=1.0 / 255
    )
    params.update(kwargs)
    return params



class CNN(nn.Module):
    def __init__(self, in_shape, policy_dim):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(True)
        )
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.constant_(m.bias, 0)

        self.policy = nn.Linear(512, policy_dim)
        nn.init.orthogonal_(self.policy.weight, 1e-2)
        nn.init.constant_(self.policy.bias, 0)

        self.value = nn.Linear(512, 1)
        nn.init.orthogonal_(self.value.weight, 1)
        nn.init.constant_(self.value.bias, 0)

    def forward(self, x):
        latent = self.feature(x)
        return self.policy(latent), self.value(latent)



def learn(env, nenv,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          dist=Categorical,
          number_timesteps,
          network, optimizer,
          save_path='./', save_interval, ob_scale,
          gamma=0.99, grad_norm=0.5, timesteps_per_batch, ent_coef,
          vf_coef, gae_lam, nminibatches, opt_iter, cliprange):
    """
    Paper:
    Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization
    algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.

    Parameters:
    ----------
    gram_norm (float | None): grad norm
    timesteps_per_batch (int): number of steps per update
    ent_coef (float): policy entropy coefficient in the objective
    vf_coef (float): value function loss coefficient in the objective
    gae_lam (float): gae lambda
    nminibatches (int): number of training minibatches per update
    opt_iter (int): number of training iterations per update
    cliprange (float): clipping range

    """

    policy = network.to(device)
    number_timesteps = number_timesteps // nenv
    generator = _generate(
        device, env, dist, policy, ob_scale,
        number_timesteps, gamma, gae_lam, timesteps_per_batch
    )
    max_iter = number_timesteps // timesteps_per_batch
    scheduler = LambdaLR(optimizer, lambda i_iter: 1 - i_iter / max_iter)

    total_timesteps = 0
    infos = {'eplenmean': deque(maxlen=100), 'eprewmean': deque(maxlen=100)}
    start_ts = time.time()
    for n_iter in range(1, max_iter + 1):
        scheduler.step()

        batch = generator.__next__()
        *data, info = batch
        for d in info:
            infos['eplenmean'].append(d['l'])
            infos['eprewmean'].append(d['r'])
        total_timesteps += data[0].size(0)
        batch_size = data[0].size(0) // nminibatches
        loader = DataLoader(list(zip(*data)), batch_size, True)
        records = {'pg': [], 'v': [], 'ent': [], 'kl': [], 'clipfrac': []}
        for _ in range(opt_iter):
            for b_o, b_a, b_r, b_logp_old, b_v_old in loader:
                # calculate advantange
                b_logits, b_v = policy(b_o)
                b_v = b_v[:, 0]
                pd = dist(b_logits)
                entropy = pd.entropy().mean()
                b_logp = pd.log_prob(b_a)
                # highlight: normalized advantage will gives better performance
                adv = b_r - b_v_old
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # update policy
                c_b_v = b_v_old + (b_v - b_v_old).clamp(-cliprange, cliprange)
                vloss = 0.5 * torch.mean(torch.max(
                    (b_v - b_r).pow(2),
                    (c_b_v - b_r).pow(2)
                ))  # highlight: Clip is also applied to value loss
                ratio = (b_logp - b_logp_old).exp()
                pgloss = torch.mean(torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - cliprange, 1 + cliprange)
                ))
                loss = pgloss + vf_coef * vloss - ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                if grad_norm is not None:
                    nn.utils.clip_grad_norm_(policy.parameters(), grad_norm)
                optimizer.step()

                # record logs
                records['pg'].append(pgloss.item())
                records['v'].append(vloss.item())
                records['ent'].append(entropy.item())
                records['kl'].append((b_logp - b_logp_old).pow(2).mean() * 0.5)
                clipfrac = ((ratio - 1).abs() > cliprange).float().mean().item()
                records['clipfrac'].append(clipfrac)

        print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))
        fps = int(total_timesteps / (time.time() - start_ts))
        print('Total timesteps {} FPS {}'.format(total_timesteps, fps))
        for k, v in chain(infos.items(), records.items()):
            v = (sum(v) / len(v)) if v else float('nan')
            print('{}: {:.6f}'.format(k, v))
        if save_interval and n_iter % save_interval == 0:
            torch.save([policy.state_dict(), optimizer.state_dict()],
                       os.path.join(save_path, '{}.checkpoint'.format(n_iter)))


def _generate(device, env, dist, policy, ob_scale,
              number_timesteps, gamma, gae_lam, timesteps_per_batch):
    """ Generate trajectories """
    record = ['o', 'a', 'r', 'done', 'logp', 'vpred']
    export = ['o', 'a', 'r', 'logp', 'vpred']
    trajectories = Trajectories(record, export,
                                device, gamma, ob_scale, gae_lam)

    o = env.reset()
    infos = []
    for n in range(1, number_timesteps + 1):
        # sample action
        with torch.no_grad():
            logits, v = policy(scale_ob(o, device, ob_scale))
            pd = dist(logits)
            a = pd.sample()
            logp = pd.log_prob(a).cpu().numpy()
            a = a.cpu().numpy()
            v = v.cpu().numpy()[:, 0]

        # take action in env
        o_, r, done, info = env.step(a)
        for d in info:
            if d.get('episode'):
                infos.append(d['episode'])

        # store batch data and update observation
        trajectories.append(o, a, r, done, logp, v)
        if n % timesteps_per_batch == 0:
            with torch.no_grad():
                ob = scale_ob(o_, device, ob_scale)
                v_ = policy(ob)[1].cpu().numpy()[:, 0] * (1 - done)
            yield trajectories.export(v_) + (infos, )
            infos.clear()
        o = o_

class Trajectories(object):
    """
    Parameters:
    -----------
    record_keys (list[str]): stored keys
    export_keys (list[str]): export keys, must be the subset of record_keys
    device (torch.device): export device
    gamma (float): reward gamma
    gae_lam (float): gae lambda

    Note that `done` describe the next timestep
    """
    SUPPORT_KEYS = {'o', 'a', 'r', 'done', 'logp', 'vpred'}

    def __init__(self, record_keys, export_keys,
                 device, gamma, ob_scale, gae_lam=1.0):
        assert set(record_keys).issubset(Trajectories.SUPPORT_KEYS)
        assert set(export_keys).issubset(record_keys)
        assert {'o', 'a', 'r', 'done'}.issubset(set(record_keys))
        self._keys = record_keys
        self._records = dict()
        for key in record_keys:
            self._records[key] = []
        self._export_keys = export_keys
        self._device = device
        self._gamma = gamma
        self._gae_lam = gae_lam
        self._offsets = {'done': [], 'r': []}
        self._ob_scale = ob_scale

    def append(self, *records):
        assert len(records) == len(self._keys)
        for key, record in zip(self._keys, records):
            if key == 'done':
                record = record.astype(int)
                self._offsets['done'].append(record)
            if key == 'r':
                self._offsets['r'].append(record)
            self._records[key].append(record)

    def __len__(self):
        return len(self._records[self._keys[0]])

    def export(self, next_value=None):
        # estimate discounted reward
        d = self._records['done']
        r = self._records['r']
        n = len(self)
        nenv = r[0].shape[0]
        if 'vpred' in self._keys and self._gae_lam != 1:  # gae estimation
            v = self._records['vpred']
            gae = np.empty((nenv, n))
            last_gae = 0
            for i in reversed(range(n)):
                flag = 1 - d[i]
                delta = r[i] + self._gamma * next_value * flag - v[i]
                delta += self._gamma * self._gae_lam * flag * last_gae
                gae[:, i] = last_gae = delta
                next_value = v[i]
            for i in range(n):
                self._records['r'][i] = gae[:, i] + v[i]
        else:  # discount reward
            if next_value is not None:
                self._records['r'][-1] += self._gamma * next_value
            for i in reversed(range(n - 1)):
                self._records['r'][i] += self._gamma * r[i + 1] * (1 - d[i])

        # convert to tensor
        res = []
        for key in self._export_keys:
            shape = (n * nenv, ) + self._records[key][0].shape[1:]
            data = np.asarray(self._records[key], np.float32).reshape(shape)
            tensor = torch.from_numpy(data).to(self._device)
            if key == 'o':
                tensor.mul_(self._ob_scale)
            res.append(tensor)

        for key in self._keys:
            self._records[key].clear()

        return tuple(res)


def scale_ob(array, device, scale):
    return torch.from_numpy(array.astype(np.float32) * scale).to(device)