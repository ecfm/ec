import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dreamcoder.domains.list.batchify import get_batch


def reparameterize(mu, logvar):
    # [dim_z, batch_size]
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

# def loss_kl_all(mu, logvar): # kl's of the whole batch
#     return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1) / len(mu)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, dim_emb, dim_h, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab.size, dim_emb)
        self.proj = nn.Linear(dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)


class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, dim_emb=64, dim_h=64, dim_z=16, lr=0.0005, nlayers=2, dropout=0.1):
        super().__init__(vocab, dim_emb, dim_h)
        self.drop = nn.Dropout(dropout)
        self.E = nn.LSTM(dim_emb, dim_h, nlayers,
            dropout=dropout if nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(dim_emb, dim_h, nlayers,
            dropout=dropout if nlayers > 1 else 0)
        self.h2mu = nn.Linear(dim_h*2, dim_z)
        self.h2logvar = nn.Linear(dim_h*2, dim_z)
        self.z2emb = nn.Linear(dim_z, dim_emb)
        self.opt = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        _, (h, _) = self.E(input)
        h = torch.cat([h[-2], h[-1]], 1)
        return self.h2mu(h), self.h2logvar(h)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, input, is_train=False):
        mu, logvar = self.encode(input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss

    def loss(self, losses):
        return losses['rec']

    def autoenc(self, inputs, targets, is_train=False):
        _, _, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is


class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, dim_emb=64, dim_h=64, dim_z=16):
        super().__init__(vocab, dim_emb, dim_h, dim_z)
        self.lambda_kl = 1

    def loss(self, losses):
        return losses['rec'] + self.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}


class MMD_VAE(DAE):
    def __init__(self, vocab, dim_emb=64, dim_h=64, dim_z=16):
        super().__init__(vocab, dim_emb, dim_h, dim_z)
        self.dim_z = dim_z

    def loss(self, losses):
        return losses['rec'] + losses['mmd']

    def reparameterize_samples(self, mu, logvar):
        eps = torch.randn(200, self.dim_z, requires_grad=False).to('cuda')
        std = torch.exp(0.5 * logvar)
        return eps.mul(std).add_(mu)


    def sample_mmd(self, mu, logvar):
        return MMD(torch.randn(200, self.dim_z, requires_grad=False).to('cuda'),
                   self.reparameterize_samples(mu, logvar)).item()

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, z, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'mmd': MMD(torch.randn(200, self.dim_z, requires_grad=False).to('cuda'), z),
                'mmd_all': [self.sample_mmd(mu[i, :], logvar[i, :])
                            for i in range(inputs.shape[1])]}

    def get_weights(self, extractor, examples):
        data = extractor.get_data(examples)
        if examples is None or len(examples) == 0:
            return None, None
        inputs, targets = get_batch(data, self.vocab, 'cuda')
        mu, logvar = self.encode(inputs)
        z = reparameterize(mu, logvar)
        mmd = MMD(torch.randn(200, self.dim_z, requires_grad=False).to('cuda'), z).item()
        return 1-mmd, mmd

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()