import collections
import os
import random
import time

import torch
from dreamcoder.utilities import testTrainSplit
from dreamcoder.domains.list.batchify import get_batches
from dreamcoder.domains.list.meter import AverageMeter
from dreamcoder.domains.list.model import VAE, MMD_VAE
from dreamcoder.domains.list.utils import set_seed, logging, load_sent
from dreamcoder.domains.list.vocab import Vocab
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCHS = 100


def evaluate(model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs, targets in batches:
            _, losses = model.autoenc(inputs, targets, use_decoder=True)
            for k, v in losses.items():
                meters[k].update(v.item(), inputs.size(1))
    loss = model.loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters


def plot_hist(x, label, alpha=1.):
    plt.hist(x, weights=np.ones_like(x) / len(x), label=label, bins=20, alpha=alpha)


def train_model(save_dir, raw_data, epochs=EPOCHS):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, 'log.txt')
    valid_sents, train_sents = testTrainSplit(raw_data, 0.7)
    # Prepare data
    logging('# train sents {}, tokens {}'.format(
        len(train_sents), sum(len(s) for s in train_sents)), log_file)
    logging('# valid sents {}, tokens {}'.format(
        len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
    vocab = Vocab(raw_data)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed(1111)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = MMD_VAE(vocab).to(device)

    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)

    train_batches, _ = get_batches(train_sents, vocab, device=device, batch_size=64)
    valid_batches, _ = get_batches(valid_sents, vocab,  device=device, batch_size=64)
    best_val_loss = None
    best_train_mmd_all = None
    best_valid_mmd_all = None
    best_model_path = os.path.join(save_dir, 'best_discriminator.pt')
    for epoch in range(epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            inputs, targets = train_batches[idx]
            _, losses = model.autoenc(inputs, targets, use_decoder=True)
            losses['loss'] = model.loss(losses)
            model.step(losses)
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % 10 == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(indices))
                for k, meter in meters.items():
                    log_output += ' {} {:.2f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)

        valid_meters = evaluate(model, valid_batches)
        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.4f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            ckpt = {'model': model.state_dict()}
            torch.save(ckpt, best_model_path)
            best_val_loss = valid_meters['loss'].avg
        logging(log_output, log_file)
    logging('Done training', log_file)
    ckpt = torch.load(best_model_path)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    return model