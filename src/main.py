import argparse
import time
import math
import torch
import torch.nn as nn
import corpus
import model
torch.manual_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def args_parse():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='../input', # /input
                        help='location of the data corpus')
    parser.add_argument('--embed_size', type=int, default=400)
    parser.add_argument('--n_hid', type=int, default=400)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=20)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save', type=str,  default='../output/model_test.pt')
    parser.add_argument('--opt', type=str,  default='SGD',
                        help='SGD, Adam, Momentum')
    args = parser.parse_args()
    return args

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    return data, target


def evaluate(data_source):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


def train():

    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        total_loss += loss.data

        if batch % interval == 0 and batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return loss

if __name__ == "__main__":
    args = args_parse()
    corpus = corpus.Corpus(args.data)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size) # size(total_len//bsz, bsz)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    # Build the model
    interval = 200 # interval to report
    ntokens = len(corpus.dictionary) # 10000
    model = model.RNNModel(ntokens, args.embed_size, args.n_hid, args.n_layers, args.dropout)

    print(model)
    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    best_val_loss = None
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    if args.opt == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
        lr = 0.001
    if args.opt == 'Momentum':
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss = train()
            val_loss = evaluate(val_data)
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.2f} | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | train ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            train_loss, val_loss, math.exp(val_loss), math.exp(train_loss)))
            print('-' * 80)
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                if args.opt == 'SGD' or args.opt == 'Momentum':
                    lr /= 4.0
                    for group in opt.param_groups:
                        group['lr'] = lr

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

