from data_loader import train_loader, test_loader, dataset
from model import EncoderDecoder
import torch
import numpy as np
import sys
from torch import nn
import os
from torch.nn.parallel import DataParallel
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

embed_size=512
hidden_size=256
num_features=2048
num_epochs = 100
vocab_threshold = 3
learning_rate = 3e-4
vocab_size = len(train_loader.dataset.dataset.vocab)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=EncoderDecoder(num_features = num_features, 
                      embedding_dim = embed_size, 
                      hidden_dim = hidden_size, 
                      vocab_size = vocab_size)

params = list(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-5)


def train_epoch(model, train_loader, criterion, optimizer, device, vocab_size, print_every=100, save_every=1):
    model.train()
    epoch_loss = 0.0
    epoch_perplex = 0.0
    
    for idx, (image, captions) in enumerate(train_loader):
        captions_target = captions[:, 1:].to(device)
        captions_train = captions[:, :-1].to(device)
        image = image.to(device)
        
        optimizer.zero_grad()
        
        outputs, _ = model(captions=captions_train, images=image)
        
        loss = criterion(outputs.view(-1, vocab_size), captions_target.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        perplex = np.exp(loss.item())
        epoch_loss += loss.item()
        epoch_perplex += perplex
        
        stats = f'Epoch train: [{epoch}/{num_epochs}], idx train: [{idx}], Loss train: {loss.item():.4f}, Perplexity train: {perplex:5.4f}'
        
        print(f'\r{stats}', end="")
        sys.stdout.flush()
        
        if (idx+1) % print_every == 0:
            print(f'\r{stats}')
    
    return epoch_loss, epoch_perplex

def evaluate(model, test_loader, criterion, device, vocab_size, print_every=100):
    model.eval()
    total_loss = 0.0
    total_perplex = 0.0
    
    with torch.no_grad():
        for idx, (image, captions) in enumerate(test_loader):
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, :-1].to(device)
            image = image.to(device)
            
            outputs, _ = model(captions=captions_train, images=image)
            
            loss = criterion(outputs.view(-1, vocab_size), captions_target.reshape(-1))
            
            perplex = np.exp(loss.item())
            total_loss += loss.item()
            total_perplex += perplex
            
            stats = f'Epoch eval: [{epoch}/{num_epochs}], idx eval: [{idx}], Loss eval: {loss.item():.4f}, Perplexity eval: {perplex:5.4f}'
            
            print(f'\r{stats}', end="")
            sys.stdout.flush()
            
            if (idx+1) % print_every == 0:
                print(f'\r{stats}')
    
    return total_loss, total_perplex
    
def append_to_file(filename, value):
    with open(filename, 'a') as f:
        f.write(f"{value}\n")


if __name__=="__main__":
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    print_every = 100
    save_every=10
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    log_files = {
        'train_loss': 'logs/train_loss.txt',
        'train_perplex': 'logs/train_perplex.txt',
        'test_loss': 'logs/test_loss.txt',
        'test_perplex': 'logs/test_perplex.txt'
    }

    # Clear existing log files
    for file in log_files.values():
        open(file, 'w').close()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_perplex = train_epoch(model, train_loader, criterion, optimizer, device, vocab_size, print_every)
        eval_loss, eval_perplex = evaluate(model, test_loader, criterion, device, vocab_size, print_every)

        train_loss_avg = train_loss / len(train_loader)
        train_perplex_avg = train_perplex / len(train_loader)
        eval_loss_avg = eval_loss / len(test_loader)
        eval_perplex_avg = eval_perplex / len(test_loader)

        print(f'\nEpoch: {epoch}')
        print(f'Avg. Train Loss: {train_loss_avg:.4f}, Avg. Train Perplexity: {train_perplex_avg:5.4f}')
        print(f'Avg. Eval Loss: {eval_loss_avg:.4f}, Avg. Eval Perplexity: {eval_perplex_avg:5.4f}\n')

        append_to_file(log_files['train_loss'], train_loss_avg)
        append_to_file(log_files['train_perplex'], train_perplex_avg)
        append_to_file(log_files['test_loss'], eval_loss_avg)
        append_to_file(log_files['test_perplex'], eval_perplex_avg)
        
        if epoch % save_every == 0 or epoch == num_epochs:
            if isinstance(model, DataParallel):
                torch.save(model.module.state_dict(), f"checkpoints/model_weights_epoch_{epoch}.pth")
            else:
                torch.save(model.state_dict(), f"checkpoints/model_weights_epoch_{epoch}.pth")
            print(f"Model weights saved for epoch {epoch}")

    print('Training completed.')


