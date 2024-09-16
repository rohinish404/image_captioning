import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
  def __init__(self):
    super(EncoderCNN, self).__init__()
    resnet = torchvision.models.resnet101(pretrained=True)
    for param in resnet.parameters():
      param.requires_grad_(False)
    modules = list(resnet.children())[:-2]
    self.resnet = nn.Sequential(*modules)

  def forward(self, images):
    features = self.resnet(images)
    # print(features.size())
    batch, feature_maps, size_1, size_2 = features.size()
    features = features.permute(0, 2, 3, 1)
    features = features.view(batch, size_1*size_2, feature_maps)
    return features


class BahdanauAttention(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim=1):
    super(BahdanauAttention, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim

    self.Wa = nn.Linear(self.input_dim, self.hidden_dim)
    self.Ua = nn.Linear(self.hidden_dim, self.hidden_dim)
    self.Va = nn.Linear(self.hidden_dim, self.output_dim)

  def forward(self, features, decoder_hidden):
    #below shapes mismatch so we add a new dimension to do the summation and remove that dim at the end
    decoder_hidden = decoder_hidden.unsqueeze(1)
    # print(decoder_hidden.shape)
    atten_1 = self.Wa(features)
    atten_2 = self.Ua(decoder_hidden)
    # print(atten_1.shape) #(torch.Size([32, 49, 256]))
    # print(atten_2.shape) #(torch.Size([32, 256]))

    atten_tan = torch.tanh(atten_1 + atten_2)
    atten_score = self.Va(atten_tan)

    atten_weight = F.softmax(atten_score, dim=1)
    
    context = torch.sum(atten_weight * features, dim=1)
    atten_weight = atten_weight.squeeze(dim=2)

    return context, atten_weight

class DecoderRNN(nn.Module):
  def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p=0.5):
    super(DecoderRNN, self).__init__()

    self.num_features = num_features
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.sample_temp = 0.5 

    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTMCell(embedding_dim + num_features, hidden_dim)

    self.fc = nn.Linear(hidden_dim, vocab_size)

    self.attention = BahdanauAttention(num_features, hidden_dim)
    self.drop = nn.Dropout(p=p)

    self.init_h = nn.Linear(num_features, hidden_dim)
    self.init_c = nn.Linear(num_features, hidden_dim)

  def forward(self, captions, features, sample_prob=0.0):
    # print(f"captions-{captions.shape}")
    embed = self.embeddings(captions)
    h, c = self.init_hidden(features)

    seq_len = captions.size(1)
    feature_size = features.size(1)
    batch_size = features.size(0)

    outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
    atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)


    for t in range(seq_len):
      # print(f"embed-{embed.shape}")
      sample_prob = 0.0 if t == 0 else 0.5
      use_sampling = np.random.random() < sample_prob
      if use_sampling == False:
        word_embed = embed[:,t,:]
      context, atten_weight = self.attention(features, h)
      # print(f"word_embed - {word_embed.shape}") 
      # print(f"context - {context.shape}") 
      input_concat = torch.cat([word_embed, context], 1)
      h, c = self.lstm(input_concat, (h,c))
      h = self.drop(h)
      output = self.fc(h)
      if use_sampling == True:
        scaled_output = output / self.sample_temp
        scoring = F.log_softmax(scaled_output, dim=1)
        top_idx = scoring.topk(1)[1]
        word_embed = self.embeddings(top_idx).squeeze(1)
      outputs[:, t, :] = output
      atten_weights[:, t, :] = atten_weight
    return outputs, atten_weights
  
  
  def init_hidden(self, features):
    mean_annotations = torch.mean(features, dim=1)
    h0 = self.init_h(mean_annotations)
    c0 = self.init_c(mean_annotations)
    return h0, c0

class EncoderDecoder(nn.Module):
  def __init__(self, num_features, embedding_dim, hidden_dim, vocab_size, p=0.5):
    super(EncoderDecoder, self).__init__()

    self.encoder = EncoderCNN()
    self.decoder = DecoderRNN(
            num_features=num_features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size
        )
  def forward(self, images, captions):
    features = self.encoder(images)
    outputs, atten_weights = self.decoder(captions, features)
    return outputs, atten_weights