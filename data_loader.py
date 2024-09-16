import torch
import os
import pandas as pd 
import spacy 
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader
from PIL import Image 
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms

spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
  def __init__(self, freq_threshold):
    self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
    self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    self.freq_threshold = freq_threshold

  def __len__(self):
    return len(self.itos)

  @staticmethod
  def tokenizer_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

  def build_vocabulary(self, sentence_list):
    frequencies = {}
    idx = 4

    for sentence in sentence_list:
      for word in self.tokenizer_eng(sentence):
        if word not in frequencies:
          frequencies[word] = 1
        else:
          frequencies[word] += 1

        if frequencies[word] == self.freq_threshold:
          self.stoi[word] = idx
          self.itos[idx] = word
          idx += 1

  def numericalize(self, text):
    tokenized_text = self.tokenizer_eng(text)

    return [
        self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
        for token in tokenized_text
    ]

class ImageCapDataset():
  def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
    self.root_dir = root_dir
    self.df = pd.read_csv(captions_file)
    self.transform = transform

    self.imgs = self.df["image"]
    self.captions = self.df["caption"]

    self.vocab = Vocabulary(freq_threshold)
    self.vocab.build_vocabulary(self.captions.tolist())
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    caption = self.captions[index]
    img_id = self.imgs[index]

    img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

    if self.transform is not None:
      img = self.transform(img)

    numericalized_caption = [self.vocab.stoi["<SOS>"]]
    numericalized_caption += self.vocab.numericalize(caption)
    numericalized_caption.append(self.vocab.stoi["<EOS>"])

    return img, torch.tensor(numericalized_caption)
  
class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        caption = [item[1] for item in batch]
        caption = pad_sequence(caption, batch_first=True, padding_value=self.pad_idx)
        return imgs, caption

def get_loader(root_folder, annotation_file, transform, split, batch_size=32, num_workers=4, pin_memory=True, shuffle=True):
    dataset = ImageCapDataset(root_folder, annotation_file, transform=transform)

    subset_indices = list(range(6000))
    subset = Subset(dataset, subset_indices)

    train_size = int(0.8 * len(subset_indices))
    test_size = len(subset_indices) - train_size


    train_indices, test_indices = train_test_split(subset_indices, train_size=train_size, test_size=test_size, random_state=42)

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    if split == 'train':
        split_dataset = train_subset
    elif split == 'test':
        split_dataset = test_subset
    else:
        raise ValueError("Split must be one of 'train', 'val', or 'test'")
    

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
          dataset=split_dataset,
          batch_size=batch_size,
          num_workers=num_workers,
          shuffle=shuffle if split == 'train' else False,
          pin_memory=pin_memory,
          collate_fn=CustomCollate(pad_idx=pad_idx),
      )
    return loader, dataset

transform_train = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ]
)

train_loader, dataset = get_loader(
        "flickr8k/Images/", "flickr8k/captions.txt", transform=transform_train, split='train'
    )
test_loader, dataset = get_loader(
        "flickr8k/Images/", "flickr8k/captions.txt", transform=transform_train, split='test'
    )



