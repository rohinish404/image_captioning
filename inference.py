import torch
from train import model
from data_loader import transform_train, test_loader
from PIL import Image

model.load_state_dict(torch.load("checkpoints/model_weights_epoch_20.pth",map_location=device))
model.eval()

def get_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_train(image).unsqueeze(0).to(device)

    print(image_tensor.shape)
    features = model.encoder(image_tensor)
    output, atten_weights = model.decoder.greedy_search(features)
    print('example output:', output)
    return output

def clean_sentence(output):
    vocab = test_loader.dataset.dataset.vocab.itos
    words = [vocab.get(idx) for idx in output]
    words = [word for word in words if word not in ('<start>', ',', '.', '<end>')]
    sentence = " ".join(words)

    return sentence
