import torch
from train import model
from data_loader import transform_train, test_loader
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("path_to_checkpoint",map_location=device))
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

if __name__=="__main__":
    image_path = "path_to_image"
    output_text = get_caption(image_path)
    image = Image.open("/kaggle/input/flickr8k/Images/1000268201_693b08cb0e.jpg").convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    clean_sentence(output_text)

