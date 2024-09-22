## Image Captioning Model

### Architecture
I've tried to follow the [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) paper as closely as possible.
Current architecture consists of - 
- An Encoder (which usually is a pretrained Convolutional Neural Networks (CNNs)) for getting the image features (Resnet101).
- A Decoder for generating a caption from the features provided by the encoder (RNN or LSTM).
- Attention which helps in attending to a certain part of image while generating captions (Bahdanau attention & Luong Attention)

### Dataset and HyperPrameters
- First 6k images divided into train and test sets of [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.
- Hyperparamters - 
    - embed_size=512
    - hidden_size=256
    - num_features=2048
    - num_epochs = 100
    - vocab_threshold = 3
    - learning_rate = 3e-4
- Criterion = CrossEntropyLoss
- Optimizer = Adam

### Training Details and results
- Trained for 100 epochs on GPU T4 x2. Total training time around 1 hour 15 mins.
- Avg Eval loss = 3.4731
- Avg Eval Perplexity = 32.7512
#### Some Results

With Bahdanau Attention - <br />
<img src="/results/rs_1.png" alt="drawing" width="500"/>

With Luong Attention - <br />
<img src="/results/luong_results/result_1.png" alt="drawing" width="500"/>
<img src="/results/luong_results/result_2.png" alt="drawing" width="500"/>




### Inference
For these types of models(seq2seq), there are different inference algorithms because we can't use forward method for inference. These inference methods tend to much faster in generating outputs (in our case captions). 
Two methods are widely used - 
- Greedy Search
- Beam Search (Not implemented yet)






### Further Improvements
- It closely follows the Show, Attend and Tell paper but not fully. Some things can still be integrated to make this work better. And maybe better evaluation metrics also (BLEU??).
- Using pretrained tokenizer such as Glove, BERT, etc. (i didn't do it because longer training time).
- Implementing Beam Search for inference.
