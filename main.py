# train.py
import torch
import numpy as np
import pandas as pd
# yes, we use tensorflow
# but not for training the model!
import tensorflow as tf
from sklearn import metrics
import config
import dataset
import engine
import lstm
from sklearn.model_selection import train_test_split

def load_vectors():
    embeddings_dict = {}
    with open("glove.6B.300d.txt", 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict
def create_embedding_matrix(word_index, embedding_dict):
 # """
 # This function creates the embedding matrix.
 # :param word_index: a dictionary with word:index_value
 # :param embedding_dict: a dictionary with word:embedding_vector
 # :return: a numpy array with embedding vectors for all known words
 # """
    #initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # loop over all the words
    for word, i in word_index.items():
    # if word is found in pre-trained embeddings,
    # update the matrix. if the word is not found,
     # the vector is zeros!

        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix

def run(df, fold):
 # """
 # Run training and validation for a given fold
 # and dataset
 # :param df: pandas dataframe with kfold column
 # :param fold: current fold, int
 # """
 # fetch training dataframe
    train_df = df[df.kfold != fold].reset_index(drop=True)
 # fetch validation dataframe
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    print("Fitting tokenizer")
 #tf.keras for tokenization

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.clean.values.tolist())
 # convert training data to sequences
    xtrain = tokenizer.texts_to_sequences(train_df.clean.values)
 # similarly convert validation data to
 # sequences
    xtest = tokenizer.texts_to_sequences(valid_df.clean.values)
 # zero pad the training sequences given the maximum length
 # this padding is done on left hand side
 # if sequence is > MAX_LEN, it is truncated on left hand side too
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(
    xtrain, maxlen=config.MAX_LEN
    )
 # zero pad the validation sequences
    xtest = tf.keras.preprocessing.sequence.pad_sequences(
    xtest, maxlen=config.MAX_LEN
    )
 # initialize dataset class for training
    train_dataset = dataset.IMDBDataset(
    clean=xtrain,
    ticket_type=train_df.ticket_type.values
    )
 # create torch dataloader for training
 # torch dataloader loads the data using dataset
 # class in batches specified by batch size
    train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAIN_BATCH_SIZE,
    num_workers=2
    )
 # initialize dataset class for validation
    valid_dataset = dataset.IMDBDataset(
    clean=xtest,
    ticket_type=valid_df.ticket_type.values
    )

 # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1
    )
    print("Loading embeddings")
 # load embeddings as shown previously
    embedding_dict = load_vectors()
    embedding_matrix = create_embedding_matrix(
    tokenizer.word_index, embedding_dict
    )
 #create torch device
    device = torch.device("cpu")
 # fetch LSTM model
    model = lstm.LSTM(embedding_matrix)
 # send model to device
    model.to(device)

 # initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Training Model")
 # set best accuracy to zero
    best_accuracy = 0
 # set early stopping counter to zero
    early_stopping_counter = 0
 # train and validate for all epochs
 # train and validate for all epochs


    for epoch in range(config.EPOCHS):
        # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        # validate
        outputs, ticket_type = engine.evaluate(
            valid_data_loader, model, device
        )
        # threshold of 0.5
        # using linear layer
        outputs = np.array(outputs) >= 0.5
        # calculate accuracy
        accuracy = metrics.accuracy_score(ticket_type, outputs)
        print(
            f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}"
        )
  # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break
if __name__ == "__main__":
    # load data
    df = pd.read_csv('output4.csv')
    # train for all folds
    run(df, fold=0)
    run(df, fold=1)
    run(df, fold=2)
    run(df, fold=3)
    run(df, fold=4)
