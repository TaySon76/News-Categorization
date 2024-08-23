import os, csv, zipfile
import tensorflow as tf, numpy as np
from tensorflow import keras
from keras import layers, preprocessing, Sequential
import matplotlib.pyplot as plt
from collections import Counter
import sys


sys.stdout.reconfigure(encoding='utf-8')
# Generally speaking, we don’t want to represent every unique word in our vocabulary with a token. Often times, 
# low frequency words have a tendency to not possess amy useful information. Unnecessarily increasing the vocabulary size 
# can result in an unprecedented tradeoff in terms of memory and performance costs. Overfitting could also come into play, whereby
# the models learns how to perform accurately on the training data and poorly on the test data. 
NUM_WORDS = 1000

# This parameter determines the dimensionality of the word embeddings learned by the model. 
# A higher EMBEDDING_DIM allows the model to learn more complex relationships between words
# which can be beneficial for tasks that require capturing nuanced semantic meanings. 
# However, a very high EMBEDDING_DIM can lead to overfitting, especially if the dataset is small.
# In terms of NLP with respect to news articles, it is crucial to strike a balance between the data set size and embedding dimension
# because news articles tend to be incredibly nuanced. 
EMBEDDING_DIM = 16

# maximum length of all sequences
MAXLEN = 120

# padding strategy 
PADDING = 'post'

# token to replace out-of-vocabulary words during text_to_sequence() calls
OOV_TOKEN = ""

# proportion of data used for training
TRAINING_SPLIT = .8

csv_file_path = os.path.expanduser("~/data/BBC News Train.csv")
csv_file_path_o = os.path.expanduser("~/data/BBC News Test.csv")

with open(csv_file_path, 'r') as csvfile:

    csvreader = csv.reader(csvfile)

    header = next(csvreader)
    print(f"CSV header:\n {header}")
    
    first_data_point = next(csvreader)
    print(f"First data point:\n {first_data_point}")


with open(csv_file_path_o, 'r') as csvfile:

    csvreader = csv.reader(csvfile)

    header = next(csvreader)
    print(f"CSV header:\n {header}")
    
    first_data_point = next(csvreader)
    print(f"First data point:\n {first_data_point}")

def remove_stop_words(sentence):
    """
    Remove stop words from a given sentence.

    Args:
    - sentence (str): Input sentence.

    Returns:
    - str: Sentence with stop words removed.
    """
    x, l = [], ""
    y = ['against', 'before', 'doing', 'having', "he'd", "he'll", "he's", "here's", 'herself', 'himself', "how's", 'i', "i'd", "i'll", "i'm", "i've", "it's", "let's", 'ought', "she'd", "she'll", "she's", "that's", 'theirs', "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "you'd", "you'll", "you're", "you've"]
    with open(r"C:\Users\awais\OneDrive\Desktop\Year 1 summer\Media\Training Set\stopwords.txt", 'r') as file:
        l += file.read().strip("")

    x = l.split("\n")
    x.extend(y)
    
    sentence = sentence.lower().split()

    for i in range(len(sentence)-1, -1, -1):
        if sentence[i] in x:
            sentence.pop(i)

    return " ".join(sentence)

    
def read_and_clean_from_file(filename):
    """
    Read data from a CSV file and remove stop words from the sentences.

    Args:
    - filename (str): Path to the CSV file.

    Returns:
    - tuple: A tuple containing two lists: sentences and labels.
    """
    sentences, labels = [], []

    with open(filename, 'r') as csvfile:
        next(csv.reader(csvfile, delimiter=',')) 

        for row in csv.reader(csvfile, delimiter=','):
            labels.append(row[2])
            sentences.append(remove_stop_words(row[1]))

    return sentences, labels


sentences, labels = read_and_clean_from_file(csv_file_path)
print('----------------------------------------------------------------------------')
print(f"Number of sentences in the training dataset: {len(sentences)}\n")
print(f"Number of words in the 1st sentence (after removing stopwords). {len(sentences[0].split())}\n")
print(f"Number of labels in the dataset: {len(labels)}\n")
print(f"First 10 labels: {labels[:10]}")
print('----------------------------------------------------------------------------')
def training_validation_split(sentences, labels, training_split):
    """
    Split the input sentences and labels into training and validation sets.

    Args:
    - sentences (list): List of input sentences.
    - labels (list): List of corresponding labels.
    - training_split (float): Fraction of data to use for training (default: 0.8).

    Returns:
    - training_sentences (list): Sentences for training.
    - validation_sentences (list): Sentences for validation.
    - training_labels (list): Labels for training.
    - validation_labels (list): Labels for validation.
    """
    training_split = TRAINING_SPLIT
    training_sentences = sentences[:int(len(sentences) * training_split)]
    training_labels = labels[:int(len(sentences) * training_split)]
    
    validation_sentences = sentences[int(len(sentences) * training_split):]
    validation_labels = labels[int(len(sentences) * training_split):]

    return training_sentences, validation_sentences, training_labels, validation_labels


train_sentences, val_sentences, train_labels, val_labels = training_validation_split(sentences, labels, TRAINING_SPLIT)
print('----------------------------------------------------------------------------')
print(f"Number of sentences for training: {len(train_sentences)} \n")
print(f"Number of labels for training: {len(train_labels)}\n")
print(f"Number of sentences for validation: {len(val_sentences)} \n")
print(f"Number of labels for validation: {len(val_labels)}")
print('----------------------------------------------------------------------------')
def fit_text_vectorizer(train_sentences:list, max_tokens):
    """
    Create and fit a TextVectorization layer to the training sentences.

    Args:
    - train_sentences (list): List of training sentences.
    - max_tokens (int): Maximum number of tokens in the vocabulary.

    Returns:
    - vectorizer (TextVectorization): Fitted TextVectorization layer.
    """
    vectorizer = layers.TextVectorization(max_tokens=max_tokens, output_mode='int', pad_to_max_tokens=False)
    vectorizer.adapt(train_sentences)
    return vectorizer

#Example Usage
vectorizer = fit_text_vectorizer(train_sentences, NUM_WORDS)
train_sequences = vectorizer(np.array(train_sentences))
validation_sequences = vectorizer(np.array(val_sentences))
print('----------------------------------------------------------------------------')
print(f"Shape of padded training sequences: {train_sequences.shape}\n")
print(f"Shape of padded validation sequences: {validation_sequences.shape}")
print('----------------------------------------------------------------------------')
vocabulary = vectorizer.get_vocabulary()

#print(f"Number of words in the vocabulary: {len(vocabulary)}\n")   

#Check index and set w/o vocab words
vocabulary = set(vectorizer.get_vocabulary())

word_index_wo_voc = set()
for text in train_sentences:
    for word in text.split():
        if word.lower() not in vocabulary:
            word_index_wo_voc.add(word.lower())

#print("Out-of-vocabulary words:", word_index_wo_voc)
#print(len(word_index_wo_voc))

word_index_ordered = ' '.join(train_sentences)
word_counts = Counter(word_index_ordered.split())
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
#print(len(sorted_words))
#for word, count in sorted_words:
    #print(f"{word}: {count}")


def seq_and_pad(sentences, tokenizer, padding, maxlen):       
    # convert training sentences to sequences
    sequences = tokenizer(sentences)
    
    # pad the sequences using the correct padding and maxlen
    padded_sequences = preprocessing.sequence.pad_sequences(sequences, 
                        maxlen=maxlen, 
                        padding=padding, 
                        truncating='post')

    return padded_sequences

train_padded_seq = seq_and_pad(train_sentences, vectorizer, PADDING, MAXLEN)
val_padded_seq = seq_and_pad(val_sentences, vectorizer, PADDING, MAXLEN)

def tokenize_labels(all_labels, split_labels): 
    """
    Tokenize the labels using a TextVectorization layer.

    Args:
    - all_labels (list): List of all labels.
    - split_labels (list): List of labels to be tokenized.

    Returns:
    - label_seq_np (numpy.ndarray): Numpy array of tokenized labels.
    """
    label_tokenizer = layers.TextVectorization()
    label_tokenizer.adapt(all_labels)
    label_seq = label_tokenizer(split_labels).numpy()

    label_seq_np = label_seq - 2

    return label_seq_np

train_label_seq = tokenize_labels(all_labels=labels, split_labels=train_labels)
val_label_seq = tokenize_labels(all_labels=labels, split_labels=val_labels)
print('----------------------------------------------------------------------------')
print(f"Shape of tokenized labels of the training set: {train_label_seq.shape}\n")
print(f"Shape of tokenized labels of the validation set: {val_label_seq.shape}\n")
print(f"First 5 labels of the training set:\n{train_label_seq[:5]}\n")
print(f"First 5 labels of the validation set:\n{val_label_seq[:5]}\n")
print('----------------------------------------------------------------------------')

def create_model(num_words, embedding_dim, lstm1_dim, lstm2_dim, num_categories):
    tf.random.set_seed(200)
    model = Sequential([
        layers.Embedding(num_words, embedding_dim),
        layers.Bidirectional(layers.LSTM(lstm1_dim, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(lstm2_dim)),
        layers.Dense(num_categories, activation='softmax')
    ])
    model.build((None, 120))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_model(NUM_WORDS, EMBEDDING_DIM, 32, 16, 5)

model.summary()

history = model.fit(train_padded_seq, train_label_seq, epochs=30, validation_data=(val_padded_seq, val_label_seq))

def evaluate_model(history):

    epoch_accuracy = history.history['accuracy']
    epoch_val_accuracy = history.history['val_accuracy']
    epoch_loss = history.history['loss']
    epoch_val_loss = history.history['val_loss']
    
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(epoch_accuracy)), epoch_accuracy, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(range(0, len(epoch_val_accuracy)), epoch_val_accuracy, 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Training & validation accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Training Loss')
    plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training & validation loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    plt.show()


evaluate_model(history)

def parse_test_data_from_file(filename):

    test_sentences = []

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) 

        for row in reader:
            sentence = row[1]
            sentence = remove_stop_words(sentence)
            test_sentences.append(sentence)

    return test_sentences

test_sentences = parse_test_data_from_file(csv_file_path_o)

print(f"Number of sentences in the test dataset: {len(test_sentences)}\n")
print(f"Number of words in the 1st sentence (after removing stopwords). {len(test_sentences[0].split())}\n")

test_tokenizer = fit_text_vectorizer(test_sentences, NUM_WORDS)
test_padded_seq = seq_and_pad(test_sentences, test_tokenizer, PADDING, MAXLEN)
predictions = model.predict(test_padded_seq)
predicted_classes = predictions.argmax(axis=1)
print(f'Predicted classes:\n\n {predicted_classes}')