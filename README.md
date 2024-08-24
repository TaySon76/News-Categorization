# News-Categorization
Text Classification with TensorFlow

Overview
This project implements a text classification model using TensorFlow and Keras. It includes data preprocessing, model training, evaluation, and prediction functionalities for classifying text data, such as news articles.

Features
  - Load and preprocess data from CSV files.
  - Remove stopwords from text.
  - Tokenize and pad text sequences.
  - Build and train an LSTM-based text classification model.
  - Evaluate model performance using accuracy and loss metrics.
  - Make predictions on new text data.

Requirements
1. Python 3.7+
2. TensorFlow 2.x
3. Keras
4. NumPy
5. Matplotlib


Installation
1. Clone the repository and install the required packages:
bash: git clone https://github.com/TaySon76/News-Categorization.git
cd News-Categorization
pip install -r requirements.txt

Example CSV format:
id,text,label
1,This is a sample text,Category1
2,Another text example,Category2

Update the paths to your CSV files in the script:
csv_file_path = os.path.expanduser("~/data/BBC News Train.csv")
csv_file_path_o = os.path.expanduser("~/data/BBC News Test.csv")

Run the main script to preprocess the data, train the model, and evaluate performance:

The script will:
  - Load and clean the data.
  - Tokenize and pad the text sequences.
  - Build and train the model.
  - Plot training and validation accuracy and loss.
  - Predict categories for test data.

The script outputs:
  - Model summary.
  - Training and validation accuracy and loss plots.
  - Predicted categories for test data.


Customization
  - Stop Words: Customize the list of stop words by editing the stopwords.txt file.
  - Model Configuration: Adjust model parameters (e.g., NUM_WORDS, EMBEDDING_DIM, MAXLEN) as needed in the script.
  - Data Splitting: Modify the TRAINING_SPLIT variable to change the proportion of data used for training.


Files
stopwords.txt: List of stop words for text preprocessing.
requirements.txt: List of Python package dependencies.
