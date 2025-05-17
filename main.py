import os, json, random
import nltk
import numpy as np 

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# nltk.download("punkt_tab") #Only first time you run nltk

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size): #input and output variables to make it flexible
        """Architecture of neural network. You define an input layer size with first variabel input_size, and an output size
        (the length of the output vector) with variable named output_size. It produces a neural network with 2 hidden layers
        of 128 and 64 neurons"""
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128) #fc1 = "fully connected layer 1" i.e. first hidden layer 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size) 
        self.relu = nn.ReLU() #activation function ReLU -> f(x)= max(0,x) meaning if x < 0, output = 0, otherwise output = x
        #ReLu is used "to break linearity" - investigate this
        self.dropout = nn.Dropout(0.5) #dropout = 50%
    
    def forward(self, x):
        x = self.relu(self.fc1(x)) #get x into the fc1, and then aply the relu function
        x = self.dropout(x) #apply dropout and pass to fc 2
        x = self.relu(self.fc2(x)) #same for fc2
        x = self.dropout(x)
        x = self.fc3(x) #lastly we don't use relu here cause for the output we want to use softmax

class ChatbotAssistant:

    def __init__(self, intents_path, function_mapping=None):
        """
        Initializes the chatbot model with data and optional intent-function mapping.

        Parameters:
            intents_path: Path to the JSON file containing the user's intent definitions.
            function_mapping (optional): A dictionary mapping intent names to functions
                                        to be executed when that intent is recognized.

        Attributes:
            model: The machine learning model (initially None).
            intents_path: Path to the intents JSON file.
            documents: A list to store tokenized training phrases and their associated intents.
            vocabulary: The set of unique lemmatized tokens across all training data.
            intents: A list of recognized intent names.
            intents_responses: A dictionary mapping each intent to a list of possible responses.
            function_mapping: Mapping of intents to executable functions.
            X: Feature matrix (bag-of-words vectors for each training example).
            y: Label vector corresponding to intent classes.
        """
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mapping = function_mapping

        self.X = None  # Feature matrix
        self.y = None  # Label vector

    
    @staticmethod  # This method can be called on the class without an instance and does not access instance attributes or methods.
    def tokenize_and_lemmatize(text: str)-> list[str]:
        """
        Tokenizes and lemmatizes the input text.

        This method performs the following steps:
        1. Tokenizes the input string into individual words.
        2. Converts all tokens to lowercase.
        3. Lemmatizes each token to its base form (e.g., "running" → "run", "was" → "be").

        Parameters:
            text (str): The input text to process.

        Returns:
            List[str]: A list of lemmatized word tokens.
        """
        lemmatizer = nltk.WordNetLemmatizer()  # Create a lemmatizer instance

        words = nltk.word_tokenize(text)  # Tokenize the text into words
        words = [lemmatizer.lemmatize(token.lower()) for token in words]
        # Convert to lowercase and lemmatize each token

        return words

    def bag_of_words(self, words: list[str]) -> list[int]:
        """
        Converts a list of input words into a binary bag-of-words vector
        based on the instance's vocabulary.

        For each word in the instance's vocabulary, this method checks whether it is present
        in the input list of words. It returns a list of 1s and 0s indicating the presence
        (1) or absence (0) of each vocabulary word.

        Parameters:
            words (list[str]): List of lemmatized input words (e.g., from user input).

        Returns:
            list[int]: A binary vector representing the presence of vocabulary words in the instance.
        """
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        """
        Parses the intents JSON file and extracts training data.

        This method reads the intents from the specified JSON file, tokenizes and lemmatizes
        each pattern, and builds internal structures needed for training a chatbot model:
        - A list of all unique intents
        - A mapping of intents to their possible responses
        - A list of training documents, where each document is a tuple of tokenized words and an intent tag
        - A sorted vocabulary of all unique lemmatized words

        Raises:
            FileNotFoundError: If the intents file does not exist.
        """
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)
        else:
            raise FileNotFoundError(f"Intents file not found at path: {self.intents_path}")

        for intent in intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]

            for pattern in intent["pattern"]:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent["tag"]))  # Each document is (tokenized pattern, intent tag)

        self.vocabulary = sorted(set(self.vocabulary))  # Remove duplicates and sort vocabulary
    
    def prepare_data(self):
        """Prepares data to be used for training, stores it in self.X and self.y"""
        bags = []
        indices = []

        for doc in self.documents:
            words = doc[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(doc[1]) #the intent index is the position in the intents list where the tag is (doc[1])
            #"What is the position in the intents of the tag from the doc?"

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags) #X will be the matrix of all the words (coded in 1s and 0s) from the docs
        self.y = np.array(indices) #Y will be the output vectors of the correct predictions 
    
    def train_model(self, batch_size, lr, epochs):
        """lr = learning rate
        epochs = how many repetitions of the same data
        batch_size = how many instances to process in parallel"""
        X_tensor = torch.tensor(self.X, dtype=torch.float32) #bag representation of all the phrases
        y_tensor = torch.tensor(self.y, dtype=torch.long) #we are using ints for the correct clasification

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(input_size=self.X.shape[1], output_size=len(self.intents)) #output has to be a probability for each intent, therefore we pass the legth of all lenghts to have a space for each one
        #X.shape[0] would be the amount of bags we have, X.shape[1] is the size of an individual bag, and we know this depends 
        #on the amount of words we have (for each word we have a 0/1)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X) #We see what our current state model gives out on the input
                loss = criterion(outputs, batch_y) #And we compare it to what it should be and get the loss
                loss.backward() #we backpropagate the error 
                optimizer.step() #This step size depends on the lr we gave before
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss/len(loader)}") #we print the loss ofr each epoch
    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, "w") as f:
            json.dump( {"input_size": self.X.shape(), "output_size": len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.loads(f)
            self.model = ChatbotModel(dimensions["input_size"], dimensions["output_size"]) 
            #When we load it, dimensions["input_size"] will be what we saved in save_model() as self.X.shape()
            #and dimensions["output_size"] ehat we saved as len(self.intents)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
    
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(input_message)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad(): #no grad cause we are not training anymore
            predictions = self.model(bag_tensor)
        
        predicted_class_index = torch.argmax(predictions, dim=1) #Chooses the index with highest probability
        predicted_intent = self.intents[predicted_class_index] #we get the intent eith the predicted index
        
        if self.function_mapping: #if we have functions mapped defined in the constructor
            if predicted_intent in self.function_mapping:
                self.function_mapping[predicted_intent]() #we call the mapped function
        
        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent]) #if we have multiple possible responses choose one ranodmly
        else: 
            return None #just in case I have no response, only a function mapped
        


