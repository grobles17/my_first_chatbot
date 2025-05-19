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
    def __init__(self, input_size, output_size):
        """
        Defines a simple feedforward neural network for intent classification.

        Parameters:
        - input_size (int): Size of the input vector (i.e. the number of words in the vocabulary).
        - output_size (int): Number of possible output classes (i.e. intents).

        The network architecture:
        - Input layer → 128 neurons
        - Hidden layer → 64 neurons
        - Output layer → output_size neurons (one per intent)
        - ReLU activation is applied between layers to introduce non-linearity.
        - Dropout (50%) is used after each hidden layer to reduce overfitting.
        """
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)          # Second hidden layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer (raw scores for each intent)

        self.relu = nn.ReLU()                  # Activation function: ReLU helps model non-linear patterns
        self.dropout = nn.Dropout(0.5)         # Randomly drops 50% of neurons during training to prevent overfitting

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))             # Layer 1 + ReLU
        x = self.dropout(x)                    # Dropout after first hidden layer
        x = self.relu(self.fc2(x))             # Layer 2 + ReLU
        x = self.dropout(x)                    # Dropout after second hidden layer
        x = self.fc3(x)                        # Final layer (no activation; softmax applied later during loss computation)

        return x


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

    def parse_intents(self)-> None:
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

            for pattern in intent["patterns"]:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent["tag"]))  # Each document is (tokenized pattern, intent tag)

        self.vocabulary = sorted(set(self.vocabulary))  # Remove duplicates and sort vocabulary
    
    def prepare_data(self)-> None:
        """
        Prepares the training data by converting documents into numerical format.

        This method transforms each document into a bag-of-words vector using the instance's
        vocabulary and maps each intent tag to its corresponding index in the intents list.
        The resulting feature matrix (X) and label vector (y) are stored as NumPy arrays
        in self.X and self.y, respectively.

        Result:
            self.X (np.ndarray): A matrix where each row is a binary bag-of-words vector representing a document.
            self.y (np.ndarray): A vector of integer labels, each representing the index of the intent tag.
        """
        bags = []     # List to store feature vectors for each document
        indices =     []  # List to store the intent index (class label) for each document

        for doc in self.documents:
            words = doc[0]  # The list of words from the input pattern
            tag = doc[1]    # The associated intent tag

            bag = self.bag_of_words(words)  # Convert words into a binary bag-of-words vector
            intent_index = self.intents.index(tag)  # Get the index of the intent tag

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)     # Feature matrix: each row represents a document
        self.y = np.array(indices)  # Label vector: each value represents an intent class

    
    def train_model(self, batch_size: int, lr: float, epochs: int)-> None:
        """
        Trains the chatbot classification model using the prepared bag-of-words input and intent labels.

        The model is trained using the Adam optimizer and cross-entropy loss. It learns to map
        bag-of-words input vectors (features) to intent labels (classes) by adjusting weights
        through backpropagation over multiple epochs.

        Parameters:
            batch_size (int): Number of samples to process simultaneously during each training step.
            lr (float): Learning rate that controls how much the model weights are adjusted at each step.
            epochs (int): Number of times to iterate over the entire training dataset.

        Model Output:
            self.model: A trained instance of ChatbotModel ready to make predictions.
        """
        # Convert training data to PyTorch tensors
        X_tensor = torch.tensor(self.X, dtype=torch.float32)  # Input features (binary bag-of-words vectors)
        y_tensor = torch.tensor(self.y, dtype=torch.long)     # Target labels (as integer indices)

        # Wrap tensors in a dataset and loader for mini-batch processing
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Shuffle for stochasticity

        # Initialize the model
        self.model = ChatbotModel(
            input_size=self.X.shape[1],     # Number of input features (size of vocabulary)
            output_size=len(self.intents)   # Number of output classes (intents)
        )

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # Combines softmax + negative log likelihood
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop over epochs
        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()               # Clear previous gradients
                outputs = self.model(batch_X)       # Forward pass: predict class scores
                loss = criterion(outputs, batch_y)  # Compute loss between predicted and true labels

                loss.backward()  # Backward pass: compute gradients via backpropagation
                optimizer.step() # Update model weights based on gradients and learning rate

                running_loss += loss.item()  # Accumulate batch loss for monitoring

            # Print average loss for the epoch
            print(f"Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")


    def save_model(self, model_path, dimensions_path):
        """
        Saves the trained model's weights and essential architecture dimensions.

        Parameters:
            model_path (str): File path to save the model's weights (state_dict).
            dimensions_path (str): Path to save model input and output sizes in JSON format.

        Notes:
            - Only the model weights are saved, not the full model object.
            - The input and output sizes are stored separately to allow proper reinitialization.
        """
        torch.save(self.model.state_dict(), model_path)  # Save model weights only

        # Save input and output dimensions for model reconstruction
        with open(dimensions_path, "w") as f:
            json.dump({
                "input_size": self.X.shape[1],       # Number of input features (vocabulary size)
                "output_size": len(self.intents)     # Number of target classes (intents)
            }, f)
    
    def load_model(self, model_path, dimensions_path):
        """
        Loads the model architecture and previously saved weights from disk.

        Parameters:
            model_path (str): Path to the file containing the saved model weights.
            dimensions_path (str): Path to the JSON file containing input/output sizes.

        Notes:
            - Reinitializes the model architecture using saved dimensions.
            - Loads weights using the 'weights_only=True' flag (requires compatible PyTorch version).
        """
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)

        # Rebuild the model architecture
        self.model = ChatbotModel(
            input_size=dimensions["input_size"],
            output_size=dimensions["output_size"]
        )

        # Load model weights using weights_only flag 
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    def process_message(self, input_message):
        """
        Processes an input message, predicts the intent using the trained model,
        and returns an appropriate response or executes a mapped function.

        Workflow:
            1. Tokenizes and lemmatizes the input message.
            2. Converts the message into a bag-of-words vector.
            3. Feeds the vector to the trained model to predict the intent.
            4. If a function is mapped to the predicted intent, it is executed.
            5. If responses exist for the intent, a random one is returned.
            6. Returns None if no response is defined but a function was executed.

        Parameters:
            input_message (str): The user's message as plain text.

        Returns:
            str or None: A randomly selected response string from the intent's responses,
                        or None if only a function was triggered.
        """
        # Step 1: Preprocess input (tokenize + lemmatize)
        words = self.tokenize_and_lemmatize(input_message)

        # Step 2: Convert processed words into a bag-of-words vector
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)  # Add batch dimension

        # Step 3: Predict the intent using the trained model
        self.model.eval()  # Ensure model is in inference mode
        with torch.no_grad():  # Disable gradient tracking for performance and reduced side-effects
            predictions = self.model(bag_tensor)

        # Step 4: Select the intent with the highest predicted probability
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        # Step 5: If a function is mapped to this intent, execute it
        if self.function_mapping and predicted_intent in self.function_mapping:
            self.function_mapping[predicted_intent]()

        # Step 6: If responses are available, return a random one
        if self.intents_responses.get(predicted_intent):
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None  # No response defined, possibly only a function was run

    
def get_stocks(): #dummy function to test function_mapping
    stocks = ["BABA", "PYPL", "PDD", "MRNA", "GOOGL"]
    print(stocks)

if __name__ == "__main__":
    # Initialize the assistant with a mapping of intents to functions
    # (Note: 'get_stocks' is passed without parentheses to avoid calling it immediately)
    assistant = ChatbotAssistant("intents.json", function_mapping={"stocks": get_stocks})

    # Load and parse the intents file to access responses and tags
    assistant.parse_intents()

    # ----------------------------------------------------------------------
    # First-time setup: train and save the model
    # Uncomment this section only if you haven't trained/saved a model yet.
    # ----------------------------------------------------------------------
    # assistant.prepare_data()
    # assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    # assistant.save_model(model_path="chatbot_model.pth", dimensions_path="dimensions.json")

    # ----------------------------------------------------------------------
    # Load an existing trained model to use it directly
    # ----------------------------------------------------------------------
    assistant.load_model("chatbot_model.pth", "dimensions.json")

    # Start the interactive message loop
    while True:
        message = input("What is your message? ")

        if message.lower() == "quit":  # Use 'quit' to exit the chat
            print("Goodbye!")
            break

        response = assistant.process_message(message)
        print(response)

