import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import requests
import os

class FeatureExtractor:
    def __init__(self, EMBEDDING_MODELS_DICT, model_name="Xenova/all-MiniLM-L6-v2"):
        print(f"selected model is {model_name}")
        model_url = EMBEDDING_MODELS_DICT.get(model_name)
        model_dir_path = "../models"
        model_path = f"{model_dir_path}/{model_name.replace('/','_')}"
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        if not os.path.exists(model_path):
            print("Downloading ONNX model...")
            response = requests.get(model_url)
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("ONNX model downloaded.")

        # Load the ONNX model
        self.ort_session = ort.InferenceSession(model_path)
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def extract_features(self, text):
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True)
        # Prepare the input_feed with all required inputs
        input_feed = {
            self.ort_session.get_inputs()[0].name: inputs["input_ids"],
            self.ort_session.get_inputs()[1].name: inputs["attention_mask"],
        }

        # Check if 'token_type_ids' is required by the model
        if len(self.ort_session.get_inputs()) > 2:
            input_feed[self.ort_session.get_inputs()[2].name] = inputs.get("token_type_ids", None)

        # Run inference
        outputs = self.ort_session.run(None, input_feed)
        # return outputs[0], inputs["attention_mask"]
        # Squeeze the batch dimension to ensure shape [seqLength, embedDim]
        return np.squeeze(outputs[0], axis=0), inputs["attention_mask"].squeeze(axis=0)


    @staticmethod
    def mean_pooling(last_hidden_state, attention_mask):
        """
        Perform mean pooling of the last hidden state followed by a normalization step.

        Args:
            last_hidden_state (np.ndarray): Tensor of shape [batchSize, seqLength, embedDim].
            attention_mask (np.ndarray): Tensor of shape [batchSize, seqLength].

        Returns:
            np.ndarray: A new array of shape [batchSize, embedDim].
        """
        batch_size, seq_length, embed_dim = last_hidden_state.shape

        # Create an empty array for the output
        returned_data = np.zeros((batch_size, embed_dim), dtype=last_hidden_state.dtype)

        # Iterate over the batch and embedding dimensions
        for i in range(batch_size):
            for k in range(embed_dim):
                sum_ = 0.0
                count = 0

                # Iterate over the sequence length
                for j in range(seq_length):
                    attn = float(attention_mask[i, j])  # Convert attention mask to float (0 or 1)
                    sum_ += last_hidden_state[i, j, k] * attn
                    count += attn

                # Avoid division by zero
                avg = sum_ / count if count > 0 else 0.0
                returned_data[i, k] = avg

        # Normalize the returned data using L2 norm
        norms = np.linalg.norm(returned_data, axis=1, keepdims=True)
        normalized_data = returned_data / np.clip(norms, a_min=1e-10, a_max=None)  # Prevent division by zero

        return normalized_data

    def get_embeddings(self, texts):
        """
        Extract and pool embeddings for a batch of texts.

        Args:
            texts (list): List of input texts.

        Returns:
            np.ndarray: Pooled and normalized embeddings of shape [batchSize, embedDim].
        """
        last_hidden_states, attention_masks = [], []
        max_seq_length = 0

        # Extract features for each text
        for text in texts:
            hidden_state, attention_mask = self.extract_features(text)
            last_hidden_states.append(hidden_state)
            attention_masks.append(attention_mask)
            max_seq_length = max(max_seq_length, hidden_state.shape[0])  # Update max sequence length

        # Pad last_hidden_states and attention_masks to max_seq_length
        padded_last_hidden_states = []
        padded_attention_masks = []
        for hidden_state, attention_mask in zip(last_hidden_states, attention_masks):
            seq_length, embed_dim = hidden_state.shape

            # Pad hidden_state
            padded_hidden_state = np.zeros((max_seq_length, embed_dim))
            padded_hidden_state[:seq_length, :] = hidden_state  # Copy original values
            padded_last_hidden_states.append(padded_hidden_state)

            # Pad attention_mask
            padded_attention_mask = np.zeros((max_seq_length,))
            padded_attention_mask[:seq_length] = attention_mask  # Copy original values
            padded_attention_masks.append(padded_attention_mask)

        # Stack to create batch tensors
        last_hidden_state = np.stack(padded_last_hidden_states)  # [batchSize, maxSeqLength, embedDim]
        attention_mask = np.stack(padded_attention_masks)        # [batchSize, maxSeqLength]

        # Perform mean pooling
        return self.mean_pooling(last_hidden_state, attention_mask)