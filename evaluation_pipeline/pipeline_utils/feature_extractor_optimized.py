import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import pandas as pd
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForFeatureExtraction

class FeatureExtractor:
    def __init__(self, model_name="Xenova/all-MiniLM-L6-v2", quantized=True, pooling='mean'):
        print(f"Selected model is {model_name}")
        # Load the ONNX model
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.quantized = quantized
        self.pooling = pooling
        if quantized:
            file_name_suffix = "_quantized.onnx"
        else:
            file_name_suffix = ".onnx"
        self.model = ORTModelForFeatureExtraction.from_pretrained(model_name, subfolder = "onnx", file_name=f"model{file_name_suffix}", trust_remote_code=True)
        print(self.model.model_save_dir)
        self.onnx_extractor = pipeline("feature-extraction",
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        truncation=True,
                                        max_length=100)


    def extract_features(self, text: str):
        attention_mask = self.tokenizer(text)['attention_mask']
        embeddings = self.onnx_extractor(text)
        embeddings = np.squeeze(np.array(embeddings), axis=0)
        return embeddings, attention_mask

    def mean_pooling(self, embeddings, attention_mask):
        expanded_attention_mask = np.expand_dims(attention_mask, axis=-1)
        masked_embeddings = embeddings * expanded_attention_mask
        # mean pooled masked
        sum_embeddings = np.sum(masked_embeddings, axis=0)
        sum_mask = np.sum(expanded_attention_mask, axis=0)
        sum_mask = np.maximum(sum_mask, 1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled

    def cls_pooling(self, embeddings):
        return embeddings[0]
    
    def get_embedding(self, text):
        embeddings, attention_mask = self.extract_features(text)

        # pooling
        if self.pooling == 'mean':
            pooled_embedding = self.mean_pooling(embeddings=embeddings,attention_mask=attention_mask)
        elif self.pooling == 'cls':
            pooled_embedding = self.cls_pooling(embeddings=embeddings)

         # Normalize the returned data using L2 norm
        norms = np.linalg.norm(pooled_embedding, axis=0)
        normalized_data = pooled_embedding / np.clip(norms, a_min=1e-10, a_max=None)  # Prevent division by zero

        return normalized_data

