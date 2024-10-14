from transformers import pipeline
from nltk import sent_tokenize
import nltk
import torch
import os
import pandas as pd
import numpy as np
import pathlib
import sys

# Set up folder path and utility imports
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(folder_path) + "/../")
from utils import extract_dialogue

# Download NLTK data for sentence tokenization
nltk.download('punkt')

class themeClassifier:
    def __init__(self, theme_list):
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else -1  # CPU is denoted by -1 in transformers
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )
        return theme_classifier

    def get_theme_inference(self, script):
        sentences = sent_tokenize(script)
        # Split into batches of 20 sentences each
        script_batches = []
        batch_size = 20
        for i in range(0, len(sentences), batch_size):
            s20 = " ".join(sentences[i:i+batch_size])
            script_batches.append(s20)
        
        # Run model inference on all batches
        theme_output = self.theme_classifier(
            script_batches[:2],  # Process all batches instead of limiting to [:2]
            self.theme_list,
            multi_label=True
        )
        
        # Data wrangling: aggregate theme scores
        themes = {}
        for batch in theme_output:
            for label, score in zip(batch["labels"], batch["scores"]):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)
        
        # Calculate the average score for each theme 
        themes = {key: np.mean(value) for key, value in themes.items()}
        return themes

    def get_themes(self, dataset_path, save_path=None):
        # Read and return saved output if it already exists
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df
        
        # Load and process the dataset
        df = extract_dialogue(dataset_path)
        df=df.head(2)
        # Inference: classify themes for each dialogue/script
        output_themes = df["Script"].apply(self.get_theme_inference)
        theme_df = pd.DataFrame.from_dict(output_themes.tolist())
        
        # Join the themes dataframe to the original one
        df[theme_df.columns] = theme_df
        
        # Save the output if a save path is provided
        if save_path is not None:
            df.to_csv(save_path, index=False)
        
        return df
