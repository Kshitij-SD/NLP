import glob
import os
import pandas as pd
import re

def extract_dialogue(file_path):
    # Get all .srt files in the directory
    sub_paths = glob.glob(os.path.join(file_path, '*.srt')) 
    movie_name = []
    scripts = []

    for path in sub_paths:
        dialogue_lines = [] 
        
        # Open the subtitle file and read lines
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # Skip lines that are just numbers or contain timestamps
                if not re.match(r'^\d+$', line) and not re.match(r'^\d{2}:\d{2}:\d{2}', line):
                    clean_line = line.strip()
                    if clean_line:
                        dialogue_lines.append(clean_line)
        
        # Join all dialogue lines into a single script string
        script = " ".join(dialogue_lines)
        scripts.append(script)
        
        # Extract the movie name from the file path using os.path.basename
        movie = os.path.basename(path)
        movie = re.split(r'1080p|720p|\.', movie)[0].replace("-", " ").strip()
        movie_name.append(movie)

    # Create a DataFrame with the movie names and scripts
    df = pd.DataFrame({"Movie": movie_name, "Script": scripts})
    return df
