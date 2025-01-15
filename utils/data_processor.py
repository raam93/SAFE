import numpy as np
import pandas as pd

def load_data_from_file(data_path):
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path)
    elif data_path.endswith('.json'):
        data = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, XLSX, or JSON file.")
    return data    

def civil_comments(self, data_path, remove_shorter_text=True, short_text_length=30):
    data = load_data_from_file(data_path)
    
    if remove_shorter_text:
        data = data[(data['comment_text'].str.len() < short_text_length)] 
        
    data = data[['id', 'comment_text', 'target']].rename(columns={'comment_text': 'input_text', 'target': 'given_label'})
    if self.sampling_method == "random":
        data = data.sample(n=self.total_samples, random_state=self.random_state)
    
    return data
    
    
    
    