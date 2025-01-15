import pandas as pd
import types
from safe.utils import data_processor
import json
import os

class DataLoader:
    """A class to load input data based on user input."""

    def __init__(self, data_name, 
                 sampling_method="random", 
                 total_samples=10000,
                 random_state=17,
                 save_intermediate=False):
        """Init method

        :data_name: Name as stored in utils/data_path_map.
        :sampling_method: Any python sampling function name (random, stratified, etc.). Defaults to None.
        TODO: Add support for different sampling strategies depending on new data usage.
        :total_samples: Total samples to be loaded. Defaults to 10,000.
        :random_state: Random state for sampling. Defaults to 17.
        :save_intermediate: Save processed input data for feeding it to LMs. Defaults to False. 
        """
        self.data_name = data_name
        self.sampling_method = sampling_method
        self.total_samples = min(total_samples, 10000)
        self.random_state = random_state
        self.save_intermediate = save_intermediate
        func = getattr(data_processor, data_name)
        self.data_processing_func = types.MethodType(func, self)

    def load(self, **kwargs):
        """Load data based on data_name."""
        
        with open("safe/utils/data_path_map.json", "r") as file:
            data_path_map = json.load(file)
            
        data_path = data_path_map[self.data_name]
        data = self.data_processing_func(data_path, **kwargs)
        if self.save_intermediate:
            data.to_csv("safe/input_data/processed_data/processed_"+self.data_name, index=False)
        return data