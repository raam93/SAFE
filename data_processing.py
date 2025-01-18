import transformers
import pandas as pd


class DataProcessing:
    """
    A class for processing data using a language model.
    """

    def __init__(self, model_name: str, data_name: str):
        """
        Initializes the DataProcessing class with a specified model and data name.

        Args:
            model_name (str): The name of the model to use for processing.
            data_name (str): The name of the data being processed.
        """
        self.model = transformers.pipeline(
            "text-generation", model=model_name, max_new_tokens=75, pad_token_id=50256
        )
        self.data_name = data_name

    @staticmethod
    def _prepare_input(prompt: str, text: str) -> str:
        """
        Combines a prompt and some text into an input for the LLM based on the specified format.

        Args:
            prompt (str): The prompt to guide the LLM.
            text (str): The text to combine with the prompt.

        Returns:
            str: The combined input string.
        """
        return f"{prompt}\n{text}"

    def process_data(self, prompt: str, input_data: pd.Series):
        """
        Processes input data using the specified prompt and model.

        Args:
            prompt (str): The prompt to guide the LLM.
            input_data (pd.Series): The data to be processed.
        """
        output_data = []
        for text in input_data:
            input_string = self._prepare_input(prompt, text)
            model_output = self.model(input_string)
            output_data.append({"input": input_string, "output": model_output})
        pd.DataFrame(output_data).to_csv(
            f"processed_data/processed_{self.data_name}.csv", index=False
        )


if __name__ == "__main__":
    input_file = pd.read_json("input_data/implicit_toxicity.json")
    input_data = input_file[0:10]["response"]

    data_processing = DataProcessing(
        model_name="distilgpt2", data_name="implicit_toxicity"
    )
    data_processing.process_data("Could this post be toxic?", input_data)
