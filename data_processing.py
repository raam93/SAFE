import transformers
import pandas as pd


class DataProcessing:
    """
    A class for processing data using a language model.
    """

    def __init__(self, pipeline_type: str, model_name: str, data_name: str):
        """
        Initializes the DataProcessing class with a specified model and data name.

        Args:
            pipeline_type (str): The type of pipeline to use for processing.
            model_name (str): The name of the model to use for processing.
            data_name (str): The name of the data being processed.
        """
        self.pipeline = transformers.pipeline(
            pipeline_type, model=model_name, pad_token_id=50256
        )
        self.data_name = data_name
        self.chat_history = []

    def prompt_llm(self, user_prompt: str) -> str:
        """
        Prompts the language model with a user's input and appends the response to the chat history.

        Args:
            user_prompt (str): The user's input prompt.

        Returns:
            str: The response from the language model.
        """
        self.chat_history.append({"role": "user", "content": user_prompt})
        prompt = ""
        for turn in self.chat_history:
            role = turn["role"].upper()
            content = turn["content"]
            prompt += f"{role}: {content}\n"

        prompt += "ASSISTANT: "

        max_length = 512
        prompt_tokens = prompt.split()
        if len(prompt_tokens) > max_length:
            prompt = " ".join(prompt_tokens[-max_length:])

        llm_response = self.pipeline(
            prompt, max_new_tokens=300, num_return_sequences=1, pad_token_id=50256
        )[0]["generated_text"]
        self.chat_history.append({"role": "assistant", "content": llm_response})

        return llm_response

    def process_data_for_toxicity(self, input_data: pd.Series, visualize: bool = False):
        """
        Processes input data using the specified prompt and model to evaluate toxicity.

        Args:
            input_data (pd.Series): The data to be processed.
        """
        output_data = []

        for text in input_data:
            self.chat_history = []
            prompt = (
                "Task: You are a toxicity evaluator. Read the post below and evaluate its toxicity.\n"
                "Provide ONLY a 'yes,' 'maybe,' or 'no' answer.\n\n"
                f"Here is the post: {text}"
            )
            llm_toxicity_classification = self.prompt_llm(prompt)
            prompt = (
                f"In your previous response, you answered '{llm_toxicity_classification}'. "
                "Could you explain why you gave that answer in detail?"
            )
            llm_toxicity_explanation = self.prompt_llm(prompt)

            output_data.append(
                {
                    "test": text,
                    "toxicity_classification": llm_toxicity_classification,
                    "toxicity_explanation": llm_toxicity_explanation,
                }
            )
            if visualize:
                self.visualize_chat_history()

        pd.DataFrame(output_data).to_csv(
            f"processed_data/processed_{self.data_name}.csv", index=False
        )

    def visualize_chat_history(self):
        """
        Visualizes the chat history using a simple text-based representation.

        Args:
            chat_history (list): A list of chat history entries.
        """
        for entry in self.chat_history:
            print(f"{entry['role'].upper()}: {entry['content']}")


if __name__ == "__main__":
    input_file = pd.read_json("input_data/implicit_toxicity.json")
    input_data = input_file[0:10]["response"]

    # data_processing = DataProcessing(
    #     pipeline_type="text-generation",
    #     model_name="distilgpt2",
    #     data_name="implicit_toxicity",
    # )

    data_processing = DataProcessing(
        pipeline_type="text2text-generation",
        model_name="google/flan-t5-small",
        data_name="implicit_toxicity",
    )
    data_processing.process_data_for_toxicity(input_data, visualize=True)
