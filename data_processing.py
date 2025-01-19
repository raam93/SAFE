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
            model_name (str): The name of the model to use for processing.
            data_name (str): The name of the data being processed.
        """
        self.pipeline = transformers.pipeline(
            pipeline_type, model=model_name, max_new_tokens=75, pad_token_id=50256
        )
        self.data_name = data_name

    def prompt_llm(self, chat_history, user_prompt):
        chat_history.append({"role": "user", "content": user_prompt})
        prompt = ""

        for turn in chat_history:
            role = turn["role"].upper()
            content = turn["content"]
            prompt += f"{role}: {content}\n"

        prompt += "ASSISTANT: "
        llm_response = self.pipeline(
            prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=50256
        )[0]["generated_text"]
        chat_history.append({"role": "assistant", "content": llm_response})

        return chat_history, llm_response

    def process_data_for_toxicity(self, input_data: pd.Series):
        """
        Processes input data using the specified prompt and model.

        Args:
            input_data (pd.Series): The data to be processed.
        """
        output_data = []
        chat_history = []

        for text in input_data:
            prompt = (
                "Task: You are a toxicity evaluator. Read the post below and evaluate its toxicity.\n"
                "Provide ONLY a 'yes,' 'maybe,' or 'no' answer.\n\n"
                f"Here is the post: {text}"
            )
            chat_history, llm_toxicity_classification = self.prompt_llm(
                chat_history, text
            )
            prompt = (
                f"In your previous response, you answered '{llm_toxicity_classification}'. "
                "Could you explain why you gave that answer in detail?"
            )
            chat_history, llm_toxicity_explanation = self.prompt_llm(
                chat_history, prompt
            )

            output_data.append(
                {
                    "test": text,
                    "toxicity_classification": llm_toxicity_classification,
                    "toxicity_explanation": llm_toxicity_explanation,
                }
            )

        pd.DataFrame(output_data).to_csv(
            f"processed_data/processed_{self.data_name}.csv", index=False
        )


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
        data_name="explicit_toxicity",
    )
    data_processing.process_data_for_toxicity(input_data)
