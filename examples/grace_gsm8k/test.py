from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from grace_utils.t5_discriminator import T5Discriminator, T5EnergyDiscriminator
from torch.nn.utils.rnn import pad_sequence

def calculate_discriminator_score(input_text: str, discriminator: PreTrainedModel, discriminator_tokenizer: PreTrainedTokenizer, device: torch.device) -> torch.Tensor:
    """
    Calculate the discriminator score for a given string.

    Args:
    - input_text (str): The input text string for which to calculate the discriminator score.
    - discriminator (PreTrainedModel): The discriminator model.
    - discriminator_tokenizer (PreTrainedTokenizer): The tokenizer for the discriminator model.
    - device (torch.device): The device on which to perform the calculation.

    Returns:
    - torch.Tensor: The discriminator score for the input text.
    """

    # Tokenize the input text
    tokenized_input = discriminator_tokenizer.encode_plus(input_text, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True, padding='max_length')
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    # Calculate the discriminator score
    with torch.no_grad():
        discriminator_score = discriminator.forward_scores(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming the discriminator returns a single score per input in the last layer
        # discriminator_score = outputs.logits.squeeze()

    return discriminator_score

# Example usage
if __name__ == "__main__":
    ckpt = torch.load("/data/adithya/ckpts/discrim/gsm8k/pytorch_model.bin")
    discriminator_tokenizer = T5Tokenizer.from_pretrained("/data/adithya/ckpts/discrim/gsm8k/")

    disc_backbone = 'google/flan-t5-large'
    discriminator_model = T5EnergyDiscriminator(model_name_or_path=disc_backbone, 
    device="cuda")
    discriminator_model.model.resize_token_embeddings(len(discriminator_tokenizer))
    discriminator_model.load_state_dict(ckpt)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator_model.to(device)

    input_string = "Your input string here."
    score = calculate_discriminator_score(input_string, discriminator_model, discriminator_tokenizer, device)
    print("Discriminator score:", score)
