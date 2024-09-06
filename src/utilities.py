import pandas as pd
import numpy as np
import torch
import re
import evaluate
import torch.nn.utils.prune as prune


# def tokenize_and_label_function(text, tokenizer, model="byt5", max_length=512):
def tokenize_and_label_function(text, tokenizer):
    """
    Tokenizes the input and target texts at the SLP1 token level.

    Parameters:
    -----------
    text: pd.Series of input_text and target_text
        A dictionary containing the 'input_text' and 'target_text' fields.
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizer
        The tokenizer to use for tokenization.
    model: str
        The model to use for tokenization.

    Returns:
    --------
    a dictionary with tokenized inputs and labels.
    """

    PAD_TOKEN_LABEL = -100

    # Tokenize the inputs with padding and truncation
    tokenized_inputs = tokenizer(
        text['input_text']  # The input text 
        # padding=True,  # Let DataCollator handle padding
        # padding='max_length',  # Pad to the maximum length
        # truncation=True,  # Truncate sequences that are longer than the maximum length
        # max_length=max_length  # The maximum length to pad or truncate sequences to
        # return_tensors="pt"  # Return the tokenized sequences as PyTorch tensors
    )

    tokenized_targets = tokenizer(
        text['target_text']  # The target text
        # padding=True,  # Let DataCollator handle padding  
        # padding='max_length',
        # truncation=True,  # Truncate sequences that are longer than the maximum length
        # max_length=max_length  # The maximum length to pad or truncate sequences to  
        # return_tensors="pt"  # Return the tokenized sequences as PyTorch tensors
    )

    # Create labels for the targets
    j = 0  # Pointer for target text
    # space_token = tokenizer(' ', padding=True)['input_ids'][0] # Get the token ID for the space token byt5
    # if model == "byt5":
    # if tokenizer
    #     space_token = tokenizer(' ')['input_ids'][0]
    if tokenizer.name_or_path == 'google/canine-s':
        space_token = tokenizer(' ')['input_ids'][1] # Get the token ID for the space token byt5
    else:
        space_token = tokenizer(' ')['input_ids'][0]

    # space_token = 32 # Get the token ID for the space token b
    # print(f'space_token: {space_token}')

    # create a lsit of zeros with the length of the input text
    labels = [0] * len(tokenized_inputs['input_ids'])
    spaces = 0 # the number of spaces in the input text

    # iterate over every token beside the last one in the input text
    # print(f'tokenized_inputs["input_ids"]: {tokenized_inputs["input_ids"]}')
    # print(f'tokenized_targets["input_ids"]: {tokenized_targets["input_ids"]}')
    for i, input_token in enumerate (tokenized_inputs['input_ids'][:-1]):
        # iterate over every token besides the last one in the input and target text. if the target token is a space, add a 1 to the labels list in the position of the token before the space
        # if the target token is not a space, akeep it as 0 to the labels list in the position of the token
        # print(f'j+spaces: {j+spaces}')
        # print(f'byt5_tokenized_targets["input_ids"][i+spaces]: {byt5_tokenized_targets["input_ids"][i+spaces]}')
        # print(f'input_token: {input_token}')
        if tokenized_targets['input_ids'][i+spaces+1] == space_token and tokenized_inputs['input_ids'][i] != tokenizer.pad_token_id:
            # if i != 0 and labels[i-1] != 1: # if the previous token in the target  is not a space
            labels[i] = 1 # add a 1 to the labels list in the position of the token before the space
            spaces += 1 # increment the number of spaces
            # print(f'token {input_token} labeled as 1')

        # if the target token is a space and the input token is a padding token, add a -100 to the labels list in the position of the token
        elif tokenized_inputs['input_ids'][i] == tokenizer.pad_token_id:
            labels[i] = PAD_TOKEN_LABEL

    # labels = []  # Initialize the list of labels

    # for input_token in tokenized_inputs['input_ids']:
    #     # iterate over every token in the input and target text. if the target token is a space, add a 1 to the labels list in the position of the token before the space
    #     # if the target token is not a space, add a 0 to the labels list in the position of the token
    #     if j < len(tokenized_targets['input_ids']) and tokenized_targets['input_ids'][j] == space_token and input_token != tokenizer.pad_token_id:
    #         labels.append(1)
    #         j += 1
    #     else:
    #         labels.append(0)
    #     j += 1  

    # # # add padding to labels
    # # input_len = len(tokenized_inputs['input_ids'])  # Get the length of the tokenized input
    # # if len(labels) < input_len:
    # #     labels += [PAD_TOKEN_LABEL] * (input_len - len(labels))

    tokenized_inputs['labels'] = labels

    return tokenized_inputs

    


def split_slp1(text):
        """
        Splits a given text into SLP1 tokens.

        Parameters:
        -----------
        text : str
            The input text in SLP1 transliteration.

        Returns:
        --------
        List[str]
            A list of SLP1 tokens.
        """
        
        # Regex pattern for SLP1 tokens, which can be one or two characters long
        pattern = r'[a-zA-Z ]|\S'

        return re.findall(pattern, text)
        # chars_list = [char[0] for char in re.findall(pattern, text)]

        # return chars_list



def compute_metrics(pred):
    """
    Computes evaluation metrics for the word segmentation model.

    This function evaluates the model's performance using accuracy, precision, recall, and F1-score.

    Parameters:
    -----------
    pred : transformers.TrainerPredictionOutput
        The prediction output from the model, which includes the predicted token IDs and the true token IDs (labels).

    Returns:
    --------
    dict
        A dictionary containing the evaluation metrics.
    """
    # Load metrics
    # print("Hi there!!!!!!")
    accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
    precision_metric = evaluate.load("precision", trust_remote_code=True)
    recall_metric = evaluate.load("recall", trust_remote_code=True)
    f1_metric = evaluate.load("f1", trust_remote_code=True)

    predictions, labels = pred.predictions, pred.label_ids
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)

    # Calculate the predicted labels
    predictions = torch.argmax(predictions, dim=-1)

    # Flatten the tensors to avoid dimension issues
    predictions = predictions.view(-1)
    labels = labels.view(-1)

    # Calculate each metric
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average=None)
    recall = recall_metric.compute(predictions=predictions, references=labels, average=None)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)

    # # # list the metrics for the trainer to read them
    # precision["precision"] = precision["precision"].tolist()
    # recall["recall"] = recall["recall"].tolist()
    # f1["f1"] = f1["f1"].tolist()


    # # Combine all metrics into a single dictionary
    # result = {
    #     "eval_accuracy": accuracy["accuracy"],
    #     "eval_precision": precision["precision"],
    #     "eval_recall": recall["recall"],
    #     "eval_f1": f1["f1"],
    # }

    # return result

        # Convert list metrics to individual class metrics
    result = {
        "eval_accuracy": accuracy["accuracy"],
    }
    
    # Assuming that the order of metrics corresponds to the class labels (0 and 1)
    for i, (p, r, f) in enumerate(zip(precision["precision"], recall["recall"], f1["f1"])):
        result[f"eval_precision_class_{i}"] = p
        result[f"eval_recall_class_{i}"] = r
        result[f"eval_f1_class_{i}"] = f

    return result



def prune_model(model, pruning_amount=0.2):
    # Iterate over all modules and prune 20% of connections in linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            # Optionally, also prune biases
            # prune.l1_unstructured(module, name='bias', amount=pruning_amount)
    return model



def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels



def spaces_decoder(tokenized_inputs, tokenizer):
    """
    Decodes the token IDs into a string and replaces the space tokens with spaces.

    Parameters:
    -----------
    tokenized_inputs : dict (or a batch of tokenized inputs)
        The tokenized inputs containing the input IDs with no spaces and labels
    tokenizer : transformers.tokenization_utils_base.PreTrainedTokenizer
        The tokenizer used to encode the text.


    Returns:
    --------
    str
        The decoded string.
    """

    # # verify that the space token is an int and a space token
    # if not isinstance(space_token, int):
    #     raise ValueError('space_token must be an integer')
    # if tokenizer.decode(space_token) != ' ':
    #     raise ValueError('space_token must be a space token')

    # if there are no keys of 'input_ids'  and 'labels' in the tokenized_inputs, raise a ValueError
    if 'input_ids' not in tokenized_inputs.keys():
        raise ValueError('tokenized_inputs must contain a key of "input_ids"')
    if 'labels' not in tokenized_inputs.keys():
        raise ValueError('tokenized_inputs must contain a key of "labels"')
    
    # tokenized_inputs['input_ids'] is a tensor. if it has nestes lists, tensor([[value1, value2, ...]]), flatten the outer list and raise an error for unflattened tensors
    if isinstance(tokenized_inputs['input_ids'], torch.Tensor):
        tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'].tolist()[0]
        print(f' Unnesacary nested loop in the input ids tensor')



    if tokenizer.name_or_path == 'google/canine-s':
        space_token = tokenizer(' ')['input_ids'][1] # Get the token ID for the space token byt5
    else:
        space_token = tokenizer(' ')['input_ids'][0]

        
    # create a new tokened text - add a space token after every token that is labeled as 1 in byt5_tokenized_inputs['input_ids']. do not decode, creae a list of token ids with added spaces
    new_input_ids = []
    for i, token in enumerate(tokenized_inputs['input_ids']):
        new_input_ids.append(token)
        if tokenized_inputs['labels'][i] == 1:
            new_input_ids.append(space_token)

    # Decode the input_ids to get the original input text
    decoded_text = tokenizer.decode(new_input_ids, skip_special_tokens=True)

    decoded_text

    return decoded_text