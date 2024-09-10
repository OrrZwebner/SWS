import pandas as pd
import numpy as np
import torch
import re
import evaluate
import torch.nn.utils.prune as prune
import os


# def tokenize_and_label_function(text, tokenizer, model="byt5", max_length=512):
def tokenize_and_label_function(text, tokenizer, input_col='input_text', target_col='target_text'):
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

    # raise a value error if the input text is not a pandas series, and if the entries in the input text and target columns are not strings
    if not isinstance(text, pd.Series):
        raise ValueError("The input text must be a pandas series.")
    
    if not isinstance(text[input_col], str):
        raise ValueError("The input text must be a string.")
    
    if not isinstance(text[target_col], str):
        raise ValueError("The target text must be a string.")

    # raise a value error if input text consist spaces. note that it is a string, not a list
    if ' ' in text[input_col]:
        raise ValueError("The input text should not contain spaces.")
    
    # raise a value error if target text consist duplicated spaces
    # if '  ' in text[target_col]:
    #     # print where the duplicated spaces are in the string
    #     print(f"target text: {text[target_col]}")
    #     raise ValueError("The target text should not contain duplicated spaces.")

    # Tokenize the inputs with padding and truncation
    tokenized_inputs = tokenizer(
        text[input_col]  # The input text 
        # padding=True,  # Let DataCollator handle padding
        # padding='max_length',  # Pad to the maximum length
        # truncation=True,  # Truncate sequences that are longer than the maximum length
        # max_length=max_length  # The maximum length to pad or truncate sequences to
        # return_tensors="pt"  # Return the tokenized sequences as PyTorch tensors
    )

    tokenized_targets = tokenizer(
        text[target_col]  # The target text
        # padding=True,  # Let DataCollator handle padding  
        # padding='max_length',
        # truncation=True,  # Truncate sequences that are longer than the maximum length
        # max_length=max_length  # The maximum length to pad or truncate sequences to  
        # return_tensors="pt"  # Return the tokenized sequences as PyTorch tensors
    )

    # Create labels for the targets

    if tokenizer.name_or_path == 'google/canine-s':
        space_token = tokenizer(' ')['input_ids'][1] # Get the token ID for the space token byt5
    else:
        space_token = tokenizer(' ')['input_ids'][0]



    # create a lsit of zeros with the length of the input text
    labels = [0] * len(tokenized_inputs['input_ids'])
    spaces = 0 # the number of spaces in the input text

    # iterate over every token beside the last one in the input text

    for i, input_token in enumerate (tokenized_inputs['input_ids'][:-1]):
        # iterate over every token besides the last one in the input and target text. if the target token is a space, add a 1 to the labels list in the position of the token before the space
        # if the target token is not a space, akeep it as 0 to the labels list in the position of the token
        # print(f"tokenized_targets['input_ids']: {tokenized_targets['input_ids']}\nlen(tokenized_targets['input_ids']): {len(tokenized_targets['input_ids'])}\nspaces: {spaces}\n i: {i}\n tokenized_inputs['input_ids']: {tokenized_inputs['input_ids']}\nlen(tokenized_inputs['input_ids']): {len(tokenized_inputs['input_ids'])}")
        if tokenized_targets['input_ids'][i+spaces+1] == space_token and tokenized_inputs['input_ids'][i] != tokenizer.pad_token_id:
            labels[i] = 1 # add a 1 to the labels list in the position of the token before the space
            spaces += 1 # increment the number of spaces

        # if the target token is a space and the input token is a padding token, add a -100 to the labels list in the position of the token
        elif tokenized_inputs['input_ids'][i] == tokenizer.pad_token_id:
            labels[i] = PAD_TOKEN_LABEL

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
    accuracy_metric = evaluate.load("accuracy", trust_remote_code=True)
    precision_metric = evaluate.load("precision", trust_remote_code=True)
    recall_metric = evaluate.load("recall", trust_remote_code=True)
    f1_metric = evaluate.load("f1", trust_remote_code=True)

    # Comptue metrics for models without training
    if isinstance(pred, dict):
        predictions = pred['predictions']
        labels = pred['labels']

    # Compute metrics for models with training
    else:
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

    # if there are no keys of 'input_ids'  and 'labels' in the tokenized_inputs, raise a ValueError
    if 'input_ids' not in tokenized_inputs.keys():
        raise ValueError('tokenized_inputs must contain a key of "input_ids"')
    if 'labels' not in tokenized_inputs.keys():
        raise ValueError('tokenized_inputs must contain a key of "labels"')
    
    # tokenized_inputs['input_ids'] is a tensor. if it has nestes lists, tensor([[value1, value2, ...]]), flatten the outer list and raise an error for unflattened tensors
    if isinstance(tokenized_inputs['input_ids'], torch.Tensor):
        tokenized_inputs['input_ids'] = tokenized_inputs['input_ids'].tolist()[0]
        # print(f' Unnesacary nested loop in the input ids tensor')



    if tokenizer.name_or_path == 'google/canine-s':
        space_token = tokenizer(' ')['input_ids'][1] # Get the token ID for the space token byt5
    else:
        space_token = tokenizer(' ')['input_ids'][0]

    # create a new tokened text - add a space token after every token that is labeled as 1 in byt5_tokenized_inputs['input_ids']. do not decode, creae a list of token ids with added spaces
    new_input_ids = []
    for i, token in enumerate(tokenized_inputs['input_ids']):
        new_input_ids.append(token)
        # if the token is a space token, add a space token to the new_input_ids. 
        # verify that the token is not a space or part of an slp1 character
        if tokenized_inputs['labels'][i] == 1 and tokenizer.decode(token) != ' ' and  tokenizer.decode(token) != '': 
            new_input_ids.append(space_token)

    # Decode the input_ids to get the original input text
    decoded_text = tokenizer.decode(new_input_ids, skip_special_tokens=True)

    decoded_text

    return decoded_text


def remove_extra_spaces(text):
    """
    Removes extra spaces from the input string, replacing multiple spaces with a single space.

    Parameters:
    -----------
    text : str
        The input string that may contain multiple spaces.

    Returns:
    --------
    str
        The input string with extra spaces removed.
    """
    cleaned_text = ' '.join(text.split())
    if cleaned_text != text:  # If any change happened
        print("Extra spaces were found and removed.")
    return cleaned_text



def predict_labels(df, model, dataset_name, input_col, target_col):
    """
    Processes the DataFrame by tokenizing and labeling the inputs and targets, and predicting labels using the model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the input texts and target texts.
    model : transformers.PreTrainedModel
        The pre-trained model used to predict labels.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to tokenize the input and target texts.
    dataset_name : str
        The name of the dataset to be used in the saved CSV file.
    input_col : str, optional
        The column name for the input texts. Default is 'unsandhied_input_text'.
    target_col : str, optional
        The column name for the target texts. Default is 'unsandhied'.

    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with target and predicted labels.
    """
    # Ensure there are no spaces in the input_col values
    if df[input_col].apply(lambda x: " " in x).any():
        raise ValueError(f"The {input_col} column contains spaces. Please remove spaces from the input text.")
    
    # Tokenize and label the target column directly using your function
    tokenized_and_labelled = df.apply(
        tokenize_and_label_function, 
        axis=1, 
        tokenizer=model.tokenizer, 
        input_col=input_col, 
        target_col=target_col
    )
    
    # Add the labels from tokenized data into the dataframe as target_labels
    df["target_labels"] = tokenized_and_labelled.apply(lambda x: x['labels'])
    
    # Predict the labels using the model (assuming the predict function is provided)
    df["predictions"] = df[input_col].apply(lambda  x: model.predict(x))


    tokenized_and_predicted = df.apply(
        tokenize_and_label_function, 
        axis=1, 
        tokenizer=model.tokenizer, 
        input_col=input_col, 
        target_col="predictions"
    )


    df["predicted_labels"] = tokenized_and_predicted.apply(lambda x: x['labels'])
    
    # Create output directory if it doesn't exist
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the DataFrame with labels to CSV file, using dataset_name in the file name
    csv_file_name = f"{output_dir}/{dataset_name}_with_predictions_labels.csv"
    df.to_csv(csv_file_name, index=False)
    print(f"DataFrame saved successfully to {csv_file_name}")
    
    return df