from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, DataCollatorForTokenClassification, Trainer, TrainerCallback
import torch 
import pandas as pd
from datasets import Dataset
import time
import random
import numpy as np
from src.data_preprocessing import preprocess_data
# from src.model.loss import weighted_loss_function
from src.utilities import compute_metrics, spaces_decoder
from src.model.training import CustomTrainer
import time
import torch.nn as nn
import os
# add seeds 
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)




class WordSegmentation:
    def __init__(self, data_path=None, model='byt5', device="cuda:0", 
                 lr=1e-4, num_epochs=5, batch_size=4, parallel_devices=False, device_ids=[0]):
        """
        Constructs all the necessary attributes for the WordSegmentation object.

        Parameters:
        -----------
        data_path : str, optional
            The path to the CSV file containing the segmented dataset.
        model : str, optional
            The model to use for word segmentation. Default is 'byt5'.
        df : pd.DataFrame, optional
            DataFrame containing the segmented dataset.
        chunk_size : int, optional
            Number of rows to read per chunk to avoid memory issues. Default is 10,000.
        max_length : int, optional
            Maximum length for padding and truncation of sequences. Default is 512.
        parallel_devcies : bool, optional
            Whether to use multiple GPUs for training. Default is False.
        device_ids : list, optional
            List of device IDs to use for training. Only for parallel. Default is [0].
        """
        self.data_path = data_path
        if model == 'byt5':
            self.tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
            self.model = AutoModelForTokenClassification.from_pretrained("google/byt5-small", num_labels=2, device_map='auto')
        elif model == 'canine':
            self.tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
            self.model = AutoModelForTokenClassification.from_pretrained("google/canine-s", num_labels=2)
        for param in self.model.parameters(): param.data = param.data.contiguous() # make the model parameters contiguous - avoid errors
        
        self.parallel_devices = parallel_devices


        self.train_dataset, self.eval_dataset = None, None

        self.model_name = model # model name
        self.lr = lr # learning rate for training
        self.num_epochs = num_epochs # number of epochs for training 
        self.batch_size = batch_size # batch size for training
        # Move the model to the appropriate device (GPU if available)
        if not self.parallel_devices:
        # Check if a GPU is available and set the device accordingly
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            print(f'device:{self.device}')
            self.model.to(self.device)

        elif self.parallel_devices and torch.cuda.device_count() > 1:
        # Wrap the model with DataParallel to utilize multiple GPUs
            # if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            #Move the model to CUDA
            self.model = self.model.cuda()
        self.trainer = None

    
    def set_datasets(self, train_dataset, eval_dataset, full_dataset=None):
        """
        Sets the training and evaluation datasets for the model.

        Parameters:
        -----------
        train_dataset : Dataset
            The training dataset.
        eval_dataset : Dataset
            The evaluation dataset.
        full_dataset : Dataset, optional
            The full dataset includes the texts columns: 'input_text' and 'target_text'. Default is None.
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.full_dataset = full_dataset



    def train_model(self):
        """
        Trains the ByT5 model on the preprocessed dataset.

        This method configures the training arguments and initializes a Trainer to train the model on the preprocessed dataset.

        The Trainer handles the training loop, including:
        - Loading the data
        - Forward passes
        - Backward passes (gradient computation)
        - Optimizer steps (weight updates)

        Training hyperparameters such as learning rate, batch size, number of epochs, etc., are specified through TrainingArguments.

        Returns:
        --------
        None
        """

        start_time = time.time()

        training_args = TrainingArguments(
            output_dir="./results",  # Directory to save the results
            eval_strategy="epoch",  # Evaluate the model at the end of each epoch
            learning_rate=self.lr,  # learning rate
            seed = 42,
            per_device_train_batch_size=self.batch_size,  # Batch size for training
            per_device_eval_batch_size=self.batch_size,  # Batch size for evaluation
            # weight_decay=0.01,  # Weight decay for regularization
            save_total_limit=1,  # Limit the total number of checkpoints saved during training (default is 5) - delete older checkpoints to save space 
            num_train_epochs=self.num_epochs,  # Number of training epochs
            logging_dir='./src/model/logs',  # Directory to save the logs
            # report_to="none",  # Disable logging to external services like TensorBoard
            # fp16=True,  # Enable mixed precision training
            gradient_accumulation_steps=8,  # Accumulate gradients for 8 steps before updating
            # eval_accumulation_steps=10,  # Accumulate evaluation steps
            logging_steps=5,  # Log every 10 steps
            save_strategy="epoch",  # Save the model at the end of each epoch
            # load_best_model_at_end=True,  # Load the best model at the end of training
            # run_name="byt5-word-segmentation_" + str(start_time),  # Name of the run,
            run_name=self.model_name+"-word-segmentation_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
            torch_empty_cache_steps=10,  # Empty the cache every 10 steps to avoid memory issues
            remove_unused_columns=False  # Prevent removal of columns
            # metric_for_best_model="eval_loss"  # Use evaluation loss to determine the best model
            # max_steps = 1e8, # The maximum number of training steps to perform - set to a large number to avoid early stopping. Must be assigned because of streaming (IterableDataset)
            # metric_for_best_model="eval_f1"  # Use evaluation F1 score to determine the best model
            # metric_for_best_model="eval_accuracy"  # Use evaluation F1 score to determine the best model
            # max_grad_norm=1.0  # Add gradient clipping to prevent exploding gradients
        )

        # DataCollatorForTokenClassification is used to dynamically pad inputs and labels to the length of the longest sequence in a batch
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True, return_tensors="pt")


        self.trainer = Trainer(
            model=self.model, # The instantiated 🤗 Transformers model to be trained
            args=training_args, # TrainingArguments
            train_dataset=self.train_dataset, # Training dataset
            eval_dataset=self.eval_dataset, # Evaluation dataset
            tokenizer=self.tokenizer, # Tokenizer for the model
            data_collator=data_collator, # Data collator
            compute_metrics=compute_metrics # The function that computes metrics of interest,
            # callbacks=[GpuMemoryCallback()]
            # compute_metrics=None
            # preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )

        torch.cuda.empty_cache() # Empty the cache to avoid memory issues


        self.trainer.train() # Train the model

        end_time = time.time() # End time
        print(f'finish train after {end_time-start_time}') # Print the time taken for training




    def evaluate_model(self, n_examples=10):
        """
        Evaluates the trained model on the validation set and prints random segmented examples.

        Parameters:
        -----------

        n_examples : int, optional
            Number of random examples to print. (default is 5)

        Returns:
        --------
        dict
            The evaluation results.
        """

        start_time = time.time()

        results = self.trainer.evaluate()

        # Print random segmented examples
        if n_examples > len(self.eval_dataset):
            n_examples = len(self.eval_dataset)
        indices = random.sample(range(len(self.eval_dataset)), n_examples)
        # number of samples in train_dataset:

        for idx in indices:
            # make sure that the current data set has the input_ids, labels, input_text, target_text. other wise raise an error for each oen seperatly 
            if 'input_ids' not in self.eval_dataset[idx]:
                raise ValueError("The input_ids are not in the current eval dataset")

            if 'labels' not in self.eval_dataset[idx]:
                raise ValueError("The labels are not in the current eval dataset")
            
                

            input_ids = self.eval_dataset[idx]['input_ids']

            # Decode the input_ids to get the original input text
            input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            # input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # input_text = self.eval_dataset[idx]['input_text']
            target_text = spaces_decoder(tokenized_inputs=self.eval_dataset[idx], tokenizer=self.tokenizer)

            # Reconstruct the target text from labels

            # Tokenize the input text at the character level and move to the appropriate device
            inputs = self.tokenizer(input_text, return_tensors="pt")

            if not self.parallel_devices:
                inputs = inputs.to(self.device)

            else:
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

            outputs = self.model(inputs['input_ids'])
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

            # dictionary with the predictions as labels and the input_ids as input_ids
            tokenized_predictions = {'input_ids': input_ids, 'labels': predictions}


            # Reconstruct the text with spaces
            predicted_text = spaces_decoder(tokenized_inputs=tokenized_predictions, tokenizer=self.tokenizer)

            print(f"\nOriginal: {input_text}")
            print(f"\nGround Truth: {target_text}")
            print(f"\nPredicted: {predicted_text}\n")

            # add the input_text, target_text, predicted_text to the results
            results[f'example_{idx}'] = {'input_text': input_text, 'target_text': target_text, 'predicted_text': predicted_text}


        end_time = time.time()
        print(f'finish evaluate after {end_time-start_time}')
        return results

    
    def save_model(self, path="./models/word_segmentation"):
        """
        Saves the trained model to the specified path.

        Parameters:
        -----------
        model : transformers.AutoModelForTokenClassification
            The trained model.
        path : str, optional
            The path to save the model. (default is "./models/word_segmentation")

        Returns:
        --------
        None
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        print(f"Model saved to {path}")


# class GpuMemoryCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         num_gpus = torch.cuda.device_count()
#         for i in range(num_gpus):
#             allocated_memory = torch.cuda.memory_allocated(i)
#             reserved_memory = torch.cuda.memory_reserved(i)
#             print(f"GPU {i} - Step {state.global_step}: Allocated Memory: {allocated_memory / 1e6:.2f} MB, Reserved Memory: {reserved_memory / 1e6:.2f} MB")

    def load_model(self, path="./models/word_segmentation", tokenizer_name="google/byt5-small"):
        """
        Loads the model and tokenizer from the specified path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForTokenClassification.from_pretrained(path)

        if not self.parallel_devices:
            self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        elif self.parallel_devices and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            self.model = self.model.cuda()

        print(f"Model loaded from {path}")



    def predict(self, text):
        """
        Predicts the segmentation of the input text.

        Parameters:
        -----------
        text : str
            The input text to segment.

        Returns:
        --------
        str
            The segmented text.
        """
        inputs = self.tokenizer(text, return_tensors="pt") # Tokenize the input text



        # Move the inputs to the appropriate device
        if not self.parallel_devices:
            inputs = inputs.to(self.device)
        else:
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
        
        # Perform a forward pass to get the model outputs
        outputs = self.model(inputs['input_ids'])

        # Get the predicted labels
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Create a dictionary with the predictions as labels and the input_ids as input_ids
        tokenized_predictions = {'input_ids': inputs['input_ids'], 'labels': predictions}

        # Reconstruct the text with spaces
        predicted_text = spaces_decoder(tokenized_inputs=tokenized_predictions, tokenizer=self.tokenizer)

        return predicted_text
    
