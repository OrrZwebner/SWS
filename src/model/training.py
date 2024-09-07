import time
import torch
from transformers import TrainingArguments, DataCollatorForTokenClassification, Trainer
from src.model.loss import weighted_loss_function
from src.utilities import compute_metrics
import torch.nn.functional as F


PAD_TOKEN_LABEL = -100  # Special padding token for labels

class CustomTrainer(Trainer):
    """
    Custom Trainer class to override the compute_loss method for weighted loss calculation.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss using the weighted loss function.

        Parameters:
        -----------
        model : nn.Module
            The model being trained.
        inputs : dict
            The inputs and labels for the model.
        return_outputs : bool, optional
            Whether to return the model outputs along with the loss. Default is False.

        Returns:
        --------
        torch.Tensor or Tuple[torch.Tensor, Any]
            The computed loss, and optionally the model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Flatten the logits and labels if dealing with sequences
        logits = logits.view(-1, logits.size(-1))  # [batch_size * sequence_length, num_classes]
        labels = labels.view(-1)  # [batch_size * sequence_length]
        # Calculate the weighted loss
        loss = weighted_loss_function(logits, labels, weight=self.class_weights)
        print (f'loss:{loss}')
        return (loss, outputs) if return_outputs else loss





def train_model(model, tokenizer, train_dataset, eval_dataset, device, label_weight=None, lr=1e-4, num_epochs=5, batch_size=4):
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
    # Calculate class weights based on the training dataset
    # class_weights = calculate_class_weights(self.train_dataset).to(self.device)
    # Manually set class weights

    start_time = time.time()


    training_args = TrainingArguments(
        output_dir="./results",  # Directory to save the results
        eval_strategy="epoch",  # Evaluate the model at the end of each epoch
        learning_rate=lr,  # learning rate
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        # weight_decay=0.01,  # Weight decay for regularization
        save_total_limit=3,  # Limit the total number of checkpoints saved during training (default is 5) - delete older checkpoints to save space 
        num_train_epochs=num_epochs,  # Number of training epochs
        logging_dir='./src/model/logs',  # Directory to save the logs
        # report_to="none",  # Disable logging to external services like TensorBoard
        # fp16=True,  # Enable mixed precision training
        # gradient_accumulation_steps=8,  # Accumulate gradients for 8 steps before updating
        # eval_accumulation_steps=10,  # Accumulate evaluation steps
        logging_steps=10,  # Log every 10 steps
        save_strategy="epoch",  # Save the model at the end of each epoch
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss"  # Use evaluation loss to determine the best model
        # max_grad_norm=1.0  # Add gradient clipping to prevent exploding gradients
    )

    # DataCollatorForTokenClassification is used to dynamically pad inputs and labels to the length of the longest sequence in a batch
    data_collator = DataCollatorForTokenClassification(tokenizer)

    if label_weight is None:
        class_weights = torch.tensor([1.0, label_weight], dtype=torch.float).to(device)

        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    
    # else use the huggingface trainer with training args and data collator
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

    trainer.train()

    end_time = time.time()
    print(f'finish train after {end_time-start_time}')

