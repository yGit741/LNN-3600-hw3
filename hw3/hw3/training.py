import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn

from cs236781.train_results import FitResult, BatchResult, EpochResult


def freeze_model_except_for_the_last_2_linear_layers(model: nn.Module) -> None:
    # You can change the header of this function if you wish to add more parameters.
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # model.classifier.requires_grad = True
    # model.pre_classifier.requires_grad = True

    for param in model.pre_classifier.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all_model_parameters(model: nn.Module) -> None:
    # for param in model.parameters():
    #     param.requires_grad = False

    for param in model.parameters():
        param.requires_grad = True


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu"):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        post_epoch_fn=None,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            print(epoch)
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            # TODO:
            #  Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======

            # current epoch training process
            train_results = self.train_epoch(dl_train, verbose=verbose)
            train_loss.append(torch.mean(torch.tensor(train_results.losses)))
            train_acc.append(train_results.accuracy)
            test_result = self.test_epoch(dl_test, verbose=verbose)
            test_loss.append(torch.mean(torch.tensor(test_result.losses)))
            test_acc.append(test_result.accuracy)
            actual_num_epochs += 1

            # checkpoints and breaking points
            if best_acc is None or test_result.accuracy > best_acc:
                best_acc = test_result.accuracy
                epochs_without_improvement = 0
                if checkpoints is not None:
                    save_checkpoint = True
            else:
                epochs_without_improvement += 1
            if early_stopping is not None:
                if epochs_without_improvement >= early_stopping:
                    self._print("early stopping")
                    break

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_results, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100.0 * num_correct / num_samples
            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Accuracy {accuracy:.1f})"
            )

        return EpochResult(losses=losses, accuracy=accuracy)


class RNNTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)

    def train_epoch(self, dl_train: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.hidden_state = None
        self.model.train()
        torch.enable_grad()
        # ========================
        return super().train_epoch(dl_train, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw):
        # TODO: Implement modifications to the base method, if needed.
        # ====== YOUR CODE: ======
        self.hidden_state = None
        # ========================
        return super().test_epoch(dl_test, **kw)

    def train_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        # TODO:
        #  Train the RNN model on one batch of data.
        #  - Forward pass
        #  - Calculate total loss over sequence
        #  - Backward pass: truncated back-propagation through time
        #  - Update params
        #  - Calculate number of correct char predictions
        # ====== YOUR CODE: ======
        if self.hidden_state is None:
            self.hidden_state = self.model.init_hidden_state(x.size(0), self.device)

        self.optimizer.zero_grad()
        pred_y, hidden_state = self.model(x, self.hidden_state)

        self.hidden_state = hidden_state.detach().to(self.device)

        loss = self.loss_fn(pred_y.view(-1, pred_y.shape[2]), y.view(-1))
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            classified_y = torch.argmax(pred_y, dim=2)
            num_correct = torch.sum(y == classified_y)
        # ========================

        # Note: scaling num_correct by seq_len because each sample has seq_len
        # different predictions.
        return BatchResult(loss.item(), num_correct.item() / seq_len)

    def test_batch(self, batch) -> BatchResult:
        x, y = batch
        x = x.to(self.device, dtype=torch.float)  # (B,S,V)
        y = y.to(self.device, dtype=torch.long)  # (B,S)
        seq_len = y.shape[1]

        if self.hidden_state is None:
            self.hidden_state = self.model.init_hidden_state(x.size(0), self.device)

        with torch.no_grad():
            # TODO:
            #  Evaluate the RNN model on one batch of data.
            #  - Forward pass
            #  - Loss calculation
            #  - Calculate number of correct predictions
            # ====== YOUR CODE: ======
            pred_y, self.hidden_state  = self.model(x, self.hidden_state)

            self.hidden_state = self.hidden_state.detach().to(self.device)

            loss = self.loss_fn(pred_y.view(-1,pred_y.shape[2]),y.view(-1))
            classified_y = torch.argmax(pred_y,dim=2)
            num_correct = torch.sum(y == classified_y)
            # raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), num_correct.item() / seq_len)


class VAETrainer(Trainer):
    def train_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)
        # TODO: Train a VAE on one batch.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())

    def test_batch(self, batch) -> BatchResult:
        x, _ = batch
        x = x.to(self.device)  # Image batch (N,C,H,W)

        with torch.no_grad():
            # TODO: Evaluate a VAE on one batch.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================

        return BatchResult(loss.item(), 1 / data_loss.item())


class TransformerEncoderTrainer(Trainer):

    def train_batch(self, batch) -> BatchResult:

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].float().to(self.device)
        label = batch['label'].float().to(self.device)

        loss = None
        num_correct = None
        # TODO:
        #  fill out the training loop.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================



        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].float().to(self.device)
            label = batch['label'].float().to(self.device)

            loss = None
            num_correct = None

            # TODO:
            #  fill out the testing loop.
            # ====== YOUR CODE: ======
            raise NotImplementedError()
            # ========================



        return BatchResult(loss.item(), num_correct.item())



class FineTuningTrainer(Trainer):

    def train_batch(self, batch) -> BatchResult:
        input_ids = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        # TODO:
        #  fill out the training loop.
        # ====== YOUR CODE: ======

        self.optimizer.zero_grad()
        pred_labels = self.model(input_ids, attention_masks)
        loss = torch.nn.CrossEntropyLoss()(pred_labels.logits, labels)
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update weights
        with torch.no_grad():
            classified_labels = torch.argmax(pred_labels.logits, dim=1)
            num_correct = torch.sum(labels == classified_labels)


        # ========================

        return BatchResult(loss.item(), num_correct.item())

    def test_batch(self, batch) -> BatchResult:
        input_ids = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        with torch.no_grad():
            # TODO:
            #  fill out the training loop.
            # ====== YOUR CODE: ======
            predictions = self.model(input_ids, attention_masks)
            loss = torch.nn.CrossEntropyLoss()(predictions.logits, labels)
            predicted_labels = torch.argmax(predictions.logits, dim=1)
            num_correct = torch.sum(labels == predicted_labels)
            # raise NotImplementedError()
            # ========================
        return BatchResult(loss.item(), num_correct.item())
