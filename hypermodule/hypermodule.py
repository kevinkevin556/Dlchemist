import torch
from tqdm import tqdm
import numpy as np
import sklearn.metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class HyperModule():
    def __init__(self, model, criterion, optimizer, scheduler=None, load_path=None):
            self.model, self.criterion, self.optimizer = model, criterion, optimizer
            self.scheduler = scheduler
            self.epoch_trained = 0
            self.train_loss = []
            self.valid_acc = []
            self.test_acc = None


    # ----------------------- train() --------------------------------------- #

    def train(self, train_dataloader, valid_dataloader=None, save_path=None, num_epochs=1):
        device = torch.device('cuda')
        self.model.to(device)
        best_acc = 0 if len(self.valid_acc)==0 else max(self.valid_acc) 
        start_epoch = self.epoch_trained

        for epoch in range(num_epochs):
            self.batch_loss, self.batch_acc = [], []
            self.epoch_trained += 1
            
            ## Training stage
            self.model.train()
            train_progress = tqdm(train_dataloader, position=0, leave=True)
            for images, targets in train_progress:
                images, targets = images.to(device), targets.to(device)
                self.update_(images, targets)
                self.update_progress_(train_progress, start_epoch+num_epochs)
            
            self.update_scheduler_()                    # Update scheduler based on training loss
            self.perform_validation_(valid_dataloader)  # Validation stage (or choose not to validate)
            self.update_history_()                      # Update training loss and validation acc.
            if np.mean(self.batch_acc) > best_acc:      # Save the best model (if any)
                best_acc = np.mean(self.batch_acc)
                self.save(save_path)

        # Save the model trained in the last epoch (if validation)
        if valid_dataloader is not None:
            self.save(save_path+".final")
        # Clear training infomation            
        self.batch_loss, self.batch_acc = [], []
        self.test_acc = None
        

    def update_(self, images, targets):
        preds = self.model(images)
        loss = self.criterion(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.batch_loss.append(loss.detach().item())

    def update_progress_(self, progress, num_epochs):
        loss = self.batch_loss[-1]
        epoch = self.epoch_trained
        progress.set_description(f'Epoch [{epoch}/{num_epochs}]')
        progress.set_postfix({'loss': loss})

    def update_scheduler_(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def update_history_(self):
        self.train_loss.append(np.mean(self.batch_loss))
        self.valid_acc.append(np.mean(self.batch_acc))
    
    def perform_validation_(self, valid_dataloader):
        batch_avg_loss = np.mean(self.batch_loss)
        if valid_dataloader is None:
            self.batch_acc = [np.nan]
            print(f"Train Loss: {batch_avg_loss:.3f}, Valid Acc: --- No Validation ---")
        else:
            self.batch_acc = self.validate(*valid_dataloader)
            batch_avg_acc = np.mean(self.batch_acc)
            print(f"Train Loss: {batch_avg_loss:.3f}, Valid Acc:{batch_avg_acc:.3f}")


    # ----------------------- validate() ------------------------------------- #

    def validate(self, dataloader):
        device = torch.device('cuda')
        self.model.eval()
        batch_acc = []
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                pred_labels = self.get_prediction_(images, targets, validation=True)
                batch_acc.append((pred_labels == targets).type(torch.float32).mean().item())
        return batch_acc
    
    def get_prediction_(self, images, targets, validation=True):
        preds = self.model(images)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(preds)
        pred_labels = torch.argmax(probs, dim=1)
        if validation:
            return pred_labels
        else:
            pred_labels = pred_labels.view(-1).detach().cpu().numpy()
            targets = targets.view(-1).detach().cpu().numpy()
            return pred_labels, targets


    # ------------------------ test() ----------------------------------------- #

    def test(self, dataloader, dataloader_f, load_path=None, confusion_matrix=True, class_names=None):
        device = torch.device('cuda')
        self.model.to(device)
        if load_path is not None:
            self.load(load_path)
        self.model.eval()

        # Obtain predictions and ground truths
        np_pred_labels, np_targets = [], [] 
        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                pred_labels, targets = self.get_prediction_(images, targets, validation=False)
                np_pred_labels.append(pred_labels)
                np_targets.append(targets)

        np_pred_labels = np.concatenate(np_pred_labels)
        np_targets = np.concatenate(np_targets)
        if confusion_matrix:
            self.visualize_class_acc_(np_pred_labels, np_targets, class_names)

        ## print total accuracy
        self.test_acc = np.mean(np_targets == np_pred_labels)
        print("\nTotal Acc:", np.mean(np_targets == np_pred_labels))


    def visualize_class_acc_(self, np_pred_labels, np_targets, class_names):
        conf_mat = sklearn.metrics.confusion_matrix(np_targets, np_pred_labels)
        conf_df = self.generate_confusion_df_(conf_mat, class_names)
        self.plot_confusion_heatmap_(conf_df)
        self.print_class_acc_(conf_df)

    def generate_confusion_df_(self, conf_mat, class_names):
        if class_names is not None:
            conf_df = pd.DataFrame(conf_mat, class_names, class_names)
        else:
            conf_df = pd.DataFrame(conf_mat)
        return conf_df
        
    def plot_confusion_heatmap_(self, conf_df):
        plt.figure(figsize = (12,8))
        sns.heatmap(conf_df, annot=True, fmt="d", cmap='Blues')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.show()

    def print_class_acc_(self, conf_df):
        for i in range(len(conf_df)):
            total = np.sum(conf_df.iloc[i, :])
            correct = conf_df.iloc[i, i]
            print(f"Acc of {conf_df.columns[i]}: {correct/total:.4f}")



    # ----------------------- load() ----------------------------------------- #

    def load(self, path):
        device = torch.device('cuda')
        state_dict = torch.load(path)
        
        self.model.load_state_dict(state_dict["model"])
        self.model.to(device)
        
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.test_acc = state_dict["test_acc"]

        n_train_loss = len(state_dict["train_loss"])
        n_valid_acc = len(state_dict["valid_acc"])
        epoch_trained = state_dict["epoch_trained"]
        n = min(epoch_trained, n_train_loss, n_valid_acc)

        self.epoch_trained = n
        self.train_loss = state_dict["train_loss"][:n]
        self.valid_acc = state_dict["valid_acc"][:n]
        print("State dict sucessfully loaded.")


    # ----------------------- save() ----------------------------------------- #

    def save(self, path):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        state_dict["scheduler"] = self.scheduler.state_dict()
        state_dict["epoch_trained"] = self.epoch_trained
        state_dict["train_loss"] = self.train_loss
        state_dict["valid_acc"] = self.valid_acc
        state_dict["test_acc"] = self.test_acc
        torch.save(state_dict, path)
        print("State dict saved.")
