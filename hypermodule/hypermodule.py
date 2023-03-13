import torch
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
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

    def train(self, train_dataloader, valid_dataloader=None, save_path=None, num_epochs=1, drop_final=False):
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
            self.save(save_path, verbose=False)         # Save the trained model
            
            if np.mean(self.batch_acc) > best_acc:      # Save the best model (if any)
                best_acc = np.mean(self.batch_acc)
                self.save(save_path + ".best")
                
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
            self.batch_acc = self.validate(valid_dataloader)
            batch_avg_acc = np.mean(self.batch_acc)
            print(f"Train Loss: {batch_avg_loss:.3f}, Valid Acc:{batch_avg_acc:.3f}")

    # ------------------------ predict() ------------------------------------- #

    def predict(self, images=None, dataloader=None, numpy=False):
        device = torch.device('cuda')
        self.model.to(device)
        self.model.eval()
        if images is not None:
            images = images.to(device)
            return self.predict_image_(images, numpy=numpy)
        if dataloader is not None:
            return self.predict_dataloader_(dataloader, numpy=numpy)

    def flatten2numpy_(self, tensor):
        return tensor.view(-1).detach().cpu().numpy()

    def predict_image_(self, images, numpy=False):
        preds = self.model(images)
        probs = softmax(preds, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        if numpy:
            return self.flatten2numpy_(pred_labels)
        else:
            return pred_labels

    def predict_dataloader_(self, dataloader, return_target=False, numpy=False):
        device = torch.device('cuda')
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            pred_list = []
            target_list = []
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                pred_labels = self.predict_image_(images, numpy=numpy)
                pred_list.append(pred_labels)
                targets = self.flatten2numpy_(targets) if numpy else targets
                target_list.append(targets)
                
        output = (pred_list, target_list) if return_target else pred_list
        return output
                    

    # ----------------------- validate() ------------------------------------- #

    def validate(self, dataloader):
        device = torch.device('cuda')
        self.model.to(device)
        self.model.eval()
        pred_list, target_list = self.predict_dataloader_(dataloader, return_target=True)
        batch_acc = []
        for pred_labels, targets in zip(pred_list, target_list):
            batch_acc.append((pred_labels == targets).type(torch.float32).mean().item())
        return batch_acc


    # ------------------------ test() ----------------------------------------- #

    def test(self, dataloader, load_path=None, viz_conf=True, class_names=None, mode='image-classification'):
        device = torch.device('cuda')
        self.model.to(device)
        if load_path is not None:
            self.load(load_path)
        self.model.eval()

        # Obtain predictions and ground truths
        pred_list, target_list = self.predict_dataloader_(dataloader, return_target=True, numpy=True)
        pred_labels = np.concatenate(pred_list).flatten()
        targets = np.concatenate(target_list).flatten()
        conf_mat = confusion_matrix(targets, pred_labels)
        conf_df = self.generate_confusion_df_(conf_mat, class_names)

        if mode == 'image-classification':
            total_acc = np.mean(targets == pred_labels)
            if viz_conf:                                    # plot confusion heatmap
                self.plot_confusion_heatmap_(conf_df)    
            print("\nTotal Acc:", total_acc)                # print total accuracy
            self.print_class_acc(conf_df)                   # print class accuracy
            self.test_acc = total_acc

        if mode == "image-segmentation":
            true_neg, false_pos, false_neg, true_pos = conf_mat.ravel()
            pixel_acc = np.mean(targets == pred_labels)
            iou = true_pos / (true_pos + false_neg + false_pos)
            dice_score = 2*true_pos / (2*true_pos + false_neg + false_pos)
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            specificity = true_neg / (true_neg + false_pos)
            print("Pixel Acc:", pixel_acc)                # print pixel accuracy
            print("Jaccard index (=IoU):", iou)             # print intersection over union
            print("Dice Score (=F1 score)", dice_score)     # print dice score
            print("Dice loss:", 1-dice_score)
            print("Precision:", precision)                  
            print("Recall (=Sensitivity):", recall)          
            print("Specificity (=Recall of 0):", specificity)
            self.test_acc = dice_score


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

    def load(self, path, verbose=True):
        device = torch.device('cuda')
        state_dict = torch.load(path)
        
        self.model.load_state_dict(state_dict["model"])
        self.model.to(device)
        
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        else:
            self.scheduler = None
        self.test_acc = state_dict["test_acc"]

        n_train_loss = len(state_dict["train_loss"])
        n_valid_acc = len(state_dict["valid_acc"])
        epoch_trained = state_dict["epoch_trained"]
        n = min(epoch_trained, n_train_loss, n_valid_acc)

        self.epoch_trained = n
        self.train_loss = state_dict["train_loss"][:n]
        self.valid_acc = state_dict["valid_acc"][:n]
        if verbose:
            print("State dict sucessfully loaded.")


    # ----------------------- save() ----------------------------------------- #

    def save(self, path, verbose=True):
        state_dict = {}
        state_dict["model"] = self.model.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        else:
            state_dict["scheduler"] = None
        state_dict["epoch_trained"] = self.epoch_trained
        state_dict["train_loss"] = self.train_loss
        state_dict["valid_acc"] = self.valid_acc
        state_dict["test_acc"] = self.test_acc
        torch.save(state_dict, path)
        if verbose:
            print("State dict saved.")
