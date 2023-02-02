
`HyperModule` is a wrapper for functions relative to training, validating and testing neural network. It also helps to log loss and validation accuracy during training and provide simple functions for loading and saving trained networks.

## Usage

### Start training

```{Python}
model = YourNeuralNetwork(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

hm = HyperModule(
    model = model,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = None
)

# Train in testing mode
hm.train(train_dataloader, valid_dataloader, save_path, num_epochs=100)
```


## Class structure

* `__init__()`: Assign neural network, optimizer, loss function and scheduler to a hypermodule instance
* `train()`
  * `update_()`: Conduct forward path and backward proporgation
  * `update_progress_()`: Update epoch information for `tqdm`
  * `update_scheduler_()`: Update scheduler
  * `update_history_()`: log training loss and validation accuracy.
  * `perform_validation_()`: perform validation by calling `valudate()` if `valid_dataloader` is not None.
* `validate()`
* `test()`: Test the performance of neural network
  * `get_prediction_()`: get predicted labels and ground truth of each batch
  * `visualize_class_acc_`: plot confusion heatmap and print testing accuracy of each class
    * `generate_confusion_df_()`: generate pandas `DataFrame` of confusion matrix
    * `plot_confusion_heatmap_()`: plot confusion heatmap
    * `print_class_acc_`: print accuracy of each class
* `load()`: read *information* of neural network from the given path.
* `save()`: save *information* to the given path.


The *information* being loaded and saved in `HyperModule` is:

* `state_dict` of neural network
* `state_dict` of optimizer
* `state_dict` of scheduler
* number of epochs that neural network has been trained
* training loss in each epoch
* training accuracy in each epoch
* testing accuracy
