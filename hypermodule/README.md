# HyperModule

`HyperModule` is a wrapper of functions for managing the training, validation, and testing processes of neural networks. With HyperModule, it is easier to monitor the progress of the training process by logging loss and validation accuracy. Additionally, HyperModule provides convenient functions for loading and saving pre-trained models, streamlining the entire process of working with neural networks.

## Usage

### Start training

```Python
model = YourNeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

hm = HyperModule(
    model = model,
    criterion = criterion,
    optimizer = optimizer,
    scheduler = None
)
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

## Loading and Saving

The *information* being loaded and saved in `HyperModule` is

* `state_dict` of neural network
* `state_dict` of optimizer
* `state_dict` of scheduler
* number of epochs that neural network has been trained
* training loss in each epoch
* validation accuracy in each epoch
* testing accuracy
