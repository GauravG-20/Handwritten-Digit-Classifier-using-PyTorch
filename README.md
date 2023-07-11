# MNIST Handwritten Digits Recognition using PyTorch
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://github.com/GauravG-20)

![](https://badgen.net/badge/Code/Python/purple?icon=https://simpleicons.org/icons/python.svg&labelColor=cyan&label)        ![](https://badgen.net/badge/Library/Pytorch/purple?icon=https://simpleicons.org/icons/pytorch.svg&labelColor=cyan&label)       ![](https://badgen.net/badge/Tools/pandas/purple?icon=https://simpleicons.org/icons/pandas.svg&labelColor=cyan&label)       ![](https://badgen.net/badge/Tools/numpy/purple?icon=https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg&labelColor=cyan&label)        ![](https://badgen.net/badge/Tools/matplotlib/purple?icon=https://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg&labelColor=cyan&label)

This project is a handwritten digit classifier implemented using PyTorch. The goal is to accurately recognize and classify handwritten digits from 0 to 9. The project utilizes deep learning techniques and convolutional neural networks (CNNs) to achieve high accuracy in digit recognition.

## MNIST dataset

The `MNIST` database is available at http://yann.lecun.com/exdb/mnist/

The `MNIST` database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

![Cover](https://github.com/priyavrat-misra/handwritten-digit-classification/blob/master/visualizations/test_results_with_val.png?raw=true "sample test results visualization")

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centred in a fixed-size image.

Thanks to Yann LeCun, Corinna Cortes and Christopher J.C. Burges.

## Training and Evaluation
The model is trained on the MNIST training dataset using techniques such as stochastic gradient descent (SGD) or Adam optimizer. The training process involves forward and backward propagation, weight updates, and iterative optimization to minimize the loss function. During training, the model's performance is monitored on a separate validation set to prevent overfitting.

After training, the model's accuracy and performance are evaluated on the MNIST test dataset. The accuracy metric represents the percentage of correctly classified digits. Additionally, other evaluation metrics such as precision, recall, and F1-score may be used to assess the model's performance across different digit classes.

## Parameters Initialization
* Both models have been initialized with random weights sampled from a normal distribution and bias with 0.
* These parameters have been initialised only for the Linear layers present in both of the models.
* If `n` represents the number of nodes in a Linear Layer, then weights are given as a sample of normal distribution in the range `(0,y)`. Here `y` represents standard deviation calculated as `y=1.0/sqrt(n)`
* Normal distribution is chosen since the probability of selecting a set of weights closer to zero in the distribution is more than that of the higher values, unlike in Uniform distribution where the probability of choosing any value is equal.

## Model Description

> |S.No.| Model Architecture | ***Epochs*** | ***Input Layer Dimension*** | ***Hidden Layer*** | Output Layer Dimension | Dropout | ***Pooling Layers*** | Loss Function | Optimizer | ***Transform on Dataset*** | ***Validation Accuracy*** | ***Test Accuracy*** |
> | :-: | :----------------: | :----------: | :-------------------------: | :----------------: | :--------------------: | :-----: | :------------------: | :-----------: | :-------: | :------------:| :----------------: | :-----------: |
> | 1. | Convolution Neural Network + Linear Neural Network | ***10*** | ***3136 Nodes*** | 2 Convolutional Layers: 8 and 16 filter resp. & ***3 Linear Layers: 784, 496, 49 Nodes resp*** | 10 Nodes | Convolution Layers: 40% & 20%, Linear Layers: 60%, 30% & 15% Resp. | ***2 MaxPool2d*** | CrossEntropyLoss | Adam, lr=0.001 | ***Random Rotation Transform only*** | ***97.85%*** | ***97.69%*** |
> | 2. | Tuned Convolution Neural Network + Linear Neural Network | ***15*** | ***784 Nodes*** | 2 Convolutional Layers: 8 and 16 filter resp. & ***3 Layers: 512, 128, 64 Nodes resp.*** | 10 Nodes |  Convolution Layers: 40% & 20%, 50%, 30% & 15% Resp. | ***1 MaxPool2d & 1 AvgPool2d*** | CrossEntropyLoss | Adam, lr=0.001 | ***Random Rotation & Normalized Transform*** | ***99.00%*** | ***99.10%*** |

## Future Enhancements
While the current model achieves high accuracy in handwritten digit classification, there are several avenues for future enhancements:

1. Exploring more advanced CNN architectures, such as ResNet or DenseNet, to improve accuracy further.
2. Investigating techniques like data augmentation to enhance the model's robustness and generalization capabilities.
3. Extending the project to handle more complex tasks, such as multi-digit recognition or digit segmentation in real-world images.

## Acknowledgements
This project would not have been possible without the valuable resources and tools provided by the PyTorch community. Additionally, the MNIST dataset and associated research have significantly contributed to the advancement of digit recognition.

## License
This project is licensed under the MIT License. Please see the LICENSE file for more details.

## Contributors
* [Gaurav Gupta](https://github.com/GauravG-20)
