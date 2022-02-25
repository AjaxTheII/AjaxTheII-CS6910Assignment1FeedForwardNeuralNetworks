# Assignment 1

This directory contains solution to [assignment 1](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44) of Fundamentals of Deep Learning (CS6910), Spring 2022. It also contains the implementation of a Feed Forward Neural Network **Feed Forward Neural Network** which can be trained to work on numerical data.

The solution report with results can be found [here](https://wandb.ai/cs21s048-cs21s058/Assignment1-FeedFwdNeuralNw/reports/Assignment-1--VmlldzoxNjA1MjIy?accessToken=i0ra3meu0j7ov4qpvhj0vwahwtgkgbzpr33chj2b6uh330xeqc9epun3ty3nepk5).

An Object oriented approach has been followed to implement the neural network.

## Contents of the File
The main content of this file is the class ```class Layer``` which implements a single layer of neural network. It initializes the weight and biases of each layer and also the activation function. 
```python
class Layer:

    activationFunc = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid),
        'relu' : (relu, d_relu),
        'softmax' : (softmax, None)
    }

    def __init__(self, inputs, neurons, activation):
        
        #Xavier initialization
        np.random.seed(33)
        sd = np.sqrt(2 / float(inputs + neurons))
        self.W = np.random.normal(0, sd, size=(neurons, inputs))  
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunc.get(activation)
        self.dW = 0
        self.db = 0
```

### The activation function for the hidden layers:  
**activation functions** : ```sigmoid```, ```tanh``` and ```relu```  




### The functions implemented for the algorithms: 
#### **optimizer** : ```sgd```, ```momentum```, ```nesterov```, ```rmsprop```, ```adam```, ```nadam``` 
 
The loss functions implemented
#### **loss** = {'cross_entropy_loss', 'squared_error'}


### The function ```forward_propagation``` has been implemented:
Takes dataset ```h``` and ```layers``` (list of objects corresponding to each layer of the neural network) as input.
Returns forward propagated value of the data points. 


### The function ```backward_propagation``` has been implemented:
Takes dataset prediction by forward propagation ```y_hat``` and ```layer``` object as input.
Compute and update the gradients of each ```layer``` object and returs the list ```layers```


### The ```optimizor``` function takes the following parameters as input:
```layers```, ```optimizer```, ```epochs```, ```learning_rate```, ```x_train```, ```y_train```, ```x_val```, ```y_val```, ```batch_size```
The function calls the appropriate ```optimizer``` function and returns the validation accuracy, validation loss, and prints the loss in each epoch.

```x_val``` and ```y_val```  are data kept for validation of the model(10%).



### One can call the ```model_train``` function with the following parameters: 

 
* ```epochs``` :      No. of steps for training
* ```learning_rate```:      Eg: 0.001 
* ```neurons```:      No. of neurons in each hidden layer
* ```h_layers```:      Np. of hidden layers 
* ```activation```:      Activation function to be used in each layer (eg: ```relu```)
* ```batch_size```:      After how many iterations gradients will be updated
* ```optimizer```:      Optimization algorithm to be used (Eg: ```nadam```)
* ```x_train```, ```y_train``` :      Training data
* ```x_val```, ```y_val``` :      Validation data


#### The ```model_train``` function initializes the ```layers``` list of objects for each layer, and calls the ```optimizer``` function based on the argument ```optimizer``` which in turn trains the model.

```python
activation = 'relu'
batch_size = 64
epochs = 10
h_layers = 4
learning_rate = 0.0001
neurons = 128
optimizer = 'nadam'

output_test = model_train(epochs, learning_rate, neurons, h_layers, activation, batch_size, optimizer, x_train, y_train, x_val, y_val)
```


#### Finally, one can call the  ```predict``` function. 
Takes input data and the ```layers``` as input and returns the ```prediction```, ```accuracy``` and average ```loss```.
```python
output_test, accuracy_test, loss_test = predict(x_test.T, y_test, layers)
```

The code has been made flexible so that we can add new optimization algorithms.
