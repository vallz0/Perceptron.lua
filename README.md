

# Multilayer Perceptron and Simple Perceptron for Roblox Studio

Welcome to this repository! Here you'll find two neural network implementations in Lua, designed for use in **Roblox Studio**. They are built using **Object-Oriented Programming (OOP)** and **Clean Code** principles, making them ideal for experimenting with artificial intelligence in games.

- **`NeuralNetwork.lua`**: A Multilayer Perceptron (MLP) with backpropagation, capable of solving nonlinear problems like XOR.
- **`Perceptron.lua`**: A simple Perceptron, suitable for linearly separable problems like the OR function.

## Repository Structure

```
├── NeuralNetwork.lua  # Multilayer Perceptron implementation
├── Perceptron.lua     # Simple Perceptron implementation
└── README.md          # This file
```

## Prerequisites

- **Roblox Studio**: Ensure you have Roblox Studio installed to run the scripts.
- Basic knowledge of Lua and Roblox scripting.

## NeuralNetwork.lua

### Description
This file contains an implementation of a **Multilayer Perceptron (MLP)** with one hidden layer. It uses the sigmoid activation function and backpropagation for learning.

### Features
- Supports customizable input, hidden, and output layers.
- Random initialization of weights and biases.
- Practical example solving the XOR problem.

### Usage
1. Copy the contents of `NeuralNetwork.lua` into a **Script** in Roblox Studio.
2. Run the script directly (via "Play") or attach it to an event.
3. The built-in example trains the network to solve XOR and prints results to the console.

### Example Code
```lua
local nn = NeuralNetwork.new(2, 4, 1) -- 2 inputs, 4 hidden, 1 output
nn:train({0, 1}, {1}, 0.1)           -- Train with input [0, 1] and target [1]
local outputs = nn:feedForward({0, 1}) -- Predict the output
print(outputs[1])                     -- Display the prediction
```

### Customization
- Adjust `inputSize`, `hiddenSize`, and `outputSize` in the constructor.
- Modify the training data in the `trainingData` array.
- Change the learning rate (default: `0.1`) and number of iterations.

## Perceptron.lua

### Description
This file implements a **Simple Perceptron**, a single-layer neural network that solves linearly separable problems. It uses a step activation function and supervised learning.

### Features
- Ideal for simple problems like logic gates (AND, OR).
- Training with an adjustable learning rate.
- Practical example implementing the OR function.

### Usage
1. Copy the contents of `Perceptron.lua` into a **Script** in Roblox Studio.
2. Run the script to train and test the perceptron in the console.
3. Results are displayed as logs in the Roblox Studio Output.

### Example Code
```lua
local perceptron = Perceptron.new(2, 0.1) -- 2 inputs, learning rate 0.1
perceptron:train({{0, 1}}, {1}, 10)       -- Train with input [0, 1] and target 1
print(perceptron:predict({0, 1}))         -- Predict the output
```

### Customization
- Change the number of inputs in the constructor (`numInputs`).
- Modify `trainingData` and `labels` for your use case.
- Adjust the learning rate (`learningRate`) and number of epochs (`epochs`).
