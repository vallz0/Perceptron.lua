local Perceptron = {}
Perceptron.__index = Perceptron

function Perceptron.new(numInputs, learningRate) 
	local self = setmetatable({}, Perceptron)

	self.weights = {}
	for i = 1, numInputs do
		self.weights[i] = 0
	end

	self.bias = 0
	self.learningRate = learningRate or 0.01

	return self
end

function Perceptron:activationFunction(z)
	return z > 0 and 1 or 0
end

function Perceptron:predict(inputs)
	local weightedSum = 0
	for i, inputValue in ipairs(inputs) do
		weightedSum = weightedSum + self.weights[i] * inputValue
	end
	weightedSum = weightedSum + self.bias
	return self:activationFunction(weightedSum)
end

function Perceptron:train(trainingData, labels, epochs)
	epochs = epochs or 10

	for epoch = 1, epochs do
		print("Epoch " .. epoch)
		for i, inputs in ipairs(trainingData) do
			local label = labels[i]

			local prediction = self:predict(inputs)

			local error = label - prediction

			for j, inputValue in ipairs(inputs) do
				self.weights[j] = self.weights[j] + self.learningRate * error * inputValue
			end
			self.bias = self.bias + self.learningRate * error

			print("Inputs: ", table.concat(inputs, ", "), ", Prediction: ", prediction, ", Error: ", error, ", Weights: ", table.concat(self.weights, ", "), ", Bias: ", self.bias)
		end
		print()
	end
end

local trainingData = {
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1}
}

local labels = {0, 1, 1, 1} 

local perceptron = Perceptron.new(2, 0.1)

perceptron:train(trainingData, labels, 10)

print("Testing the perceptron:")
for _, inputs in ipairs(trainingData) do
	print("Input: ", table.concat(inputs, ", "), ", Prediction: ", perceptron:predict(inputs))
end
