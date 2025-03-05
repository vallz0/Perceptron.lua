local NeuralNetwork = {}
NeuralNetwork.__index = NeuralNetwork

local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

local function sigmoidDerivative(x)
    local sig = sigmoid(x)
    return sig * (1 - sig)
end

function NeuralNetwork.new(inputSize, hiddenSize, outputSize)
    local self = setmetatable({}, NeuralNetwork)
    
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.outputSize = outputSize
    
    self.weightsInputHidden = self:initializeWeights(inputSize, hiddenSize)
    self.weightsHiddenOutput = self:initializeWeights(hiddenSize, outputSize)
    self.biasHidden = self:initializeBias(hiddenSize)
    self.biasOutput = self:initializeBias(outputSize)
    
    return self
end

function NeuralNetwork:initializeWeights(inputSize, outputSize)
    local weights = {}
    for i = 1, inputSize do
        weights[i] = {}
        for j = 1, outputSize do
            weights[i][j] = math.random() - 0.5 
        end
    end
    return weights
end

function NeuralNetwork:initializeBias(size)
    local bias = {}
    for i = 1, size do
        bias[i] = math.random() - 0.5
    end
    return bias
end

function NeuralNetwork:feedForward(inputs)
    if #inputs ~= self.inputSize then
        error("Tamanho da entrada inválido!")
    end
    
    local hidden = {}
    for i = 1, self.hiddenSize do
        local sum = self.biasHidden[i]
        for j = 1, self.inputSize do
            sum = sum + inputs[j] * self.weightsInputHidden[j][i]
        end
        hidden[i] = sigmoid(sum)
    end
    
    local outputs = {}
    for i = 1, self.outputSize do
        local sum = self.biasOutput[i]
        for j = 1, self.hiddenSize do
            sum = sum + hidden[j] * self.weightsHiddenOutput[j][i]
        end
        outputs[i] = sigmoid(sum)
    end
    
    return outputs, hidden
end

function NeuralNetwork:train(inputs, targets, learningRate)
    local outputs, hidden = self:feedForward(inputs)
    
    local outputErrors = {}
    for i = 1, self.outputSize do
        outputErrors[i] = targets[i] - outputs[i]
    end
    
    for i = 1, self.hiddenSize do
        for j = 1, self.outputSize do
            local delta = outputErrors[j] * sigmoidDerivative(outputs[j]) * hidden[i] * learningRate
            self.weightsHiddenOutput[i][j] = self.weightsHiddenOutput[i][j] + delta
        end
    end
    for i = 1, self.outputSize do
        local delta = outputErrors[i] * sigmoidDerivative(outputs[i]) * learningRate
        self.biasOutput[i] = self.biasOutput[i] + delta
    end
    
    local hiddenErrors = {}
    for i = 1, self.hiddenSize do
        hiddenErrors[i] = 0
        for j = 1, self.outputSize do
            hiddenErrors[i] = hiddenErrors[i] + outputErrors[j] * self.weightsHiddenOutput[i][j]
        end
    end
    
    for i = 1, self.inputSize do
        for j = 1, self.hiddenSize do
            local delta = hiddenErrors[j] * sigmoidDerivative(hidden[j]) * inputs[i] * learningRate
            self.weightsInputHidden[i][j] = self.weightsInputHidden[i][j] + delta
        end
    end
    for i = 1, self.hiddenSize do
        local delta = hiddenErrors[i] * sigmoidDerivative(hidden[i]) * learningRate
        self.biasHidden[i] = self.biasHidden[i] + delta
    end
end

local function main()
    local nn = NeuralNetwork.new(2, 4, 1)
    
    -- (XOR problem)
    local trainingData = {
        {inputs = {0, 0}, target = {0}},
        {inputs = {0, 1}, target = {1}},
        {inputs = {1, 0}, target = {1}},
        {inputs = {1, 1}, target = {0}}
    }
    
    for i = 1, 10000 do
        for _, data in pairs(trainingData) do
            nn:train(data.inputs, data.target, 0.1)
        end
    end
    
    for _, data in pairs(trainingData) do
        local outputs = nn:feedForward(data.inputs)
        print("Entrada:", data.inputs[1], data.inputs[2], "Saída:", outputs[1])
    end
end

main()
