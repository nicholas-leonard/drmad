--[[
Multiple meta-iterations for DrMAD on MNIST
tuning learning rates and L2 norms together
MIT license
]]


-- Import libs
require 'torch'
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local dl = require 'dataload'
local xlua = require 'xlua'
local deepcopy = require 'deepcopy'
--local debugger = require 'fb.debugger'

grad.optimize(true)


--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Multiple meta-iterations for DrMAD on MNIST tuning learning rates and L2 norms together')
cmd:text('Options:')
cmd:option('-cuda', false, 'use CUDA')
cmd:option('-device', 1, 'sets the device (GPU) to use')
cmd:option('-epochSize', -1, 'number of iterations per training epoch')
cmd:text()
local opt = cmd:parse(arg or {})

if opt.cuda then
   require 'cunn'
   cutorch.setDevice(opt.device)
end

-- Load in MNIST
local trainset, validset, testset = dl.loadMNIST()
local transValidData = {
    size = 10000,
    x = torch.FloatTensor(10000, 1, 28 * 28):fill(0),
    y = torch.FloatTensor(10000, 1, 10):fill(0)
}

local inputSize = trainset.inputs[1]:nElement()
local classes = testset.classes
local cm = optim.ConfusionMatrix(classes)

local initHyper = 0.001
local predict, fTrain, params, prevParams

-- elementary learning rate: eLr
-- set it small to avoid NaN issue
local eLr = 0.0001
local numEpoch = 1
local batchSize = 1
local epochSize = opt.epochSize
local nLayers = 3
-- number of iterations
local numIter = numEpoch * (epochSize == -1 and trainset:size() or epochSize)
--local numIter = 50000/numEpoch

-- prototype tensor used to initialize other tensors via proto.new()
local proto = opt.cuda and torch.CudaTensor() or torch.FloatTensor()

-- initialize learning rate vector for each layer, at every iteration
local LR = proto.new(numIter, 3):fill(eLr)

-- Define elementary parameters
torch.manualSeed(0)

-- define velocities for weights
local VW = { proto.new(inputSize, 50), proto.new(50, 50), proto.new(50, #classes) }

-- define velocities for biases
local VB = { proto.new(50), proto.new(50), proto.new(#classes) }

-- Trainable parameters (weight and bias):
local params = {}
params.W = { proto.new(inputSize, 50), proto.new(50, 50), proto.new(50, #classes)}
params.B = { proto.new(50), proto.new(50), proto.new(#classes) }

-- define validGrads
local validGrads = deepcopy(params)

-- initialize hyperparameters as global variables to be shared across different meta-iterations
params.HY = { proto.new(inputSize, 50):fill(initHyper), proto.new(50, 50):fill(initHyper), proto.new(50, #classes) }

-- copies of the parameter tensors
local initParams = deepcopy(params)
local finalParams = deepcopy(params)

-- Initialize derivative w.r.t. hyperparameters)
local DHY = deepcopy(params.HY)

-- Initialize derivative w.r.t. learning rates
local DLR = proto.new(numIter, 3)

local proj = deepcopy(params.HY)

-- Initialize derivative w.r.t. velocity
local DV = deepcopy(params.HY)

local beta = torch.linspace(0.001, 0.999, numIter)

-- buffers
local y_, x_ = proto.new(), proto.new()

-- What model to train:

-- Define neural net
local function predict(params, input)
  local h1 = torch.tanh(input * params.W[1] + params.B[1])
  local h2 = torch.tanh(h1 * params.W[2] + params.B[2])
  local h3 = h2 * params.W[3] + params.B[3]
  local out = util.logSoftMax(h3)
  return out
end

-- Define training loss
local function fTrain(params, input, target)
  local prediction = predict(params, input)
  local loss = lossFuns.logMultinomialLoss(prediction, target)
  local penalty1 = torch.sum(torch.cmul(torch.cmul(params.W[1], params.HY[1]), params.W[1]))
  local penalty2 = torch.sum(torch.cmul(torch.cmul(params.W[2], params.HY[2]), params.W[2]))
  local penalty3 = torch.sum(torch.cmul(torch.cmul(params.W[3], params.HY[3]), params.W[3]))
  loss = loss + penalty1 + penalty2 + penalty3
  return loss, prediction
end


local function train_meta()
    --[[
    One meta-iteration to get directives w.r.t. hyperparameters
    ]]
    
    -- initialize elementary parameters and velocities
    torch.manualSeed(0)
    for i, W in ipairs(params.W) do
        local bound = 1 / math.sqrt(W:size(2))
        W:uniform(-bound, bound)
        VW[i]:fill(0)
    end
    for i, B in ipairs(params.B) do
        B:fill(0)
        VB[i]:fill(0)
    end

    
    -- copy initial weights
    initParams = nn.utils.recursiveCopy(initParams, params)

    -- Get the gradients closure magically:
    local dfTrain = grad(fTrain, { optimize = true })

    ------------------------------------
    -- [[Forward pass]]
    -----------------------------------


    -- weight decay for elementary parameters
    local gamma = 0.7
    -- Train a neural network to get final parameters
    local function makesample(inputs, targets)
        assert(inputs:size(1) == 1)
        assert(inputs:dim() == 4)
        --assert(torch.type(inputs) == 'proto.new')
        local x = inputs:view(1, -1)
        if opt.cuda then
            x_:resize(x:size()):copy(x)
        else
            x_ = x
        end
        y_:resize(10):zero()
        y_[targets[1]] = 1 -- onehot
        return x_, y_:view(1, 10)
    end

    for epoch = 1, numEpoch do
        print('Forward Training Epoch #' .. epoch)
        for i, inputs, targets in trainset:subiter(batchSize, epochSize) do
            -- Next sample:
            local x, y = makesample(inputs, targets)

            -- Grads:
            local grads, loss, prediction = dfTrain(params, x, y)

            -- Update weights and biases at each layer
            -- consider weight decay
            for j = 1, #params.W do
                VW[j] = VW[j]:mul(gamma) - grads.W[j]:mul(1-gamma)
                VB[j] = VB[j]:mul(gamma) - grads.B[j]:mul(1-gamma)
                params.W[j] = params.W[j] + VW[j] * LR[{i, j}]
                params.B[j] = params.B[j] + VB[j] * LR[{i, j}]
            end

            -- Log performance:
            cm:add(prediction[1], y[1])
            if i % 1000 == 0 then
                print("Epoch " .. epoch .. ", iteration " .. i)
                cm:updateValids()
                print("Accuracy "..cm.totalValid)
                cm:zero()
            end
        end
    end

    -- copy final parameters after convergence
    finalParams = nn.utils.recursiveCopy(finalParams, params)

    ----------------------
    -- [[Backward pass]]
    -----------------------

    -- Transform validation data

    transValidData.y:zero()
    for t, inputs, targets in validset:subiter(batchSize, epochSize) do
        transValidData.x[t]:copy(inputs:view(-1))
        transValidData.y[{ t, 1, targets[1] }] = 1 -- onehot
    end

    -- Define validation loss
    local validLoss = 0

    function fValid(params, input, target)
        local prediction = predict(params, input)
        local loss = lossFuns.logMultinomialLoss(prediction, target)
        return loss, prediction
    end

    local dfValid = grad(fValid, { optimize = true })

    -- zero validGrads
    for i=1,3 do
        validGrads.W[i]:zero()
        validGrads.B[i]:zero()
    end
    
    -- Get gradient of validation loss w.r.th. finalParams
    -- Test network to get validation gradients w.r.t weights
    for epoch = 1, numEpoch do
        print('Forward Training Epoch #' .. epoch)
        for i = 1, epochSize == -1 and transValidData.size or epochSize do
            -- Next sample:
            local x = transValidData.x[i]:view(1, inputSize)
            local y = torch.view(transValidData.y[i], 1, 10)
            
            if opt.cuda then
               x_:resize(x:size()):copy(x)
               y_:resize(y:size()):copy(y)
               x, y = x_, y_
            end

            -- Grads:
            local grads, loss, prediction = dfValid(params, x, y)
            for i = 1, #params.W do
                validGrads.W[i] = validGrads.W[i] + grads.W[i]
                validGrads.B[i] = validGrads.B[i] + grads.B[i]
            end
        end
    end

    -- Get average validation gradients w.r.t weights and biases
    for i = 1, #params.W do
        validGrads.W[i] = validGrads.W[i] / numEpoch
        validGrads.B[i] = validGrads.B[i] / numEpoch
    end

    -------------------------------------

    -- Initialize gradients
    for i=1,nLayers do
        DHY[i]:zero() -- w.r.t. hyper-params
        proj[i]:zero()
        DV[i]:zero() -- w.r.t. velocity
    end
    DLR:zero() -- w.r.t. learning rates

    -- https://github.com/twitter/torch-autograd/issues/66
    -- torch-autograd needs to track all variables
    local function gradProj(params, input, target, proj_1, proj_2, proj_3, DV_1, DV_2, DV_3)
        local grads, loss, prediction = dfTrain(params, input, target)
        proj_1 = proj_1 + torch.cmul(grads.W[1], DV_1)
        proj_2 = proj_2 + torch.cmul(grads.W[2], DV_2)
        proj_3 = proj_3 + torch.cmul(grads.W[3], DV_3)
        local loss = torch.sum(proj_1) + torch.sum(proj_2) + torch.sum(proj_3)
        return loss
    end

    local dHVP = grad(gradProj)

    ----------------------------------------------
    -- Backpropagate the validation errors

    local buffer
    for epoch = 1, numEpoch do

        print('Backword Training Epoch #' .. epoch)
        for i, inputs, targets in trainset:subiter(batchSize, epochSize) do
            -- Next sample:
            local x, y = makesample(inputs, targets)

            -- start from the learning rate for the last time-step, i.e. reverse
            -- currently only consider weights

            prevParams = nn.utils.recursiveCopy(prevParams, params)
            for j = 1, nLayers do
                params.W[j]:mul(initParams.W[j], 1 - beta[i + (numEpoch * (epoch - 1))])
                buffer = buffer or initParams.W[j].new()
                buffer:mul(finalParams.W[j], beta[i + (numEpoch * (epoch - 1))])
                params.W[j]:add(buffer)

                -- using the setup 6 in Algorithm 2, ICML 2015 paper
                local lr = LR[{numIter - (i-1), j}]
                VW[j]:div(params.W[j], lr)
                VW[j]:add(-1/lr, prevParams.W[j])
                DLR[{numIter - (i-1), j}] = torch.dot(validGrads.W[j], VW[j])
                DV[j]:add(LR[{i, j}], validGrads.W[j])
            end

            local grads, loss = dHVP(params, x, y, proj[1], proj[2], proj[3], DV[1], DV[2], DV[3])
            --        print("loss", loss)
            for j = 1, nLayers do
                buffer = buffer or DHY[j].new()

                buffer:mul(grads.W[j], 1.0 - gamma)
                validGrads.W[j]:add(-1, buffer)

                buffer:mul(grads.HY[j], 1.0 - gamma)
                DHY[j]:add(-1, buffer)

                DV[j]:mul(DV[j], gamma)
            end
            --xlua.progress(i, trainset:size())
        end
    end
    return DHY, DLR
end

-----------------------------
-- entry point
------------------------

-- Hyperparameter learning rate, cannot be too huge
-- this is a super-parameter...

-- hyperparameter learning rate for elementary learning rate
local hLr_lr = 0.00000001
-- hyperparameter learning rate for l2 penalty
local hLr_l2 = 0.0001
local numMeta = 5

for i = 1, numMeta do
    local dhy, dlr = train_meta()

    for j = 1, #params.W do
        -- modify elementary learning rates
        dlr[{{}, j}]:mul(hLr_lr)
        LR[{{}, j}]:add(dlr[{{}, j}])
        -- modify l2 penalties
        dhy[j]:mul(-hLr_l2)
        params.HY[j]:add(dhy[j])
    end
end

for i, hy in ipairs(params.HY) do
    print("HY " .. i, hy:sum())
end

