-- https://github.com/twitter/torch-autograd/tree/master/examples

local util = require 'autograd.util'
require 'torch'
local path = require 'pl.path'
-- Get/create dataset
local function setupData(opt)
--   if not path.exists(sys.fpath()..'/mnist') then
--      os.execute[[
--      curl https://s3.amazonaws.com/torch.data/mnist.tgz -o mnist.tgz
--      tar xvf mnist.tgz
--      rm mnist.tgz
--      ]]
--   end

   local classes = {'0','1','2','3','4','5','6','7','8','9'}
   local trainData = torch.load('/home/jie/Downloads/mnist/train.t7')
   local testData = torch.load('/home/jie/Downloads/mnist/test.t7')
   trainData.y = util.oneHot(trainData.y)
   testData.y = util.oneHot(testData.y)
   return trainData, testData, classes
end

return setupData
