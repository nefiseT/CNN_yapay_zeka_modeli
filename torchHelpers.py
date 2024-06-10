import time
import datetime
import logging
import torch

class Trainer:
    """
    Trainer class for training a neural network model in PyTorch.


    Parameters
    ----------

    model : The neural network model to train.

    lossFunction : The loss function to use for training.

    optimizer : The optimizer to use for training.

    device : The device to use for training. Can be either 'cpu', 'mps' or 'cuda'.

    logLevel :  The log level to use for logging. Can be one of the following:
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        
    Usage
    -----

    # create a trainer
    trainer = Trainer(model, lossFunction, optimizer, device, logLevel=logging.INFO)
    # train the model
    trainer.train(trainLoader, valLoader, numberOfEpochs)

    Author
    ------
    Markus Enzweiler (markus.enzweiler@hs-esslingen.de)

    """

    def __init__(self, model, lossFunction, optimizer, device, logLevel=logging.INFO):
        self.model = model
        self.lossFunction = lossFunction
        self.optimizer = optimizer  
        self.device = device
        self.haveValData = False
        self.trainBatchSize = 0
        self.valBatchSize = 0

        # logging
        self.logLevel = logLevel
        self.logger = None
        self.loggerStreamHandler = None
        self._setupLogger()
        
        # metrics, computed in each epoch
        self.metrics = dict()
        self.metrics["epochTrainLoss"] = []
        self.metrics["epochValLoss"] = []
        self.metrics["epochTrainAccuracy"] = []
        self.metrics["epochValAccuracy"] = []

        # timing
        self.metrics["epochStartTime"] = None
        self.metrics["epochEndTime"] = None
        self.metrics["trainingStartTime"] = None
        self.metrics["trainingEndTime"] = None


    def _setupLogger(self):
        logging.basicConfig(level = self.logLevel, force=True)
        self.logger = logging.getLogger('Trainer')

        self.loggerStreamHandler = logging.StreamHandler()
        self.loggerStreamHandler.setLevel(self.logLevel)
        formatter = logging.Formatter('%(message)s')
        self.loggerStreamHandler.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(self.loggerStreamHandler)
        self.logger.propagate = False


    def _initMetrics(self, numEpochs):
        self.metrics["epochTrainLoss"    ] = [0.0] * numEpochs
        self.metrics["epochValLoss"     ] = [0.0] * numEpochs
        self.metrics["epochTrainAccuracy"] = [0.0] * numEpochs
        self.metrics["epochValAccuracy" ] = [0.0] * numEpochs


    def _updateMetrics(self, epoch, numTrainSamples, numValSamples):
        # average loss and accuracy
        if numTrainSamples:
            self.metrics["epochTrainLoss"][epoch]     = self.metrics["epochTrainLoss"][epoch]     / numTrainSamples
            self.metrics["epochTrainAccuracy"][epoch] = self.metrics["epochTrainAccuracy"][epoch] / numTrainSamples 

        if self.haveValData and numValSamples:      
            self.metrics["epochValLoss"][epoch]      = self.metrics["epochValLoss"][epoch]      / numValSamples
            self.metrics["epochValAccuracy"][epoch]  = self.metrics["epochValAccuracy"][epoch]  / numValSamples


    def _logMetrics(self, epoch):       
        # log metrics
        self.loggerStreamHandler.terminator = ""
        self.logger.info(f'[Epoch {epoch:3}] : | ')
        self.logger.info(f'time: {self.metrics["epochEndTime"] - self.metrics["epochStartTime"] :.3f}s | ')

        self.logger.info(f'trainLoss: {self.metrics["epochTrainLoss"][epoch] :.3f}  | ')
        self.logger.info(f'trainAccuracy: {self.metrics["epochTrainAccuracy"][epoch] :.3f}  | ')

        if self.haveValData:
            self.logger.info(f'valLoss: {self.metrics["epochValLoss"][epoch] :.3f}  | ')
            self.logger.info(f'valAccuracy: {self.metrics["epochValAccuracy"][epoch] :.3f} | ')  
        else:
            self.logger.info(f'no validation data | ')  

        self.logger.info('\n')


    def _onTrainBegin(self, numEpochs):
        # Push the network model to the device we are using to train
        self.model.to(self.device)

        # init the metrics
        self._initMetrics(numEpochs)

        # start time
        self.metrics["trainingStartTime"] = time.monotonic()


    def _onTrainEnd(self, numEpochs):        
        # end time
        self.metrics["trainingEndTime"] = time.monotonic()
        timeDelta = datetime.timedelta(seconds=(self.metrics["trainingEndTime"] - self.metrics["trainingStartTime"]))

        # log
        self.loggerStreamHandler.terminator = "\n"
        self.logger.info(f'Training finished in {str(timeDelta)} hh:mm:ss.ms')

    
    def _onEpochBegin(self, epoch):
        # start time of epoch
        self.metrics["epochStartTime"] = time.monotonic()  

        # log info
        self.loggerStreamHandler.terminator = " "
        self.logger.info( f'[Epoch {epoch:3}] : ') 


    def _onEpochEnd(self, epoch, numTrainSamples, numValSamples, numBatches):        
        # end time of epoch
        self.metrics["epochEndTime"] = time.monotonic()  

        # log info
        self.loggerStreamHandler.terminator = "\n"
        self.logger.info(f' done ({numBatches} batches)') 

        # update metrics
        self._updateMetrics(epoch, numTrainSamples, numValSamples)

        # log metrics
        self._logMetrics(epoch)


    def _computeAccuracy(self, outputs, labels):
         # find predicted labels (the output neuron index with the highest output value)
        _, predictedLabels = torch.max(outputs, 1) 
        return torch.sum(predictedLabels == labels).detach().cpu().numpy()
        

    def _trainEpoch(self, epoch, trainLoader):
         # loop over batches in the dataset
        numBatches = 0

        for i, data in enumerate(trainLoader, 0):        
            # get the training data : data is a list of [images, labels]
            # and push the data to the device we are using       
            images, labels = data[0].to(self.device), data[1].to(self.device)

            # zero the parameter gradients before the next data batch is processed
            self.optimizer.zero_grad()

            # forward pass of the batch
            outputs = self.model(images)

            # loss computation at the output of the network
            loss = self.lossFunction(outputs, labels)

            # backpropagate the loss through the network
            loss.backward()

            # optimize the network parameters
            self.optimizer.step()

            # accumulate train loss
            self.metrics["epochTrainLoss"][epoch] += loss.item() * self.trainBatchSize

            # compute and accumulate train accuracy  
            batchAccuracy = self._computeAccuracy(outputs, labels)             
            self.metrics["epochTrainAccuracy"][epoch] += batchAccuracy

            if ((i % 100) == 0):
                self.loggerStreamHandler.terminator = ""
                self.logger.info ('.')

            numBatches = numBatches + 1

        return numBatches


    def _testEpoch(self, epoch, valLoader):
        # test on the validation data
        if valLoader:        
            # we do not compute gradients in inference mode     
            with torch.no_grad(): 

                # loop over validation data
                for data in valLoader:  

                    # push to device
                    images, labels = data[0].to(self.device), data[1].to(self.device)  

                    # forward pass through the network
                    outputs = self.model(images)

                    # compute loss (just to report it)
                    loss = self.lossFunction(outputs, labels)          
                    self.metrics["epochValLoss"][epoch] += loss.item() * self.valBatchSize

                    # compute accuracy                
                    self.metrics["epochValAccuracy"][epoch] += self._computeAccuracy(outputs, labels)


    def train(self, trainLoader, valLoader, numEpochs):
        # main training method    

        # check training and validation data
        if not (trainLoader and len(trainLoader) > 0):
            msg = 'No training data available'
            self.logger.error(msg)
            raise Exception(msg)

        if not (valLoader and len(valLoader) > 0):
            self.haveValData = False
        else:
            self.haveValData = True

        # number of train and validation samples
        numTrainSamples = (len(trainLoader.dataset) if trainLoader else 0)
        numValSamples   = (len(valLoader.dataset)  if valLoader  else 0)

        # batch sizes of train and validation loader
        self.trainBatchSize = (trainLoader.batch_size if trainLoader else 0)
        self.valBatchSize   = (valLoader.batch_size  if valLoader  else 0)


        # ------ Main training loop ------

        # do some stuff at the beginning of the training
        self._onTrainBegin(numEpochs)

        # loop over the dataset in each epoch
        for epoch in range(numEpochs):           

            # do some stuff at the beginning of each epoch
            self._onEpochBegin(epoch)

            # train an epoch on the training data
            numBatches = self._trainEpoch(epoch, trainLoader)

            # test on the validation data
            if self.haveValData:
                self._testEpoch(epoch, valLoader)

            # do some stuff at the end of each epoch
            self._onEpochEnd(epoch, numTrainSamples, numValSamples, numBatches)

         # do some stuff at the end of the training
        self._onTrainEnd(numEpochs)




class Tester:
    """
    Tester class for testing a neural network model in PyTorch.


    Parameters
    ----------

    model : The neural network model to test.

    device : The device to use for testing. Can be either 'cpu', 'mps' or 'cuda'.

    logLevel :  The log level to use for logging. Can be one of the following:
        logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
        
    Usage
    -----

    # create a tester
    tester = Tester(model, device, logLevel=logging.INFO)
    # test the model
    tester.test(testLoader)

    Author
    ------
    Markus Enzweiler (markus.enzweiler@hs-esslingen.de)

    """

    def __init__(self, model, device, logLevel=logging.INFO):
        self.model = model     
        self.device = device      
        self.testBatchSize = 0

        # logging
        self.logLevel = logLevel
        self.logger = None
        self.loggerStreamHandler = None
        self._setupLogger()
        
        # metrics
        self.metrics = dict()
        self.metrics["accuracy"] = 0.0


    def _setupLogger(self):
        logging.basicConfig(level = self.logLevel, force=True)
        self.logger = logging.getLogger('Tester')

        self.loggerStreamHandler = logging.StreamHandler()
        self.loggerStreamHandler.setLevel(self.logLevel)
        formatter = logging.Formatter('%(message)s')
        self.loggerStreamHandler.setFormatter(formatter)
        
        self.logger.handlers.clear()
        self.logger.addHandler(self.loggerStreamHandler)
        self.logger.propagate = False


    def _initMetrics(self):
        self.metrics["accuracy"] = 0.0


    def _updateMetrics(self, numTestSamples):
        # average accuracy
        if numTestSamples:
            self.metrics["accuracy"] = self.metrics["accuracy"] / numTestSamples


    def _logMetrics(self, numTestSamples):       
        # log metrics
        self.loggerStreamHandler.terminator = ""     
       
        self.logger.info('\n')
        self.logger.info(f'Test Metrics ({numTestSamples} test samples):\n')
        self.logger.info(f'  - Accuracy: {self.metrics["accuracy"]:.3f}')
        self.logger.info('\n')
           

    def _onTestBegin(self):
        # Push the network model to the device we are using to train
        self.model.to(self.device)

        # init the metrics
        self._initMetrics()

        # start time
        self.metrics["testingStartTime"] = time.monotonic()

        # logging output
        self.loggerStreamHandler.terminator = ""          
        self.logger.info(f'Testing ')


    def _onTestEnd(self, numTestSamples):  

        # update Metrics
        self._updateMetrics(numTestSamples)

         # log metrics
        self._logMetrics(numTestSamples)

        # end time
        self.metrics["testingEndTime"] = time.monotonic()
        timeDelta = datetime.timedelta(seconds=(self.metrics["testingEndTime"] - self.metrics["testingStartTime"]))

        # log
        self.loggerStreamHandler.terminator = "\n"
        self.logger.info(f'Testing finished in {str(timeDelta)} hh:mm:ss.ms')


    def _computeAccuracy(self, outputs, labels):
         # find predicted labels (the output neuron index with the highest output value)
        _, predictedLabels = torch.max(outputs, 1) 
        return torch.sum(predictedLabels == labels).detach().cpu().numpy()
        

    def test(self, testLoader):   
        # number of test samples
        numTestSamples = (len(testLoader.dataset) if testLoader else 0)
        
        # batch size of test loader
        self.testBatchSize = (testLoader.batch_size if testLoader else 0)
        

        # ------ Main test loop ------

        # do some stuff at the beginning of the training
        self._onTestBegin()

        if testLoader:        
            # we do not compute gradients in inference mode     
            with torch.no_grad(): 

                # loop over test data
                i = 0
                for data in testLoader:  

                    # push to device
                    images, labels = data[0].to(self.device), data[1].to(self.device)  

                    # forward pass through the network
                    outputs = self.model(images)
                   
                    # compute accuracy                
                    self.metrics["accuracy"] += self._computeAccuracy(outputs, labels)

                    # status update
                    if ((i % 100) == 0):
                        self.loggerStreamHandler.terminator = ""
                        self.logger.info ('.')

                    i = i+1

         # do some stuff at the end of the testing
        self._onTestEnd(numTestSamples)
