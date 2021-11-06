import os
import time
import datetime

from data import Data
from matplotlib import pyplot
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, Flatten, Embedding
from tensorflow.keras.layers import ReLU, LeakyReLU, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

class AuxiliaryClassifierGAN:
    def __init__(self, data:Data, inputShape, imageDim, labelDim, latentDim, classes, params=None) -> None:
        self.initHyperParameters(params)
        self.initMetricsVars()
        self.evalDirectoryName = 'eval'

        self.data = data
        self.latentDim = latentDim
        self.discriminator = self.createDiscriminator(inputShape, classes)
        self.generator = self.createGenerator(latentDim, imageDim, labelDim, classes)
        self.gan = self.createGan()

    def initHyperParameters(self, params):
        self.convFilters = [int(x) for x in params.convFilters.split(',')]
        self.convTransposeFilters = [int(x) for x in params.convTransposeFilters.split(',')]
        self.adamLearningRate = params.adamLearningRate
        self.adamBeta1 = params.adamBeta1
        self.kernelInitStdDev = params.kernelInitStdDev
        self.generatorInputFilters = params.generatorInputFilters
        self.leakyReluAlpha = params.leakyReluAlpha
        self.dropoutRate = params.dropoutRate
        self.convLayerKernelSize = (3,3)
        self.convTransposeLayerKernelSize = (4,4)
        self.generatorOutputLayerKernelSize = (7,7)

    def initMetricsVars(self):
        self.realBinaryLossHistory = list()
        self.realLabelsLossHistory = list()
        self.fakeBinaryLossHistory = list()
        self.fakeLabelsLossHistory = list()
        self.lossHistory = list()
        self.metricHistory = list()

    def createDiscriminator(self, inputShape, classes, batchNorm=True) -> Model:
        init = RandomNormal(stddev=self.kernelInitStdDev)
        imageInput = Input(shape=inputShape)
        convLayer = self.buildConvLayers(batchNorm, init, imageInput)

        flattenLayer = Flatten()(convLayer)
        binaryOutputLayer = Dense(1, activation='sigmoid')(flattenLayer)
        labelsOutputLayer = Dense(classes, activation='softmax')(flattenLayer)
        model = Model(imageInput, [binaryOutputLayer, labelsOutputLayer])

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        # TODO: check how to get accuracy metrics for multi-output models
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
        return model

    def createGenerator(self, latentDim, imageDim, labelDim, classes, batchNorm=True) -> Model:
        labelInputNodes = imageDim * imageDim
        init = RandomNormal(stddev=self.kernelInitStdDev)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(classes, labelDim)(labelInput)
        labelDense = Dense(labelInputNodes, kernel_initializer=init)(labelEmbedding)
        labelShaped = Reshape((imageDim, imageDim, 1))(labelDense)

        imageInputNodes = self.generatorInputFilters * imageDim * imageDim
        imageInput = Input(shape=(latentDim,))
        imageDense = Dense(imageInputNodes, kernel_initializer=init)(imageInput)
        imageActv = LeakyReLU(self.leakyReluAlpha)(imageDense)
        imageShaped = Reshape((imageDim, imageDim, self.generatorInputFilters))(imageActv)

        imageLabelConcat = Concatenate()([imageShaped, labelShaped])

        convLayer = self.buildConvTransposeLayers(batchNorm, init, imageLabelConcat)

        outputLayer = Conv2D(1, self.generatorOutputLayerKernelSize, padding='same', activation='tanh', kernel_initializer=init)(convLayer)
        model = Model([imageInput, labelInput], outputLayer)
        return model

    def createGan(self) -> Model:
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        ganOutput = self.discriminator(self.generator.output)
        model = Model(self.generator.input, ganOutput)

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def buildConvLayer(self, filters, batchNorm, kernelInit, inLayer):
        # downsample layers
        layer = Conv2D(filters, self.convLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        layer = Dropout(self.dropoutRate)(layer)

        # normal sample layers
        layer = Conv2D(filters, self.convLayerKernelSize, padding='same', kernel_initializer=kernelInit)(layer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        layer = Dropout(self.dropoutRate)(layer)

        return layer

    def buildConvTransposeLayer(self, filters, batchNorm, kernelInit, inLayer):
        layer = Conv2DTranspose(filters, self.convTransposeLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        outLayer = Dropout(self.dropoutRate)(layer)
        return outLayer

    def buildConvLayers(self, batchNorm, kernelInit, inLayer):
        layer = inLayer
        for f in self.convFilters:
            layer = self.buildConvLayer(f, batchNorm, kernelInit, layer)
        return layer

    def buildConvTransposeLayers(self, batchNorm, kernelInit, inLayer):
        layer = inLayer
        for f in self.convTransposeFilters:
            layer = self.buildConvTransposeLayer(f, batchNorm, kernelInit, layer)
        return layer

    def train(self, epochs, batchSize, evalFreq):
        if not os.path.exists(self.evalDirectoryName):
            os.makedirs(self.evalDirectoryName)

        batchesPerEpoch = int(self.data.getDatasetShape() / batchSize)
        halfBatch = int(batchSize / 2)

        self.plotStartingImageSamples()

        self.startTime = time.time()

        for i in range(epochs):
            for j in range(batchesPerEpoch):
                [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(halfBatch)
                _, dRealLossBinary, dRealLossLabels, _, _ = self.discriminator.train_on_batch(xReal, [yReal, xRealLabel])

                [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, halfBatch)
                _, dFakeLossBinary, dFakeLossLabels, _, _ = self.discriminator.train_on_batch(xFake, [yFake, xFakeLabel])

                [xGan, xGanLabel], yGan = self.data.generateFakeTrainingGanSamples(self.latentDim, batchSize)
                _, gLossBinary, gLossLabels = self.gan.train_on_batch([xGan, xGanLabel], [yGan, xGanLabel])

                self.realBinaryLossHistory.append(dRealLossBinary)
                self.realLabelsLossHistory.append(dRealLossLabels)
                self.fakeBinaryLossHistory.append(dFakeLossBinary)
                self.fakeLabelsLossHistory.append(dFakeLossLabels)
                self.lossHistory.append(gLossBinary)

                metrics = ('> %d, %d/%d, dRealLossBinary=%.3f, dFakeLossBinary=%.3f, gLossBinary=%.3f' %
                    (i + 1, j, batchesPerEpoch, dRealLossBinary, dFakeLossBinary, gLossBinary))
                self.metricHistory.append(metrics)
                print(metrics)

            if (i + 1) % evalFreq == 0:
                self.evaluate(i + 1)

    def evaluate(self, epoch, samples=150):
        [xReal, xRealLabel], yReal = self.data.generateRealTrainingSamples(samples)
        # TODO: unpack results properly (i.e. figure out which output values are which)
        _, dRealAcc, _, _, _ = self.discriminator.evaluate(xReal, [yReal, xRealLabel])

        [xFake, xFakeLabel], yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc, _, _, _ = self.discriminator.evaluate(xFake, [yFake, xFakeLabel])

        accuracyMetrics = '> %d, accuracy real: %.0f%%, fake: %.0f%%' % (epoch, dRealAcc * 100, dFakeAcc * 100)
        self.metricHistory.append(accuracyMetrics)
        print(accuracyMetrics)

        elaspedTime = f'> {epoch}, elapsed time: {self.getElapsedTime()}'
        self.metricHistory.append(elaspedTime)
        print(elaspedTime)

        modelFilename = '%s/generated_model_e%03d.h5' % (self.evalDirectoryName, epoch)
        self.generator.save(modelFilename)

        metricsFilename = '%s/metrics_e%03d.txt' % (self.evalDirectoryName, epoch)
        with open(metricsFilename, 'w') as fd:
            for i in self.metricHistory:
                fd.write(i + '\n')
            self.metricHistory.clear()

        outputPath = f'{self.evalDirectoryName}/generated_plot_e{epoch}_random.png'
        self.plotImageSamples([xFake, xFakeLabel], outputPath)

        xFakeOrdered, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples, False)
        outputPath = f'{self.evalDirectoryName}/generated_plot_e{epoch}_ordered.png'
        self.plotImageSamples(xFakeOrdered, outputPath)

        self.plotHistory(epoch)

    def plotImageSamples(self, samples, outputPath, n=10):
        images, _ = samples
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        pyplot.savefig(outputPath)
        pyplot.close()

    def plotStartingImageSamples(self, samples=150):
        xReal, _ = self.data.generateRealTrainingSamples(samples)
        self.plotImageSamples(xReal, f'{self.evalDirectoryName}/target_plot.png')

        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        self.plotImageSamples(xFake, f'{self.evalDirectoryName}/generated_plot_e0.png')

    def plotHistory(self, epoch):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.realBinaryLossHistory, label='dRealLoss')
        pyplot.plot(self.fakeBinaryLossHistory, label='dFakeLoss')
        pyplot.plot(self.lossHistory, label='gLoss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.realLabelsLossHistory, label='accReal')
        pyplot.plot(self.fakeLabelsLossHistory, label='accFake')
        pyplot.legend()

        pyplot.savefig('%s/loss_acc_history_e%03d.png' % (self.evalDirectoryName, epoch))
        pyplot.close()

    def getElapsedTime(self):
        elapsedTime = time.time() - self.startTime
        return str(datetime.timedelta(seconds=elapsedTime))

    def summary(self):
        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()

class Inference:
    def __init__(self, data:Data, modelPath:str, latentDim:int) -> None:
        self.data = data
        self.modelPath = modelPath
        self.latentDim = latentDim
        self.samples = 100
        self.evalDirectoryName = 'eval'

    def run(self):
        model = load_model(self.modelPath)
        input = self.data.generateLatentPointsAndOrderedLabels(self.latentDim, self.samples)
        output = model.predict(input)
        self.plotImageSamples(output)

    def plotImageSamples(self, images, n=10):
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        filename = f'{self.evalDirectoryName}/generated_samples.png'
        pyplot.savefig(filename)
        pyplot.close()