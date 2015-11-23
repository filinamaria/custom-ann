package weka.customClassifier.MLP_WIRA;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.core.matrix.Maths;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;

/**
 * Created by wiragotama on 11/23/15.
 */
public class MultilayerPerceptron extends Classifier {
    private double learningRate; // learning rate for weight update
    private double momentum;
    private double mseThreshold; // MSE threshold
    private int maxIteration; // maximum number of epoch
    private boolean isRandomInitialWeight;
    private double initialWeight;

    private int[] neuronPerLayer; //tidak ada input layer ya
    private int[] neuronPerHiddenLayer;
    private Neuron network[][]; //network[layer][node]

    private Attribute classAttribute;
    private Instances dataSet;

    private NominalToBinary nominalToBinaryFilter;
    private Normalize normalizeFilter;
    private int nInputFeatures;
    private boolean normalizeAttribute;

    /**
     * Output configuration to stdout
     */
    public void printConfiguration() {
        System.out.println("MLP Configuration");
        System.out.println("learning rate       = "+learningRate);
        System.out.println("momentum            = "+momentum);
        System.out.println("mseThreshold        = "+mseThreshold);
        System.out.println("max iteration       = "+maxIteration);
        System.out.println("random weight       = "+isRandomInitialWeight);
        System.out.println("initial weight      = "+initialWeight);
        System.out.println("normalize attribute = "+normalizeAttribute);
        System.out.println("n hidden layer      = "+neuronPerHiddenLayer.length);
        StringBuffer str = new StringBuffer("");
        for (int i=0; i<neuronPerHiddenLayer.length; i++)
            str.append(neuronPerHiddenLayer[i]+" ");
        System.out.println("neuron hidden layer = "+str);
        System.out.println("------------------\n\n");
    }

    /**
     * Default Constructor, using default parameter
     */
    public MultilayerPerceptron() {
        this.learningRate = 0.1;
        this.momentum = 0.0;
        this.mseThreshold = 0.02;
        this.maxIteration = 10;
        this.neuronPerLayer = null;
        this.neuronPerHiddenLayer = null;
        this.isRandomInitialWeight = true;
        this.network = null;
        this.dataSet = null;
        this.classAttribute = null;
        this.nominalToBinaryFilter = null;
        this.normalizeFilter = null;
        this.normalizeAttribute = true;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    /**
     * Build MLP Classifier
     * @param data
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.dataSet = data;

        // test whether classifier can handle the data
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        //change to numeric
        nominalToBinaryFilter = new NominalToBinary();
        nominalToBinaryFilter.setInputFormat(data);
        data = Filter.useFilter(data, nominalToBinaryFilter);

        //normallize data
        if (this.normalizeAttribute) {
            normalizeFilter = new Normalize();
            normalizeFilter.setInputFormat(data);
            data = Filter.useFilter(data, normalizeFilter);
        }

        this.nInputFeatures = data.numAttributes()-1; //jumlah fitur input

        //build network
        this.neuronPerLayer = new int[this.neuronPerHiddenLayer.length+1];
        for (int i=0; i<neuronPerLayer.length-1; i++) {
            neuronPerLayer[i] = neuronPerHiddenLayer[i];
        }
        neuronPerLayer[neuronPerLayer.length-1] = data.classAttribute().numValues(); //output layer
        //hidden layer
        this.network = new Neuron[this.neuronPerLayer.length][];
        for (int i=0; i<this.neuronPerLayer.length; i++) {
            this.network[i] = new Neuron[neuronPerLayer[i]];
            if (i==this.neuronPerLayer.length-1) {
                for (int j = 0; j < neuronPerLayer[i]; j++)
                    if (data.classAttribute().isNumeric())
                        this.network[i][j] = new Neuron(Neuron.ActivationFunction.LINEAR);
                    else
                        this.network[i][j] = new Neuron(Neuron.ActivationFunction.SIGMOID);
            }
            else {
                for (int j = 0; j < neuronPerLayer[i]; j++)
                    this.network[i][j] = new Neuron(Neuron.ActivationFunction.SIGMOID);
            }
        }

        this.classAttribute = data.classAttribute();
        //initialize weight
        initializeWeight();

        //learning
        int epoch = 0;
        double mse = Double.POSITIVE_INFINITY;
        while (epoch < this.maxIteration && mse>=this.mseThreshold) {
            Enumeration instances = data.enumerateInstances();
            int dataCount = 0;
            mse = 0.0;
            while(instances.hasMoreElements()) {
                Instance instance = (Instance) instances.nextElement();
                double[] targetOutputs;
                if (instance.classAttribute().isNominal()) {
                    targetOutputs = new double[this.classAttribute.numValues()]; //udah 0.0
                    targetOutputs[(int) instance.classValue()] = 1.0;
                }
                else {
                    targetOutputs = new double[1]; //pasti 1
                    targetOutputs[0] = instance.classValue();
                }

                //prepare inputs
                double[] inputs = new double[instance.numAttributes()-1];
                for (int i=0; i<inputs.length; i++) {
                    inputs[i] = instance.value(i);
                }

                //feedfoward & backprop
                double outputs[] = feedFoward(inputs);
                mse += this.mseCalculationOutputLayer(outputs, targetOutputs);
                backProp(outputs, targetOutputs, inputs);
                dataCount++;
            }
            mse /= (dataCount*network[network.length-1].length); //correct MSE calculation
            System.out.println("Epoch "+epoch+" MSE="+mse);
            epoch++;
        }
    }

    /**
     * Feed network foward
     * @param inputs
     * @return neural network's outputs
     */
    private double[] feedFoward(double inputs[]) {
        double[] outputs = null;
        for (int i=0; i<network.length; i++) {
            //for each neuron in each layer
            double[] outputResult = new double[network[i].length];
            for (int j=0; j<network[i].length; j++) {
                outputResult[j] = network[i][j].output(inputs);
            }
            inputs = outputResult.clone(); //output first layer become input for the next layer
        }
        outputs = inputs.clone();
        return outputs;
    }

    /**
     * Back propagation algorithm
     * @param outputs outputs of network
     * @param targets training target outputs
     * @param inputs inputs to the network
     */
    private void backProp(double outputs[], double[] targets, double[] inputs) {
        //error for each neurons
        double[][] error = new double [network.length][];
        for (int i=0; i<network.length; i++) {
            error[i] = new double[network[i].length];
        }

        //output layer
        for (int neuron=0; neuron < network[network.length-1].length; neuron++) {
            error[network.length-1][neuron] = outputs[neuron]*(1-outputs[neuron])*(targets[neuron]-outputs[neuron]);
            double[] lastWeigts = network[network.length-1][neuron].getLastWeights().clone();
            double[] weightsToNeuron = network[network.length-1][neuron].getWeights().clone(); //previous weight
            double[] newWeight = new double[network[network.length-1][neuron].getWeights().length];

            //update each weight to itself, also update bias's weight
            newWeight[0] = weightsToNeuron[0] + (learningRate * error[network.length-1][neuron] * 1.0) //bias
                        + (momentum*(lastWeigts[0]-weightsToNeuron[0]));

            for (int j=1; j<weightsToNeuron.length; j++) {
                newWeight[j] = weightsToNeuron[j] + learningRate * error[network.length-1][neuron] * network[network.length-2][j-1].getLastOutput()
                            + (momentum*(weightsToNeuron[j]-lastWeigts[j]));
                //W[j,i] = (learning rate * error[i] * x[j]) + (momentum * lastDeltaWeight); x[j] is either output from previous node or input
                error[network.length-2][j-1] += (weightsToNeuron[j] * error[network.length-1][neuron]); //to make life easier
            }
            network[network.length-1][neuron].setWeights(newWeight); //update the weight
        }

        //hidden layer to (hidden layer or input layer)
        for (int layer=this.neuronPerHiddenLayer.length-1; layer>=0; layer--){
            for (int neuron=0; neuron < network[layer].length; neuron++) {
                double[] lastWeigts = network[layer][neuron].getLastWeights().clone();
                double[] weightsToNeuron = network[layer][neuron].getWeights().clone(); //previous weight
                double[] newWeight = new double[network[layer][neuron].getWeights().length];

                /*error in here is the sum of error from this node to next layer
                already computed above to make life easier, this is just precaution, don't discard this section...
                double sumError = 0.0;
                for (int k=0; k<network[layer+1].length; k++) {
                    sumError += (error[layer+1][k] * network[layer+1][k].getLastWeights()[neuron]); //ambil getLastWeight karna weight yang baru sudah diupdate
                    //error[layer+1][k] * W[j,k]
                }*/
                error[layer][neuron] = network[layer][neuron].getLastOutput() * (1-network[layer][neuron].getLastOutput()) * error[layer][neuron];

                //output from previous layer (in backprop's case is the next layer from this layer in the nextwork)
                double[] prevLayerOutputs;
                if (layer-1 < 0) {
                    prevLayerOutputs = inputs.clone();
                }
                else {
                    prevLayerOutputs = new double[network[layer - 1].length];
                    for (int j=0; j<prevLayerOutputs.length; j++) {
                        prevLayerOutputs[j] = network[layer-1][j].getLastOutput();
                    }
                }

                //update each weight to itself, also update bias's weight
                newWeight[0] = weightsToNeuron[0] + (learningRate * error[layer][neuron] * 1.0) //bias
                        + (momentum*(lastWeigts[0]-weightsToNeuron[0]));
                for (int j=1; j<weightsToNeuron.length; j++) {
                    newWeight[j] = weightsToNeuron[j] + learningRate * error[layer][neuron] * prevLayerOutputs[j-1]
                            + (momentum*(weightsToNeuron[j]-lastWeigts[j]));
                    //W[j,i]           = (learning rate * error[i] * x[j]) + (momentum * lastDeltaWeight); x[j] is either output from previous node or input
                    if (layer-1 >0)
                        error[layer-1][j-1] += (weightsToNeuron[j]*error[layer][neuron]);
                }
                network[layer][neuron].setWeights(newWeight); //update the weight
            }
        }
    }

    /**
     * Initialize neural network's weight
     */
    private void initializeWeight() {
        for (int i=0; i<network.length; i++) {
            for (int j=0; j<network[i].length; j++) {
                int length;
                if (i == 0)
                    length = nInputFeatures+1;
                else length = network[i-1].length+1; //+1 for bias

                if (isRandomInitialWeight)
                    network[i][j].setWeights(generateRandomWeight(length));
                else
                    network[i][j].setWeights(generateInitialWeight(length));
            }
        }
    }

    /**
     * Generate array of random weight
     * @param length of array
     * @return array of weight
     */
    private double[] generateRandomWeight(int length) {
        double[] weights = new double[length];
        Random random = new Random();
        for (int i = 0; i < length; i++) {
            weights[i] = random.nextDouble();
        }
        return weights;
    }

    /**
     * Generate array of initial weight
     * @param length of array
     * @return array of weight
     */
    private double[] generateInitialWeight(int length) {
        double[] weights = new double[length];
        for (int i = 0; i < length; i++) {
            weights[i] = this.initialWeight;
        }
        return weights;
    }

    /**
     * For classification
     * @param instance
     * @return probability for each class
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MultiLayerPerceptron: cannot handle missing value");
        }
        nominalToBinaryFilter.input(instance);
        Instance predict = nominalToBinaryFilter.output();
        if (this.normalizeAttribute) {
            normalizeFilter.input(predict);
            predict = normalizeFilter.output();
        }
        //masukin input
        double[] inputs = new double[predict.numAttributes()-1];
        for (int i=0; i<inputs.length; i++) {
            inputs[i] = predict.value(i);
        }
        double[] outputs = feedFoward(inputs);
        return outputs;
    }

    /**
     * mse calculation for output layer
     * @param output of output layer
     * @param target desired output of training
     * @return
     */
    private double mseCalculationOutputLayer(double[] output, double[] target) {
        double sum = 0.0;
        for (int i=0; i<target.length; i++) {
            sum += Maths.square(target[i] - output[i]);
        }
        sum /= 2.0;
        return sum;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getMomentum() {
        return momentum;
    }

    public double getMseThreshold() {
        return mseThreshold;
    }

    public double getMaxIteration() {
        return maxIteration;
    }

    public boolean isRandomIntialWeight() {
        return isRandomInitialWeight;
    }

    public double getInitialWeight() {
        return initialWeight;
    }

    public int[] getNeuronPerLayer() {
        return neuronPerLayer;
    }

    public int[] getNeuronPerHiddenLayer() {
        return neuronPerHiddenLayer;
    }

    public boolean getUsingNormalizeAttribute() {
        return this.normalizeAttribute;
    }

    public String toString() {
        StringBuffer str = new StringBuffer("");
        str.append("\nNEURAL NETWORK TOPOLOGY \n");
        str.append("-----------------------------\n");
        for (int i=0; i<neuronPerHiddenLayer.length; i++) {
            str.append("Hidden Layer "+i+"\n");
            str.append("-----------------\n");
            for (int j=0; j<neuronPerHiddenLayer[i]; j++) {
                str.append("Node "+j);
                str.append("\t"+network[i][j].toString());
                str.append("\n");
            }
            str.append("-----------------\n");
        }

        str.append("Output Layer \n");
        str.append("-----------------\n");
        for (int j=0; j<network[network.length-1].length; j++) {
                str.append(network[network.length-1][j].toString());
            str.append("\n");
        }
        str.append("-----------------\n\n");

        return str.toString();
    }

    public void setLearningRate(double _learningRate) {
        learningRate = _learningRate;
    }

    public void setMomentum(double _momentum) {
        momentum = _momentum;
    }

    public void setMseThreshold(double _mseThreshold) {
        mseThreshold = _mseThreshold;
    }

    public void setMaxIteration(int _maxIter) {
        maxIteration = _maxIter;
    }

    public void setRandomIntialWeight(boolean _isRandomInitialWeight) {
        isRandomInitialWeight = _isRandomInitialWeight;
    }

    public void setNormalizeAttribute(boolean _normalizeAttribute) {
        this.normalizeAttribute = _normalizeAttribute;
    }

    public void setInitialWeight(double _initialWeight) {
        initialWeight = _initialWeight;
    }

    public void setNeuronPerHiddenLayer(int[] _neuronPerHiddenLayers) {
        neuronPerHiddenLayer = new int[_neuronPerHiddenLayers.length];
        for (int i=0; i<_neuronPerHiddenLayers.length; i++)
            neuronPerHiddenLayer[i] = _neuronPerHiddenLayers[i];
    }

    public static Instances loadDatasetArff(String filePath) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        return loader.getDataSet();
    }

    public static void main(String[] args) throws Exception {
        String dataset = "example/weather.numeric.arff";

        Instances data = loadDatasetArff(dataset);
        data.setClassIndex(data.numAttributes() - 1);

        int[] neuronPerHiddenLayer = new int[]{3};
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setNeuronPerHiddenLayer(neuronPerHiddenLayer);
        mlp.setMaxIteration(100000);
        mlp.setNormalizeAttribute(true);
        /*mlp.setInitialWeight(0.0);
        mlp.setRandomIntialWeight(false);*/
        //mlp.printConfiguration();

        mlp.buildClassifier(data);
        Instance instance = data.instance(1);
        //System.out.println(data.classAttribute().value((int) mlp.classifyInstance(instance)));
        System.out.println(mlp.toString());
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(mlp, data);
        System.out.println(eval.toSummaryString());
    }
}
