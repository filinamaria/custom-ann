package weka.customClassifier.multiLayerPerceptron;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;

import org.jblas.DoubleMatrix;

import random.RandomGen;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

public class MultiLayerPerceptron extends Classifier{
	private static final double bias = 1.0; // bias unit
	
	private double learningRate; // learning rate for weight update
	private double momentum; 
	private double mseThreshold; // MSE threshold
	private int maxIteration; // maximum number of epoch
	
	private int hiddenLayer;
	private int[] numNodes;
	
	private boolean randomWeight;
	private double initialWeight;
	
	private long randomSeed; // seed used for random number generator
	
	private List<DoubleMatrix> weights; //weight[0][x][y] adalah weight input node ke-x to hidden layer berikutnya ke node-y
	private List<DoubleMatrix> lastDeltaWeight;
	private List<DoubleMatrix> layers; //layers[0] -> input, layers[terakhir] -> output layer
	
	private Attribute classAttribute;
	private NominalToBinary nominalToBinaryFilter;
	
	private Instances dataSet;
	
	
	/**
	 * Default constructor
	 */
	public MultiLayerPerceptron() {
		this.weights = new ArrayList<DoubleMatrix>();
		this.layers = new ArrayList<DoubleMatrix>();
		this.lastDeltaWeight = new ArrayList<DoubleMatrix>();
	}
	
	/**
	 * User-defined constructor with random initial weights
	 * @param learningRate
	 * @param threshold
	 * @param maxIteration
	 * @param momentum
	 * @param hiddenLayer
	 * @param numNodes
	 */
	public MultiLayerPerceptron(double learningRate, double mseThreshold, int maxIteration, double momentum, int[] numNodes){
		this.weights = new ArrayList<DoubleMatrix>();
		this.layers = new ArrayList<DoubleMatrix>();
		this.lastDeltaWeight = new ArrayList<DoubleMatrix>();
		this.learningRate = learningRate;
		this.mseThreshold = mseThreshold;
		this.maxIteration = maxIteration;
		this.momentum = momentum;
		this.hiddenLayer = numNodes.length;
		this.numNodes = numNodes;
		this.randomWeight = true;
	}
	
	/**
	 * User-defined constructor with user-defined initial weights
	 * @param learningRate
	 * @param threshold
	 * @param maxIteration
	 * @param momentum
	 * @param hiddenLayer
	 * @param numNodes
	 * @param initialWeight
	 */
	public MultiLayerPerceptron(double learningRate, double mseThreshold, int maxIteration, double momentum, int[] numNodes, double initialWeight){
		this.weights = new ArrayList<DoubleMatrix>();
		this.lastDeltaWeight = new ArrayList<DoubleMatrix>();
		this.layers = new ArrayList<DoubleMatrix>();
		this.learningRate = learningRate;
		this.mseThreshold = mseThreshold;
		this.maxIteration = maxIteration;
		this.momentum = momentum;
		this.hiddenLayer = numNodes.length;
		this.numNodes = numNodes;
		this.randomWeight = false;
		this.initialWeight = initialWeight;
	}
	
	/**
	 * 
	 * @param learningRate
	 */
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	
	/**
	 * 
	 * @param threshold
	 */
	public void setMSEThreshold(double mseThreshold) {
		this.mseThreshold = mseThreshold;
	}
	
	/**
	 * 
	 * @param maxIteration
	 */
	public void setMomentum(int momentum) {
		this.momentum = momentum;
	}
	
	/**
	 * 
	 * @param maxIteration
	 */
	public void setMaxIteration(int maxIteration) {
		this.maxIteration = maxIteration;
	}
	
	/**
	 * Set seed for random number generator
	 * @param seed
	 */
	public void setSeed(long seed) {
		if (seed >= 0)
			randomSeed = seed;
	}
	
	/**
	 * @return capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.BINARY_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
    
		return result;
	}
	
	/**
	 * Sigmoid activation function
	 * @param x
	 * @return
	 */
	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.pow(Math.E, -1.0 * x));
	}
	
	/**
	 * Randomize vector elements with uniform distribution
	 * @param weightVec Vector with it's elements randomly initialized
	 */
	private void randomizeWeight(DoubleMatrix weightVec) {
		RandomGen rand = new RandomGen(randomSeed);
		
		for(int i = 0; i < weightVec.length; i++){
			weightVec.put(i, rand.uniform());
		}
	}

	/**
	 * Check whether training data contains nominal attributes
	 * @param data training data
	 * @return true if training data contains nominal attributes
	 */
	private boolean nominalData(Instances data) {
		boolean found = false;
		
		Enumeration attributes = data.enumerateAttributes();
		while(attributes.hasMoreElements() && !found){
			Attribute attribute = (Attribute) attributes.nextElement();
			if(attribute.isNominal())
				found = true;
		}
		
		return found;
	}
	
	/**
	 * Convert nominal attribute to binary numeric attribute
	 * @param data
	 * @return instances with numeric attributes
	 * @throws Exception 
	 */
	public Instances nominalToNumeric(Instances data) throws Exception {
		this.nominalToBinaryFilter = new NominalToBinary();
		this.nominalToBinaryFilter.setInputFormat(data);
		
		data = Filter.useFilter(data, this.nominalToBinaryFilter);
		
		return data;
	}
	
	/**
	 * Call this function to build and train a neural network for the training data provided.
	 * @param data the training data
	 */
	public void buildClassifier(Instances data) throws Exception {
		this.dataSet = data;
		
		// test whether classifier can handle the data
		getCapabilities().testWithFail(data);
		
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
				
		// remove instances with missing values
		Enumeration attributes = data.enumerateAttributes();
		while(attributes.hasMoreElements()){
			Attribute attribute = (Attribute) attributes.nextElement();
			data.deleteWithMissing(attribute);
		}
		
		// check if data contains nominal attributes
		if(nominalData(data))
			data = nominalToNumeric(data);
		
		this.classAttribute = data.classAttribute();
		
		Enumeration instancess = data.enumerateInstances();
		while(instancess.hasMoreElements()){
			Instance instance = (Instance) instancess.nextElement();
			for(int i = 0; i < instance.numAttributes(); i++){
				if(instance.attribute(i).isNominal())
					System.out.print(instance.stringValue(i) + " ");
				else
					System.out.print(instance.value(i) + " ");
			}
			
			System.out.println();
		}
		
		// Initialize the topology
		// input node
		DoubleMatrix dm = new DoubleMatrix(data.numAttributes(), 1); //+1 untuk bias, DoubleMatrix[0] adalah bias
		layers.add(dm);
		
		// hidden node
		for (int i=0; i<this.hiddenLayer; i++) {
			DoubleMatrix ff = new DoubleMatrix(this.numNodes[i]+1, 1); //+1 untuk bias, DoubleMatrix[0] adalah bias
			layers.add(ff);
		}
		
		// output node
		DoubleMatrix aa = new DoubleMatrix(data.numClasses(), 1); //+1 untuk bias, DoubleMatrix[0] adalah bias
		layers.add(aa);
		
		
		// Initialize weight matrix
		// input to hidden
		DoubleMatrix bb = new DoubleMatrix(layers.get(0).rows, layers.get(1).rows);
		DoubleMatrix cc = new DoubleMatrix(layers.get(0).rows, layers.get(1).rows);
		weights.add(bb);
		lastDeltaWeight.add(cc);
		
		// hidden to hidden
		for (int i=1; i<this.hiddenLayer; i++) {
			DoubleMatrix dd = new DoubleMatrix(layers.get(i).rows, layers.get(i+1).rows);
			DoubleMatrix ee = new DoubleMatrix(layers.get(i).rows, layers.get(i+1).rows);
			weights.add(dd);
			lastDeltaWeight.add(ee);
		}
		
		// hidden layer to output layer
		DoubleMatrix gg = new DoubleMatrix(layers.get(this.hiddenLayer).rows, layers.get(this.hiddenLayer+1).rows);
		DoubleMatrix hh = new DoubleMatrix(layers.get(this.hiddenLayer).rows, layers.get(this.hiddenLayer+1).rows);
		weights.add(gg);
		lastDeltaWeight.add(hh);
		
		setSeed(System.currentTimeMillis());
		for (int i=0; i<weights.size(); i++) {
			if(this.randomWeight){
				randomizeWeight(weights.get(i));
			}else{
				for (int j=0; j<weights.get(i).length; j++) {
					weights.get(i).put(j, this.initialWeight);
				}
			}
		}
		
		// learning
		int epoch = 0;
		double MSE = Double.POSITIVE_INFINITY;
		while (epoch < this.maxIteration && MSE>=this.mseThreshold) {
			Enumeration instances = data.enumerateInstances();
			while(instances.hasMoreElements()) {
				Instance instance = (Instance) instances.nextElement();
				double[] desiredOutput;
				if (instance.classAttribute().isNominal()) {
					desiredOutput = new double[this.classAttribute.numValues()]; //udah 0.0
					desiredOutput[(int) instance.classValue()] = 1.0;
				}
				else {
					desiredOutput = new double[1]; //pasti 1
					desiredOutput[0] = instance.classValue();
				}
				
				//masukin input
				//bias
				layers.get(0).put(0, bias);
				for (int i=0; i<instance.numAttributes()-1; i++) {
					layers.get(0).put(i+1, instance.value(i));
				}
				//feedfoward & backprop
				feedFoward();
				backPropagation(desiredOutput);
			}
			MSE = calculateMSE(data);
			System.out.println("Epoch "+epoch+" MSE="+MSE);
			System.out.println(this);
			epoch++;
		}
	}
	
	/**
	 * Calculate sigma of weights
	 * @param currentLayer
	 * @param nodeNum
	 */
	public void calculateSummation(int currentLayer, int nodeNum) {
		double sum = 0.0;
		for (int i=0; i<layers.get(currentLayer-1).length; i++) {
			sum += weights.get(currentLayer-1).get(i, nodeNum) * layers.get(currentLayer-1).get(i);
		}
		if (nodeNum!=0 || currentLayer==layers.size()-1)
			layers.get(currentLayer).put(nodeNum, sigmoid(sum));
		else
			layers.get(currentLayer).put(nodeNum, bias);
	}
	
	/**
	 * As the name stated
	 */
	public void feedFoward() {
		int currentLayer = 0;
		while (currentLayer < layers.size()-1) {
			for (int i=0; i<layers.get(currentLayer+1).length; i++) {
				calculateSummation(currentLayer+1, i);
			}
			currentLayer++;
		}
	}
	
	/**
	 * As the name stated
	 * @param data
	 * @return MSE
	 */
	public double calculateMSE(Instances data) {
		int numAttr = this.classAttribute.numValues();
		double sumErr = 0.0;
		Enumeration instances = data.enumerateInstances();
		while(instances.hasMoreElements()){
			Instance instance = (Instance) instances.nextElement();
			if (instance.classAttribute().isNominal()) {
				double[] dm = new double[numAttr]; //udah 0.0
				dm[(int) instance.classValue()] = 1.0;
				double tempErr = 0.0;
				for (int i=0; i<layers.get(this.hiddenLayer+1).length; i++) {
					tempErr += Math.pow(dm[i]-layers.get(this.hiddenLayer+1).get(i), 2);
				}
				tempErr/=layers.get(this.hiddenLayer+1).length;
				sumErr +=tempErr;
			}
			else {
				sumErr += Math.pow((instance.classValue()-layers.get(this.hiddenLayer+1).get(0)), 2); 
			}
		}
		return (sumErr/2.0);
	}
	
	/**
	 * As the name stated
	 * @param lastDeltaWeight
	 * @param desiredOutput
	 */
	public void backPropagation(double[] desiredOutput) {
		List<List<Double>> xV = new ArrayList();
		for (int i=0; i<layers.size(); i++) {
			List<Double> temp = new ArrayList();
			xV.add(temp);
		}
		
		//output layer to hidden layer
		for (int i=0; i<layers.get(this.hiddenLayer+1).length; i++) {
			double x = layers.get(this.hiddenLayer+1).get(i) - (1-layers.get(this.hiddenLayer+1).get(i)) * (desiredOutput[i]-layers.get(this.hiddenLayer+1).get(i));
			xV.get(this.hiddenLayer+1).add(x);
			for (int j=0; j<layers.get(this.hiddenLayer).length; j++) {
				double deltaWeight = 0.0;
				deltaWeight += (learningRate*xV.get(this.hiddenLayer+1).get(i)*layers.get(this.hiddenLayer).get(j)) + (momentum*lastDeltaWeight.get(this.hiddenLayer).get(j,i));
				//System.out.println("deltaWeight = "+deltaWeight);
				lastDeltaWeight.get(this.hiddenLayer).put(j,i, deltaWeight);
				weights.get(this.hiddenLayer).put(j, i, weights.get(this.hiddenLayer).get(j, i) + deltaWeight);
			}
		}
		
		//hidden layer to input layer
		for (int currentLayer=this.hiddenLayer; currentLayer>0; currentLayer--) {
			for (int i=0; i<layers.get(currentLayer).length; i++) {
				double sum = 0.0;
				for (int k=0; k<layers.get(currentLayer+1).length; k++) {
					sum += xV.get(currentLayer+1).get(k) * weights.get(currentLayer).get(i,k);
				}
				double x = layers.get(currentLayer).get(i) * (1-layers.get(currentLayer).get(i)) * sum;
				xV.get(currentLayer).add(x);
				for (int j=0; j<layers.get(currentLayer-1).length; j++) {
					double deltaWeight = 0.0;
					/*System.out.println("cibai "+xV.get(currentLayer).get(i));
					System.out.println("cibai 2 "+layers.get(currentLayer-1).get(j));*/
					deltaWeight += (learningRate*xV.get(currentLayer).get(i)*layers.get(currentLayer-1).get(j)) + (momentum*lastDeltaWeight.get(currentLayer-1).get(j,i));
					lastDeltaWeight.get(currentLayer-1).put(j,i, deltaWeight);
					weights.get(currentLayer-1).put(j, i, weights.get(currentLayer-1).get(j,i)+deltaWeight);
				}
			}
		}
	}
	
	/**
	 * @return If classAttribute is numeric, output is the output of perceptron. If nominal, index of largest output layer
	 */
	public double classifyInstance(Instance instance) throws Exception {
		Instances instances = new Instances(dataSet);
		instances.delete();
		instances.add(instance);
		
		if(this.nominalData(instances))
			instances = this.nominalToNumeric(instances);
		
		// masukin input
		// bias
		layers.get(0).put(0, bias);
		for (int i=0; i<instances.firstInstance().numAttributes()-1; i++) {
			layers.get(0).put(i+1, instances.firstInstance().value(i));
		}
		
		feedFoward();
		
		double output = 0.0;
		
		if(this.classAttribute.isNominal()){
			int maxIndex = 0;
			double max = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < layers.get(layers.size() - 1).length; i++) {
			    if (layers.get(layers.size() - 1).get(i) > max) {
			        max = layers.get(layers.size() - 1).get(i);
			        maxIndex = i;
			    }
			}
			//System.out.println(instances.firstInstance());
			//System.out.println(layers.get(layers.size()-1));
			
			return (double) maxIndex;
		}else if(this.classAttribute.isNumeric()){
			output = layers.get(layers.size() - 1).get(0);
		}
 		
		return output;
	}
	
	/**
	 * Evaluasi
	 * @param data
	 * @throws Exception 
	 */
	public void evaluate(Instances data) throws Exception {
		if (this.classAttribute.isNominal()) {
			double correctlyClassifiedInstances = 0.0;
			Enumeration instances = data.enumerateInstances();
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
				double retVal = classifyInstance(instance);
				if (Double.compare(instance.classValue(),retVal)==0) {
					correctlyClassifiedInstances+=1.0;
				}
			}
			System.out.println("Accuracy = "+correctlyClassifiedInstances/(double)data.numInstances());
		}
		else { //numeric
			double deltaError = 0.0;
			Enumeration instances = data.enumerateInstances();
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
				double retVal = classifyInstance(instance);
				deltaError += Math.pow(retVal-instance.classValue(), 2);
			}
			System.out.println("MSE = "+deltaError/2.0);
		}
	}
	
	/**
	 * @return string describing the model
	 */
	public String toString() {
		StringBuffer output = new StringBuffer();
		
		for(int i = 0; i < weights.size(); i++){
			output.append(weights.get(i) + "\n");
		}
		
		return output.toString().replace(';', '|');
	}
	
	/**
	 * Load Arff data file. For testing purpose only
	 * @param filePath
	 * @return
	 * @throws IOException
	 */
	public static Instances loadDatasetArff(String filePath) throws IOException { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }
	
	/**
	 * Main program. For testing purpose only
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		String dataset = "example/weather.nominal.arff";
		
		Instances data = loadDatasetArff(dataset);
		data.setClassIndex(data.numAttributes() - 1);
		
		int[] numNodes = new int[]{2};
		
		MultiLayerPerceptron mlp = new MultiLayerPerceptron(0.01, 0.1, 100000, 0.1, numNodes, 0.0);
		mlp.buildClassifier(data);
		
		//System.out.println(mlp);
		
		Instance instance = data.instance(0);
		System.out.println(instance);
		System.out.println(data.classAttribute().value((int) mlp.classifyInstance(instance)));
		mlp.evaluate(data);
	}
}
