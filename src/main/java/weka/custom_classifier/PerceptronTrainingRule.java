package weka.custom_classifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;

import org.jblas.DoubleMatrix;

import random.RandomGen;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;

public class PerceptronTrainingRule extends Classifier {
	private static final double bias = 1.0; // bias unit
	
	private DoubleMatrix weightVector; // vector for storing weights
	private DoubleMatrix deltaWeightVector; // vector for storing delta weights
	
	private double learningRate; // learning rate for weight update
	private double threshold; // MSE threshold
	private int maxIteration; // maximum number of epoch
	
	private boolean randomWeight;
	private DoubleMatrix initialWeight; // user-given initial weights
	
	private long randomSeed; // seed used for random number generator
	
	private StringBuffer output; // string buffer describing the model
	
	private NominalToBinary nominalToBinaryFilter; // filter to convert nominal attributes to binary numeric attributes
	
	/**
	 * Default constructor
	 */
	public PerceptronTrainingRule(){
		this.learningRate = 0.0;
		this.threshold = 0.0;
		this.maxIteration = 0;
		this.randomWeight = true;
		this.randomSeed = 0;
		output = new StringBuffer();
	}
	
	/**
	 * User-defined constructor with random initial weights
	 * @param learningRate
	 * @param threshold
	 * @param maxIteration
	 */
	public PerceptronTrainingRule(double learningRate, double threshold, int maxIteration){
		this.learningRate = learningRate;
		this.threshold = threshold;
		this.maxIteration = maxIteration;
		this.randomWeight = true;
		this.randomSeed = 0;
		output = new StringBuffer();
	}
	
	/**
	 * User-defined constructor with given initial weights
	 * @param learningRate
	 * @param threshold
	 * @param maxIteration
	 * @param initialWeight
	 */
	public PerceptronTrainingRule(double learningRate, double threshold, int maxIteration, DoubleMatrix initialWeight){
		this.learningRate = learningRate;
		this.threshold = threshold;
		this.maxIteration = maxIteration;
		this.randomWeight = false;
		this.initialWeight = initialWeight;
		this.randomSeed = 0;
		output = new StringBuffer();
	}
	
	/**
	 * 
	 * @param learningRate
	 */
	public void setLearningRate(double learningRate){
		this.learningRate = learningRate;
	}
	
	/**
	 * 
	 * @param threshold
	 */
	public void setThreshold(double threshold){
		this.threshold = threshold;
	}
	
	/**
	 * 
	 * @param maxIteration
	 */
	public void setMaxIteration(int maxIteration){
		this.maxIteration = maxIteration;
	}
	
	/**
	 * Set seed for random number generator
	 * @param seed
	 */
	public void setSeed(long seed){
		if (seed >= 0)
			randomSeed = seed;
	}
	
	/**
	 * Randomize vector elements with uniform distribution
	 * @param weightVec Vector with it's elements randomly initialized
	 */
	private void randomizeWeight(DoubleMatrix weightVec){
		RandomGen rand = new RandomGen(randomSeed);
		
		for(int i = 0; i < weightVec.length; i++){
			weightVec.put(i, rand.uniform());
		}
	}
	
	/**
	 * Get current settings of classifier
	 */
	public String[] getOptions(){
		String[] options = new String[4];
		
		StringBuffer learningRate = new StringBuffer("-LearningRate ");
		learningRate.append(this.learningRate);
		options[0] = learningRate.toString();
		
		StringBuffer threshold = new StringBuffer("-Threshold ");
		threshold.append(this.threshold);
		options[1] = threshold.toString();
		
		StringBuffer maxIteration = new StringBuffer("-MaxIteration ");
		maxIteration.append(this.maxIteration);
		options[2] = maxIteration.toString();
		
		StringBuffer randomWeight = new StringBuffer("-RandomWeight ");
		randomWeight.append(this.randomWeight);
		options[3] = randomWeight.toString();
		
		return options;
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
		result.enable(Capability.MISSING_CLASS_VALUES);
    
		return result;
	}

	/**
	 * Sign activation function
	 * @param x the floating-point value whose signum is to be returned
	 * @return
	 */
	private double sign(double x){
		if(Double.compare(x, 0.0) >= 0){
			return 1.0;
		}else{
			return -1.0;
		}
	}
	
	/**
	 * Compute sum of xi * wi
	 * @param instance
	 * @return sum of xi * wi in an instance
	 */
	private double sum(Instance instance){
		double sum = 0.0;
		
		for(int i = 0; i < instance.numAttributes(); i++){
			if(i == 0){
				sum += bias * weightVector.get(i);
			}else{
				sum += instance.value(i - 1) * weightVector.get(i);
			}
		}
		
		return sum;
	}
	
	/**
	 * Check whether training data contains nominal attributes
	 * @param data training data
	 * @return true if training data contains nominal attributes
	 */
	private boolean nominalData(Instances data){
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
	public Instances nominalToNumeric(Instances data) throws Exception{
		nominalToBinaryFilter = new NominalToBinary();
		nominalToBinaryFilter.setInputFormat(data);
		
		return Filter.useFilter(data, nominalToBinaryFilter);
	}
	
	/**
	 * Call this function to build and train a neural network for the training data provided.
	 * @param data the training data
	 */
	public void buildClassifier(Instances data) throws Exception {
		// test whether classifier can handle the data
		getCapabilities().testWithFail(data);
		
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		
		// check if data contains nominal attributes
		if(nominalData(data))
			data = nominalToNumeric(data);
		
		// create weight and delta weight vector
		this.weightVector = new DoubleMatrix(1, data.numAttributes());
		this.deltaWeightVector = new DoubleMatrix(1, data.numAttributes());
		
		// set initial weight either with random number or given initial weight
		if(this.randomWeight){
			randomizeWeight(weightVector);
		}else{
			if(this.initialWeight.rows != this.weightVector.rows || this.initialWeight.columns != this.weightVector.columns)
				throw new RuntimeException("initial weight size doesn't match with number of attributes");
			
			this.weightVector = this.initialWeight;
		}
		
		int epoch = 0;
		double meanSquaredError = Double.POSITIVE_INFINITY;
		
		// training iteration, finishes either when epoch reaches max iteration or MSE < threshold
		while(epoch < this.maxIteration && Double.compare(meanSquaredError, this.threshold) >= 0){
			
			Enumeration instances = data.enumerateInstances();
			
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
								
				double sum = 0.0;
				double output = 0.0;
				double error = 0.0;
				
				sum = this.sum(instance);				
				output = this.sign(sum);
				error = instance.value(instance.classAttribute()) - output;
				
				for(int i = 0; i < instance.numAttributes(); i++){
					if(i == 0){
						deltaWeightVector.put(i, this.learningRate * error * bias);
					}else{
						deltaWeightVector.put(i, this.learningRate * error * instance.value(i - 1));
					}
				}
				
				weightVector.addi(deltaWeightVector);
			}
			
			instances = data.enumerateInstances();
			
			double squaredError = 0.0;
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
				
				double sum = 0.0;
				double output = 0.0;
				double error = 0.0;
				
				sum = this.sum(instance);				
				output = this.sign(sum);
				error = instance.value(instance.classAttribute()) - output;
				
				squaredError += Math.pow(error, 2.0);
			}
			
			meanSquaredError = squaredError / 2.0;
			
			output.append("epoch " + epoch + ": " + weightVector + "\n");
			
			epoch++;
		}
	}
	
	/**
	 * @return string describing the model
	 */
	public String toString(){
		return this.output.toString();
	}
	
	
	/**
	 * Load Arff data file. For testing purpose only
	 * @param filePath
	 * @return
	 * @throws IOException
	 */
	public static Instances loadDatasetArff(String filePath) throws IOException
    { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }

	/**
	 * Main program. For testing purpose only
	 * @param args
	 * @throws Exception
	 */
	public static void main(String [] args) throws Exception{
		String dataset = "example/weather.numeric.arff";
		
		Instances data = loadDatasetArff(dataset);
		data.setClass(data.attribute(data.numAttributes() - 1));

		/*MultilayerPerceptron mlp = new MultilayerPerceptron();
		
		mlp.buildClassifier(data);
		mlp.setHiddenLayers("0");
		System.out.println(mlp);*/
		
		DoubleMatrix initialWeight = new DoubleMatrix(1, 7);
		PerceptronTrainingRule ptr = new PerceptronTrainingRule(0.1, 0.01, 10, initialWeight);
		
		/*Enumeration instances = data.enumerateInstances();
		while(instances.hasMoreElements()){
			Instance instance = (Instance) instances.nextElement();
			for(int i = 0; i < instance.numAttributes(); i++){
				System.out.print(instance.value(i) + " ");
			}
			
			System.out.println();
		}*/
		
		ptr.buildClassifier(data);
		System.out.println(ptr);
		System.out.println(Arrays.asList(ptr.getOptions()));
	}
	
}
