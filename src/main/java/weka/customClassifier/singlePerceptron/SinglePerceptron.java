package weka.customClassifier.singlePerceptron;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;

import org.jblas.DoubleMatrix;

import random.RandomGen;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Capabilities.Capability;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

public class SinglePerceptron extends Classifier{
	private static final double bias = 1.0; // bias unit
	
	private DoubleMatrix weightVector; // vector for storing weights
	private DoubleMatrix deltaWeightVector; // vector for storing delta weights
	
	private double learningRate; // learning rate for weight update
	private double mseThreshold; // MSE threshold
	private int maxIteration; // maximum number of epoch
	
	private boolean randomWeight;
	private double initialWeight; // user-given initial weights
	private long randomSeed; // seed used for random number generator
	
	private StringBuffer output; // string buffer describing the model
	
	private NominalToBinary nominalToBinaryFilter; // filter to convert nominal attributes to binary numeric attributes
	
	private Attribute classAttribute;
	
	private int selectedAlgo;
	
	private Instances dataSet;
	
	private boolean useNormalization;
	private Normalize normalizeFilter;
	
	/**
	 * Default constructor
	 */
	public SinglePerceptron() {
		this.learningRate = 0.1;
		this.mseThreshold = 0.0;
		this.maxIteration = 10;
		this.randomWeight = true;
		this.randomSeed = 0;
		this.useNormalization = true;
		output = new StringBuffer();
	}
	
	/**
	 * User-defined constructor with random initial weights
	 * @param learningRate
	 * @param threshold
	 * @param maxIteration
	 */
	public SinglePerceptron(double learningRate, double mseThreshold, int maxIteration) {
		this.learningRate = learningRate;
		this.mseThreshold = mseThreshold;
		this.maxIteration = maxIteration;
		this.randomWeight = true;
		this.randomSeed = 0;
		this.useNormalization = true;
		output = new StringBuffer();
	}
	
	/**
	 * User-defined constructor with given initial weights
	 * @param learningRate
	 * @param msethreshold
	 * @param deltaMSE
	 * @param maxIteration
	 * @param initialWeight
	 */
	public SinglePerceptron(double learningRate, double msethreshold, int maxIteration, double initialWeight) {
		this.learningRate = learningRate;
		this.mseThreshold = msethreshold;
		this.maxIteration = maxIteration;
		this.randomWeight = false;
		this.initialWeight = initialWeight;
		this.randomSeed = 0;
		this.useNormalization = true;
		output = new StringBuffer();		
	}
	
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	
	public void setMSEThreshold(double mseThreshold) {
		this.mseThreshold = mseThreshold;
	}
	
	public void setMaxIteration(int maxIteration) {
		this.maxIteration = maxIteration;
	}
	
	public void setInitialWeight(double initialWeight) {
		this.randomWeight = false;
		this.initialWeight = initialWeight;
	}
	
	public void setSeed(long seed) {
		this.randomWeight = true;
		
		if (seed >= 0)
			this.randomSeed = seed;
		else
			this.randomSeed = 0;
	}
	
	public void setAlgo(int algorithm) {
		if(algorithm != Options.DeltaRuleBatch && algorithm != Options.DeltaRuleIncremental && algorithm != Options.PerceptronTrainingRule)
			throw new RuntimeException("invalid algorithm");
		
		this.selectedAlgo = algorithm;
	}

	public void setUseNormalization(boolean useNormalize) {
		this.useNormalization = useNormalize;
	}
	
	public double getLearningRate() {
		return this.learningRate;
	}
	
	public double getMSEThreshold() {
		return this.mseThreshold;
	}
	
	public int getMaxIteration() {
		return this.maxIteration;
	}
	
	public double getInitialWeight() {
		return this.initialWeight;
	}
	
	public double getSeed() {
		return this.randomSeed;
	}
	
	public String getAlgo() {
		return Options.algorithm(this.selectedAlgo);
	}
	
	public boolean getUseNormalization() {
		return this.useNormalization;
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
	 * Get current settings of classifier
	 */
	public String[] getOptions() {
		String[] options = new String[5];
		
		StringBuffer learningRate = new StringBuffer("-LearningRate ");
		learningRate.append(this.learningRate);
		options[0] = learningRate.toString();
		
		StringBuffer threshold = new StringBuffer("-Threshold ");
		threshold.append(this.mseThreshold);
		options[1] = threshold.toString();
		
		StringBuffer maxIteration = new StringBuffer("-MaxIteration ");
		maxIteration.append(this.maxIteration);
		options[2] = maxIteration.toString();
		
		StringBuffer randomWeight = new StringBuffer("-RandomWeight ");
		randomWeight.append(this.randomWeight);
		options[3] = randomWeight.toString();
		
		StringBuffer algorithm = new StringBuffer("-Algorithm ");
		algorithm.append(Options.algorithm(this.selectedAlgo));
		options[4] = algorithm.toString();
		
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
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.BINARY_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);
    
		return result;
	}

	/**
	 * Sign activation function
	 * @param x the floating-point value whose signum is to be returned
	 * @return
	 */
	private double sign(double x) {
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
	private double sum(Instance instance) {
		double sum = 0.0;
		
		sum += bias * weightVector.get(0);
		
		for(int i = 1; i < weightVector.length; i++){
			sum += instance.value(i - 1) * weightVector.get(i);
		}
		
		return sum;
	}
	
	/**
	 * Target depending on whether class attribute is nominal
	 * @param instance
	 * @param nominal
	 * @return
	 */
	private double target(Instance instance, boolean nominal) {
		double target = 0.0;
		
		if(nominal && this.selectedAlgo == Options.PerceptronTrainingRule){
			if(Double.compare(instance.value(instance.classAttribute()), 1.0) == 0)
				target = 1.0;
			else if (Double.compare(instance.value(instance.classAttribute()), 0.0) == 0)
				target = -1.0;
		}else{
			target = instance.value(instance.classAttribute());
		}
		
		return target;
	}
	
	/**
	 * Check whether training data contains nominal attributes
	 * @param data training data
	 * @param instances
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
	 * Check whether single instance contains nominal attributes
	 * @param data training data
	 * @param instances
	 * @return true if training data contains nominal attributes
	 */
	private boolean nominalData(Instance instance) {
		boolean found = false;
		
		Enumeration attributes = instance.enumerateAttributes();
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

		// normalize numeric data
		if (useNormalization) {
			normalizeFilter = new Normalize();
			normalizeFilter.setInputFormat(data);
			data = Filter.useFilter(data, normalizeFilter);
		}
		
		this.classAttribute = data.classAttribute();
		
		/*Enumeration instancess = data.enumerateInstances();
		while(instancess.hasMoreElements()){
			Instance instance = (Instance) instancess.nextElement();
			for(int i = 0; i < instance.numAttributes(); i++){
				if(instance.attribute(i).isNominal())
					System.out.print(instance.stringValue(i) + " ");
				//else
					System.out.print(instance.value(i) + " ");
			}
			
			System.out.println();
		}*/

		// create weight and delta weight vector
		this.weightVector = new DoubleMatrix(1, data.numAttributes());
		this.deltaWeightVector = new DoubleMatrix(1, data.numAttributes());
		
		// set initial weight either with random number or given initial weight
		if(this.randomWeight){
			randomizeWeight(weightVector);
		}else{
			for(int i = 0; i < this.weightVector.length; i++){
				this.weightVector.put(i, this.initialWeight);
			}
		}
		
		int epoch = 0;
		double meanSquaredError = Double.POSITIVE_INFINITY;
		
		// training iteration, finishes either when epoch reaches max iteration or MSE < threshold
		while(epoch < this.maxIteration && Double.compare(meanSquaredError, this.mseThreshold) >= 0) {
			
			Enumeration instances = data.enumerateInstances();
			
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
								
				double sum = this.sum(instance);
				double output = 0.0;
				
				if(this.selectedAlgo == Options.PerceptronTrainingRule)
					output = this.sign(sum);	
				else 
					output = sum;	
								
				double target = this.target(instance, instance.classAttribute().isNominal());
				double error = target - output;
				
				for(int i = 0; i < instance.numAttributes(); i++){
					if(i == 0){
						if(this.selectedAlgo != Options.DeltaRuleBatch)
							deltaWeightVector.put(i, this.learningRate * error * bias);
						else
							deltaWeightVector.put(i, deltaWeightVector.get(i) + error * bias);
					}else{
						if(this.selectedAlgo != Options.DeltaRuleBatch)
							deltaWeightVector.put(i, this.learningRate * error * instance.value(i - 1));
						else
							deltaWeightVector.put(i, deltaWeightVector.get(i) + error * instance.value(i - 1));
					}
				}
				
				if(this.selectedAlgo != Options.DeltaRuleBatch)
					weightVector.addi(deltaWeightVector);
			}
			
			if(this.selectedAlgo == Options.DeltaRuleBatch){
				deltaWeightVector.muli(this.learningRate);
				weightVector.addi(deltaWeightVector);
				deltaWeightVector = DoubleMatrix.zeros(deltaWeightVector.rows, deltaWeightVector.columns);
			}
			
			instances = data.enumerateInstances();
			
			double squaredError = 0.0;
			while(instances.hasMoreElements()){
				Instance instance = (Instance) instances.nextElement();
				
				double sum = this.sum(instance);
				double output = 0.0;
				
				if(this.selectedAlgo == Options.PerceptronTrainingRule)
					output = this.sign(sum);
				else
					output = sum;
					
				double target = this.target(instance, instance.classAttribute().isNominal());
				double error = target - output;
				
				squaredError += Math.pow(error, 2.0);
			}
			
			meanSquaredError = squaredError / 2.0;
			output.append("epoch " + epoch + ": " + weightVector + "\n");
			output.append("epoch " + epoch + " MSE: " + meanSquaredError + "\n");
			
			epoch++;
		}
	}
	
	/**
	 * Compute output for delta rule
	 * @param value
	 * @param lowerBound
	 * @param upperBound
	 * @return
	 */
	private double computeOutput(double value, double lowerBound, double upperBound) {	
		if (Double.compare(value, lowerBound)<0)
			return lowerBound;
		else if (Double.compare(value, upperBound)>0)
			return upperBound;
		else if(Double.compare(Math.abs(lowerBound - value), Math.abs(upperBound-value))>=0) {
			return upperBound;
		}else{
			return lowerBound;
		}
			
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

        double[] outputs = new double[instance.numClasses()];
        if(this.classAttribute.isNumeric())
        	outputs[0] = this.classifyInstance(instance);
        else if(this.classAttribute.isNominal()){
        	outputs[(int) classifyInstance(instance)] = 1.0;
        }
        	               
        return outputs;
    }
	
	/**
	 * @param instance instance to be classified
	 * @return class value of instance
	 * @throws Exception 
	 */
	public double classifyInstance(Instance instance) throws Exception {	
		Instance predict = instance;
		
		if(this.nominalData(instance)){
			nominalToBinaryFilter.input(instance);
			predict = nominalToBinaryFilter.output();
		}

		if (this.useNormalization){
			normalizeFilter.input(predict);
			predict = normalizeFilter.output();
			
		}
		
		double sum = this.sum(predict);
		double output = 0.0;
		
		if(this.selectedAlgo == Options.PerceptronTrainingRule){	
			output = this.sign(sum);
			
			if(this.classAttribute.isNominal()){
				if(Double.compare(output, 1.0) == 0)
					output = 1.0;
				else if (Double.compare(output, -1.0) == 0)
					output = 0.0;
			}
		}else{
			output = sum;
			
			if(this.classAttribute.isNominal()){
				output = computeOutput(sum, 0.0, 1.0);
			}
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
	 * @return class attribute
	 */
	public Attribute classAttribute() {
		return this.classAttribute;
	}
	
	/**
	 * @return string describing the model
	 */
	public String toString() {
		return this.output.toString();
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
		String dataset = "example/weather.numeric.arff";
		
		Instances data = loadDatasetArff(dataset);
		data.setClass(data.attribute(data.numAttributes() - 1));
		
		SinglePerceptron ptr = new SinglePerceptron();
		ptr.setLearningRate(0.1);
		ptr.setMSEThreshold(0.01);
		ptr.setMaxIteration(10);
		ptr.setInitialWeight(0.0);
		ptr.setUseNormalization(true);
		ptr.setAlgo(Options.DeltaRuleIncremental);
		
		ptr.buildClassifier(data);		
		
		System.out.println(ptr);
	
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(ptr, data);
		System.out.println(eval.toSummaryString());
	}
}
