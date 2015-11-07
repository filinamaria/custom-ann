package weka.custom_classifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.WeightedInstancesHandler;
import weka.core.converters.ArffLoader;

public class PerceptronTrainingRule extends Classifier {
	private static final String dataset = "example/weather.nominal.arff";
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	public static Instances loadDatasetArff(String filePath) throws IOException
    { 
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(filePath));
		return loader.getDataSet();
    }

	public static void main(String [] args) throws Exception{
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		
		Instances data = loadDatasetArff(dataset);
		data.setClass(data.attribute(data.numAttributes() - 1));
		mlp.buildClassifier(data);
		mlp.setHiddenLayers("0");
		System.out.println(Arrays.asList(mlp.getOptions()));
	}
	
}
