package weka.customClassifier.multiLayerPerceptron;

import java.io.File;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class MLP extends Classifier{

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
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
		String dataset = "example/test.arff";
		
		Instances data = loadDatasetArff(dataset);
		data.setClassIndex(data.numAttributes() - 1);
		
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.
		mlp.buildClassifier(data);
		
		System.out.println(mlp);
	}
}
