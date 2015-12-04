package weka.customClassifier.singlePerceptron;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

/**
 * Created by wiragotama on 12/4/15.
 */
public class MultiClassSinglePerceptron extends Classifier {
    private List<SinglePerceptron> ensemble;
    private double learningRate;
    private double mseThreshold;
    private int maxIteration;
    private double initialWeight;
    private boolean randomWeight;
    private int numClasses;
    private NominalToBinary nominalToBinaryFilter;
    private Normalize normalizeFilter;
    private Instances dataset;

    public MultiClassSinglePerceptron(double learningRate, double mseThreshold, int maxIteration, boolean randomWeight, double initialWeight)
    {
        this.learningRate = learningRate;
        this.mseThreshold = mseThreshold;
        this.maxIteration = maxIteration;
        this.randomWeight = randomWeight;
        this.initialWeight = initialWeight;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.dataset = data;
        this.ensemble = new ArrayList<SinglePerceptron>();

        //Siapkan jumlah single perceptron
        for (int i=0; i<data.classAttribute().numValues()-1; i++) {
            SinglePerceptron sp;
            if (!this.randomWeight)
                 sp = new SinglePerceptron(this.learningRate, this.mseThreshold, this.maxIteration, this.initialWeight);
            else
                sp = new SinglePerceptron(this.learningRate, this.mseThreshold, this.maxIteration);
            sp.setAlgo(Options.PerceptronTrainingRule);
            this.ensemble.add(sp);
        }

        //pecah data untuk masing2 perceptron
        List<Instances> transformeds = new ArrayList<Instances>();
        for (int i=0; i<this.ensemble.size(); i++) {
            Instances transformed = transformData(this.dataset, i);
            transformeds.add(transformed);
        }


        //build classifier
        for (int i=0; i<this.ensemble.size(); i++) {
            this.ensemble.get(i).buildClassifier(transformeds.get(i));
        }
    }

    public List<SinglePerceptron> getSinglePerceptrons() {
        return ensemble; //untuk ganti config per single perceptronnya, pakai fungsi ini
    }

    public Instances transformData(Instances data, int currentPerceptron) throws Exception {
        // test whether classifier can handle the data
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        Instances datas = new Instances(data);
        datas.deleteWithMissingClass();

        // remove instances with missing values
        Enumeration attributes = datas.enumerateAttributes();
        while(attributes.hasMoreElements()){
            Attribute attribute = (Attribute) attributes.nextElement();
            datas.deleteWithMissing(attribute);
        }

        // check if data contains nominal attributes
        if(nominalData(datas))
            datas = nominalToNumeric(datas);

        // check if data contains nominal attributes
        if(nominalData(datas))
            datas = nominalToNumeric(datas);

        // normalize numeric data
        this.normalizeFilter = new Normalize();
        normalizeFilter.setInputFormat(datas);
        datas = Filter.useFilter(datas, normalizeFilter);

        Instances transformedData = new Instances(datas);

        Add filter = new Add();
        filter.setAttributeIndex("last");
        filter.setNominalLabels("Yes, No");
        filter.setAttributeName("NewNominal");
        filter.setInputFormat(transformedData);
        transformedData = Filter.useFilter(transformedData, filter);

        transformedData.setClass(transformedData.attribute("NewNominal"));
        transformedData.deleteAttributeAt(datas.classIndex());

        int i = 0;
        int x = 0;
        while (i<transformedData.numInstances()) {
            boolean tidakDimasukkan = false;
            for (int j=0; j<currentPerceptron && !tidakDimasukkan; j++) {
                //System.out.println(datas.instance(x).classValue()+" "+j);
                if (datas.instance(x).classValue()==(double)j)
                    tidakDimasukkan = true;
            }
            if (!tidakDimasukkan) {
                if (datas.instance(i).classValue() == (double) currentPerceptron) {
                    transformedData.instance(i).setClassValue(0.0);
                } else transformedData.instance(i).setClassValue(1.0);
                i++;
            }
            else {
                transformedData.delete(i);
            }
            x++;
        }
        //System.out.println(transformedData);

        return transformedData;
    }

    /**
     * @return capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

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

    public Instances nominalToNumeric(Instances data) throws Exception {
        this.nominalToBinaryFilter = new NominalToBinary();
        this.nominalToBinaryFilter.setInputFormat(data);

        data = Filter.useFilter(data, this.nominalToBinaryFilter);

        return data;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("MultiLayerPerceptron: cannot handle missing value");
        }
        Instance predict = instance;
        if (nominalData(instance)) {
            nominalToBinaryFilter.input(instance);
            predict = nominalToBinaryFilter.output();
        }

        this.normalizeFilter.input(predict);
        predict = normalizeFilter.output();

        //masukin input
        double[] outputs = new double[dataset.numClasses()];
        boolean found = false;
        for (int i=0; i<ensemble.size()-1 && !found; i++) {
            double out = this.ensemble.get(i).classifyInstance(instance);
            if (Double.compare(out, 0.0) == 0) { //yes
                outputs[i] = 1.0;
                found = true;
            }
            else {
                outputs[i] = 0.0;
            }
        }

        if (!found) {
            double out = this.ensemble.get(ensemble.size()-1).classifyInstance(instance);
            if (Double.compare(out, 0.0)==0) {
                outputs[this.ensemble.size()-1] = 1.0;
            }
            else outputs[this.ensemble.size()] = 1.0;
        }
        
        return outputs;
    }

    public static Instances loadDatasetArff(String filePath) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        return loader.getDataSet();
    }

    public static void main(String args[]) throws Exception {
        String dataset = "example/iris.2D.arff";

        Instances data = loadDatasetArff(dataset);
        data.setClass(data.attribute(data.numAttributes() - 1));

        MultiClassSinglePerceptron ptr = new MultiClassSinglePerceptron(0.1, 0.01, 100, true, 0.0);
        ptr.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(ptr, data);
        System.out.println(eval.toSummaryString());
    }
}
