package weka.customClassifier.MLP_WIRA;

/**
 * Created by wiragotama on 11/23/15.
 */
public class Neuron {

    private double weights[]; //weight[0] untuk bias
    private double lastWeights[];
    private ActivationFunction activationFunction;
    private double lastOutput;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        this.weights = null;
        this.lastWeights = null;
        this.lastOutput = 0;
    }

    public double[] getWeights() {
        return weights;
    }

    public double[] getLastWeights() {
        return lastWeights;
    }

    public void setWeights(double[] _weights) {
        if (this.weights == null)
            setLastWeights(new double[_weights.length]);
        else
            setLastWeights(this.weights);

        this.weights = new double[_weights.length];
        for (int i=0; i<_weights.length; i++)
            this.weights[i] = _weights[i];
    }

    public void setLastWeights(double[] _lastWeights) {
        this.lastWeights = new double[_lastWeights.length];
        for (int i=0; i<_lastWeights.length; i++)
            this.lastWeights[i] = _lastWeights[i];
    }

    public double output(double[] inputs) {
        double sum = weights[0]; //bias factor 1*weights[0]
        for (int i = 0; i < inputs.length; i++) {
            sum += (weights[i+1] * inputs[i]);
        }

        if (activationFunction.equals(ActivationFunction.SIGMOID)) {
            lastOutput =  1.0 / (1.0 + Math.pow(Math.E, -1.0 * sum));
        } else lastOutput = sum;

        return lastOutput;
    }

    public double getLastOutput() {
        return lastOutput;
    }

    public String toString() {
        StringBuffer str = new StringBuffer("");
        str.append("Activation Function = " + activationFunction.name() + "\n");
        for (int i=0; i<weights.length; i++) {
            if (i==0) str.append("Bias Weight = " + weights[i] + "\n");
            else      str.append("Weight from previous layer neuron" + (i - 1) + " = " + weights[i] + "\n");
        }
        return str.toString();
    }

    public static enum ActivationFunction {
        LINEAR, SIGMOID;
    }
}
