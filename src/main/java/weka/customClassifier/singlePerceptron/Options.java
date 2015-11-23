package weka.customClassifier.singlePerceptron;

public class Options {
	public static final int PerceptronTrainingRule = 0;
	public static final int DeltaRuleBatch = 1;
	public static final int DeltaRuleIncremental = 2;
	
	public static final String ptr = "PerceptronTrainingRule";
	public static final String drb = "DeltaRuleBatch";
	public static final String dri = "DeltaRuleIncremental";
	
	public static String algorithm(int algo){
		if(algo == PerceptronTrainingRule)
			return ptr;
		else if(algo == DeltaRuleBatch)
			return drb;
		else if(algo == DeltaRuleIncremental)
			return dri;
		else
			return "Invalid algorithm";
	}
}
