package random;

import java.util.Random;

public class RandomGen {
	private Random random; // pseudo-random number generator
	private long seed; // pseudo-random number generator seed
	
	public RandomGen(){
		
	}
	
	public RandomGen(long seed){
		this.seed = seed;
		random = new Random(seed);
	}
	
	public void setSeed(long seed){
		this.seed = seed;
		random = new Random(seed);
	}
	
	public long getSeed(){
		return this.seed;
	}
	
	/**
     * Return real number uniformly in [0, 1).
     */
	public double uniform(){
		return random.nextDouble();
	}
	
	 /**
     * Returns an integer uniformly between 0 (inclusive) and N (exclusive).
     * @throws IllegalArgumentException if <tt>N <= 0</tt>
     */
    public int uniform(int N) {
        if (N <= 0) throw new IllegalArgumentException("Parameter N must be positive");
        return random.nextInt(N);
    }
    
    /**
     * Returns an integer uniformly in [a, b).
     * @throws IllegalArgumentException if <tt>b <= a</tt>
     * @throws IllegalArgumentException if <tt>b - a >= Integer.MAX_VALUE</tt>
     */
    public int uniform(int a, int b) {
        if (b <= a) throw new IllegalArgumentException("Invalid range");
        if ((long) b - a >= Integer.MAX_VALUE) throw new IllegalArgumentException("Invalid range");
        return a + uniform(b - a);
    }
    
    /**
     * Returns a real number uniformly in [a, b).
     * @throws IllegalArgumentException unless <tt>a < b</tt>
     */
    public double uniform(double a, double b) {
        if (!(a < b)) throw new IllegalArgumentException("Invalid range");
        return a + uniform() * (b-a);
    }
}
