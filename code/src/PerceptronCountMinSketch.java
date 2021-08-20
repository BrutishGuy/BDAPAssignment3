/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.Set;


/**
 * This class is a stub for a perceptron with count-min sketch
 */
public class PerceptronCountMinSketch extends OnlineTextClassifier{

    private int nbOfHashes;
    private int logNbOfBuckets;
    private int nbOfBuckets;
    private double learningRate;
    private double bias;
    private int prime;
    private int seed;
    private int[][] hashAB;		//hashAB[h][0] and hashAB[h][1] are the resp. constants 'a' and 'b' used in universal hashing for the h'th hash function. 
    
    private double sum_error;
    private double[][] weights; // weights[h][i]: The h'th weight estimate for n-grams that hash to value i for the h'th hash function


    /* FILL IN HERE */

    /**
     * Initialize the perceptron classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses the count-min sketch to estimate the weights of the n-grams
     *
     * @param nbOfHashes The number of hash functions in the count-min sketch
     * @param logNbOfBuckets The hash functions hash to the range [0,2^NbOfBuckets-1]
     * @param learningRate The size of the updates of the weights
     */
    public PerceptronCountMinSketch(int nbOfHashes, int logNbOfBuckets, double learningRate){
        this.nbOfHashes = nbOfHashes;
        this.logNbOfBuckets=logNbOfBuckets;
        this.learningRate = learningRate;
        this.threshold = 0;

        this.nbOfBuckets=((int) Math.pow(2, logNbOfBuckets));
        this.sum_error = 0;
        bias = 0;
        weights = new double[this.nbOfHashes][this.nbOfBuckets];
        this.prime = Primes.findLeastPrimeNumber(this.nbOfBuckets);
    	this.seed = (int) Math.random() * 1000;
    	
    	// Init hashAB
    	hashAB = new int[this.nbOfHashes][2];
    	Random rand = new Random();
    	for (int i = 0; i < this.nbOfHashes; i++) {
    		hashAB[i][0] = rand.nextInt(this.prime - 1) + 1; // a != 0
    		hashAB[i][1] = rand.nextInt(this.prime);
    	}
        
        // here we initialize the weights to random values between 0 and 1
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (int j = 0; j < this.nbOfBuckets; j++) {
                weights[hash_i][j] = 0;//Math.random();
            }   
        }

    }
    
        /**
     * Initialize the perceptron classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses the count-min sketch to estimate the weights of the n-grams
     *
     * @param nbOfHashes The number of hash functions in the count-min sketch
     * @param logNbOfBuckets The hash functions hash to the range [0,2^NbOfBuckets-1]
     * @param learningRate The size of the updates of the weights
     * @param threshold The threshold for a positive prediction. Useful for ROC curve
     */
    public PerceptronCountMinSketch(int nbOfHashes, int logNbOfBuckets, double learningRate, double threshold){
        this.nbOfHashes = nbOfHashes;
        this.logNbOfBuckets=logNbOfBuckets;
        this.learningRate = learningRate;
        this.threshold = threshold;

        this.nbOfBuckets=((int) Math.pow(2, logNbOfBuckets));
        this.sum_error = 0;
        bias = 0;
        weights = new double[this.nbOfHashes][this.nbOfBuckets];
        // here we initialize the weights to random values between 0 and 1
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (int j = 0; j < this.nbOfBuckets; j++) {
                weights[hash_i][j] = 0;//Math.random();
            }   
        }

    }

    /**
     * Calculate the hash value of the h'th hash function for string str
     *
     * THIS METHOD IS REQUIRED
     *
     * The hash function hashes to the range [0,2^NbOfBuckets-1]
     * This method should work for h in the range [0, nbOfHashes-1]
     *
     * @param str The string to calculate the hash function for
     * @param h The number of the hash function to use.
     * @return the hash value of the h'th hash function for string str
     */
    private int hash(String str, int h){
        int v;

        if (h < 0 || h >= nbOfBuckets){
        	v = -1;
        	System.out.println("Failure in NB CMS hash(): h out of range");
        } else {
        	
        	int x = MurmurHash.hash32(str, seed);
        	int a = hashAB[h][0];
        	int b = hashAB[h][1];
        	
        	int y = HelperFunctions.posMod(a*x + b, prime);
        	v = HelperFunctions.posMod(y, nbOfBuckets);
        	
        }
        	

        return v;
    }
    /**
     * This method will update the parameters of your model using the incoming mail.
     *
     * THIS METHOD IS REQUIRED
     *
     * @param labeledText is an incoming e-mail with a spam/ham label
     */
    @Override
    public void update(LabeledText labeledText){
        super.update(labeledText);

        int feature_label = labeledText.label;
        int gradient_direction;
        if (feature_label == 0) {
            gradient_direction = -1;
            feature_label = -1;
        } else {
            gradient_direction = 1;
        }
        
        int hashValue;

        double[][] feature_vector = new double[this.nbOfHashes][this.nbOfBuckets];
        Set<String> feature_ngrams = labeledText.text.ngrams;
         
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (String ngram: feature_ngrams) {
                hashValue = hash(ngram, hash_i);
                feature_vector[hash_i][hashValue] += 1;
            } 
        }

        
        //double[][] weighted_sum_mat = new double[this.nbOfHashes][this.nbOfBuckets];
        double weighted_sum = 0;
        for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i ++) {
            double[] weighted_sum_vec = new double[this.nbOfHashes];
            // here we calculate the x[i] * w[i] for each feature component for each h
            // then we take the median across all h to get a final x[i] * w[i] for our weighted sum
            for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
                weighted_sum_vec[hash_i] = feature_vector[hash_i][feature_i] * this.weights[hash_i][feature_i];
            }
            weighted_sum += findMeanSketch(weighted_sum_vec, this.nbOfHashes);
        }
        weighted_sum += bias;

        //System.out.println(Double.toString(weighted_sum));
        //double probabilityOfSpam = sigmoid_activation(weighted_sum);
        int prediction = super.classify(weighted_sum);
        if (prediction == 0) {
            prediction = -1;
        }
        
        double error = feature_label - prediction;
        
        this.sum_error += Math.pow(error, 2);
        double lambda = 0.01;
        int m = 1;
        // updating the weights, with an L2 penalty
        /*double[] L2_regularization_penalty = new double[this.nbOfHashes];
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (int weight_i = 0; weight_i < this.nbOfBuckets; weight_i++) {
                L2_regularization_penalty[hash_i] +=  Math.pow(weights[hash_i][weight_i], 2);
            }  
            L2_regularization_penalty[hash_i] *= lambda / (2 * m);
        }*/
        
        bias = bias + this.learningRate * error;
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i++) {
                weights[hash_i][feature_i] = weights[hash_i][feature_i] + this.learningRate * error * feature_vector[hash_i][feature_i];
            }
        }
    }

    /**
     * Uses the current model to make a prediction about the incoming e-mail belonging to class "1" (spam)
     * If the prediction is positive, then the e-mail is classified as spam.
     *
     * This method gives the output of the perceptron, before it is passed through the threshold function.
     *
     * THIS METHOD IS REQUIRED
     *
     * @param text is an parsed incoming e-mail
     * @return the prediction
     */
    @Override
    public double makePrediction(ParsedText text) {
        double pr = 0;
        int hashValue;

        double[][] feature_vector = new double[this.nbOfHashes][this.nbOfBuckets];
        Set<String> feature_ngrams = text.ngrams;
         
        for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
            for (String ngram: feature_ngrams) {
                hashValue = hash(ngram, hash_i);
                feature_vector[hash_i][hashValue] += 1;
            } 
        }

        
        //double[][] weighted_sum_mat = new double[this.nbOfHashes][this.nbOfBuckets];
        double weighted_sum = 0;
        for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i ++) {
            double[] weighted_sum_vec = new double[this.nbOfHashes];
            // here we calculate the x[i] * w[i] for each feature component for each h
            // then we take the median across all h to get a final x[i] * w[i] for our weighted sum
            for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
                weighted_sum_vec[hash_i] = feature_vector[hash_i][feature_i] * this.weights[hash_i][feature_i];
            }
            weighted_sum += findMeanSketch(weighted_sum_vec, this.nbOfHashes);
        }
        weighted_sum += bias;

        //pr = sigmoid_activation(weighted_sum);
        pr = weighted_sum;
        return pr;
    }

    /**
     * Finds the median of a given double array. Used in implementing CountMedianSketch
     * for a perceptron
     * @param a is a double array of terms x[i] * w[i] for various hash functions h
     * @param n is the number of hashes used, to know the length of a
     * @return the median
     */
    public static double findMedianSketch(double a[], int n) 
    { 
        // First we sort the array 
        Arrays.sort(a); 
  
        // check for even case 
        if (n % 2 != 0) {
            return (double)a[n / 2]; 
        } else { 
            return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0; 
        }
    }
    
    /**
     * Finds the mean of a given double array. Used in implementing CountMedianSketch
     * for a perceptron
     * @param a is a double array of terms x[i] * w[i] for various hash functions h
     * @param n is the number of hashes used, to know the length of a
     * @return the mean
     */
    public static double findMeanSketch(double a[], int n) 
    { 
        int sum = 0; 
        for (int i = 0; i < n; i++)  
            sum += a[i]; 
      
        return (double)sum / (double)n; 
    }

    /**
     * This runs your code.
     */
    public static void main(String[] args) throws IOException {

        if (args.length < 8) {
            System.err.println("Usage: java PerceptronCountMinSketch <indexPath> <stopWordsPath> <logNbOfBuckets> <nbOfHashes> <learningRate> <outPath> <reportingPeriod> <maxN> [-writeOutAllPredictions]");
            throw new Error("Expected 8 or 9 arguments, got " + args.length + ".");
        }
        try {
            // parse input
            String indexPath = args[0];
            String stopWordsPath = args[1];
            int logNbOfBuckets = Integer.parseInt(args[2]);
            int nbOfHashes = Integer.parseInt(args[3]);
            double learningRate = Double.parseDouble(args[4]);
            String out = args[5];
            int reportingPeriod = Integer.parseInt(args[6]);
            int n = Integer.parseInt(args[7]);
            boolean writeOutAllPredictions = args.length>8 && args[8].equals("-writeOutAllPredictions");

            // initialize e-mail stream
            MailStream stream = new MailStream(indexPath, new EmlParser(stopWordsPath,n));

            // initialize learner
            PerceptronCountMinSketch perceptron = new PerceptronCountMinSketch(nbOfHashes ,logNbOfBuckets, learningRate);

            // generate output for the learning curve
            EvaluationMetric[] evaluationMetrics = new EvaluationMetric[5]; //ADD AT LEAST TWO MORE EVALUATION METRICS
            evaluationMetrics[0] = new Accuracy();
            evaluationMetrics[1] = new Recall();
            evaluationMetrics[2] = new Precision();
            evaluationMetrics[3] = new F1Score();
            evaluationMetrics[4] = new BalancedAccuracy();
            perceptron.makeLearningCurve(stream, evaluationMetrics, out+".pcms", reportingPeriod, writeOutAllPredictions);

        } catch (FileNotFoundException e) {
            System.err.println(e.toString());
        }
    }


}
