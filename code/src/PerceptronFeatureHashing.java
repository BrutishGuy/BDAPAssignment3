/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;


/**
 * This class is a stub for a perceptron with count-min sketch
 */
public class PerceptronFeatureHashing extends OnlineTextClassifier{

    private int logNbOfBuckets;
    private int nbOfBuckets;
    private double learningRate;
    private double bias;
    private double[] weights; //weights[i]: The weight for n-grams that hash to value i
    private double sum_error;
    private int seed;
    /* FILL IN HERE */

    /**
     * Initialize the perceptron classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses simple feature hashing: The features of this classifier are the hash values that n-grams
     * hash to.
     *
     * @param logNbOfBuckets The hash functions hash to the range [0,2^NbOfBuckets-1]
     * @param learningRate The size of the updates of the weights
     */
    public PerceptronFeatureHashing(int logNbOfBuckets, double learningRate){
        this.logNbOfBuckets=logNbOfBuckets;
        this.learningRate = learningRate;
        this.nbOfBuckets=((int) Math.pow(2, logNbOfBuckets));
        this.sum_error = 0;
        bias = 0;
        this.seed = (int) Math.random() * 1000;
        weights = new double[this.nbOfBuckets];
        // here we initialize the weights to random values between 0 and 1
        for (int i = 0; i < this.nbOfBuckets; i++) {
            weights[i] = 0;//Math.random();
        }
    }

        /**
     * Initialize the perceptron classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses simple feature hashing: The features of this classifier are the hash values that n-grams
     * hash to.
     *
     * @param logNbOfBuckets The hash functions hash to the range [0,2^NbOfBuckets-1]
     * @param learningRate The size of the updates of the weights
     */
    public PerceptronFeatureHashing(int logNbOfBuckets, double learningRate, double threshold){
        this.logNbOfBuckets=logNbOfBuckets;
        this.learningRate = learningRate;
        this.nbOfBuckets=((int) Math.pow(2, logNbOfBuckets));
        this.sum_error = 0;
        this.threshold = threshold;
        bias = 0;
        this.seed = (int) Math.random() * 1000;
        
        weights = new double[this.nbOfBuckets];
        // here we initialize the weights to random values between 0 and 1
        for (int i = 0; i < this.nbOfBuckets; i++) {
            weights[i] = 0;//Math.random();
        }
    }

    /**
     * Calculate the hash value for string str
     *
     * THIS METHOD IS REQUIRED
     *
     * The hash function hashes to the range [0,2^NbOfBuckets-1]
     *
     * @param str The string to calculate the hash function for
     * @return the hash value of the h'th hash function for string str
     */
    public int hash(String str){
    	
    	int v = HelperFunctions.posMod(MurmurHash.hash32(str, seed), nbOfBuckets);
        return v;
        
    }


    /**
     * This method will update the parameters of your model using the incoming mail.
     *
     *
     * @param labeledText is an incoming e-mail with a spam/ham label
     */
    @Override
    public void update(LabeledText labeledText){
        super.update(labeledText);
        
        // if label = 0 then y = -1. If label = 1 then y = 1.
        double y = labeledText.label*2 - 1;
        double out = makePrediction(labeledText.text); // unthresholded (Delta Rule)
        
        double learningWeight = learningRate * (y - out);
        bias += learningWeight;
        for (String ngram : labeledText.text.ngrams)
        	weights[hash(ngram)] += learningWeight;  
    }

     /**
     * Uses the current model to make a prediction about the incoming e-mail belonging to class "1" (spam)
     * If the prediction is positive, then the e-mail is classified as spam.
     *
     * This method gives the output of the perceptron, before it is passed through the threshold function.
     *
     *
     * @param text is an parsed incoming e-mail
     * @return the prediction
     */
    @Override
    public double makePrediction(ParsedText text) {
        
    	// Calculate prediction pr = bias + (w . x)
    	// input vector x: x_i = 1 if a ngram hashes to i. x_i = 0 otherwise.
    	double pr = bias;
    	for (String ngram : text.ngrams)
    		pr += weights[hash(ngram)];

        
        return pr;
        
    }

    
   

    /**
     * This runs your code.
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 7) {
            System.err.println("Usage: java PerceptronFeatureHashing <indexPath> <stopWordsPath> <logNbOfBuckets> <learningRate> <outPath> <reportingPeriod> <maxN> [-writeOutAllPredictions]");
            throw new Error("Expected 7 or 8 arguments, got " + args.length + ".");
        }
        try {
            // parse input
            String indexPath = args[0];
            String stopWordsPath = args[1];
            int logNbOfBuckets = Integer.parseInt(args[2]);
            double learningRate = Double.parseDouble(args[3]);
            String out = args[4];
            int reportingPeriod = Integer.parseInt(args[5]);
            int n = Integer.parseInt(args[6]);
            boolean writeOutAllPredictions = args.length>7 && args[7].equals("-writeOutAllPredictions");

            // initialize e-mail stream
            MailStream stream = new MailStream(indexPath, new EmlParser(stopWordsPath,n));

            // initialize learner
            PerceptronFeatureHashing perceptron = new PerceptronFeatureHashing(logNbOfBuckets, learningRate);

            // generate output for the learning curve
            EvaluationMetric[] evaluationMetrics = new EvaluationMetric[5]; //ADD AT LEAST TWO MORE EVALUATION METRICS
            evaluationMetrics[0] = new Accuracy();
            evaluationMetrics[1] = new Recall();
            evaluationMetrics[2] = new Precision();
            evaluationMetrics[3] = new F1Score();
            evaluationMetrics[4] = new BalancedAccuracy();
            perceptron.makeLearningCurve(stream, evaluationMetrics, out+".pfh", reportingPeriod, writeOutAllPredictions);

        } catch (FileNotFoundException e) {
            System.err.println(e.toString());
        }
    }

    /*public void applyWeightConstraint(double weightConstraint) {
        double squared_length = 0;
        for(int i=0; i < inputDimension; ++i){
            squared_length += getWeight(i) * getWeight(i);
        }
        if(squared_length > weightConstraint){
            double ratio = Math.sqrt(weightConstraint / squared_length);
            for(int i=0; i < inputDimension; ++i) {
                setWeight(i, getWeight(i) * ratio);
            }
        }
    }*/

}
