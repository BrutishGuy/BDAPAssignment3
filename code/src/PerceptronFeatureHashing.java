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
    private int hash(String str){
        int v;
        int seed = 1;

        if (this.logNbOfBuckets <= 32) {
            v = MurmurHash.hash32(str, seed);
            //System.out.println(Integer.toString(v));
            //System.out.println(Integer.toString(this.nbOfBuckets));
            v = v & (this.nbOfBuckets - 1);
            //System.out.println(Integer.toString(v));

        } else  {
            long long_v = MurmurHash.hash64(str, seed);
            long_v = long_v % this.nbOfBuckets;
            v = Math.toIntExact(long_v);
            v = v & (this.nbOfBuckets - 1);

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

        double[] feature_vector = new double[this.nbOfBuckets];
        Set<String> feature_ngrams = labeledText.text.ngrams;

        for (int i = 0; i < this.nbOfBuckets; i++) {
            feature_vector[i] = 0;
        }
        
        for (String ngram: feature_ngrams) {
            hashValue = hash(ngram);
            feature_vector[hashValue] += 1;
        }
        
        double weighted_sum = 0;
        double probabilityOfSpam = 0;
        for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i ++) {
            weighted_sum += feature_vector[feature_i] * this.weights[feature_i];
        }
        weighted_sum += bias;
        
        int prediction;
        if (this.threshold != 0.0) {
            probabilityOfSpam = sigmoid_activation(weighted_sum);
            prediction = super.classify(probabilityOfSpam);
        } else {
            prediction = super.classify(weighted_sum);
        }
        //System.out.println(Double.toString(weighted_sum));
        //double probabilityOfSpam = sigmoid_activation(weighted_sum);
        
        if (prediction == 0) {
            prediction = -1;
        }
        
        double error = feature_label - prediction;
        
        this.sum_error += Math.pow(error, 2);
        double lambda = 0.01;
        int m = 1;
        // updating the weights, with an L2 penalty
	bias = bias + this.learningRate * error;
        double L2_regularization_penalty = 0;
	for (int weight_i = 0; weight_i < this.nbOfBuckets; weight_i++) {
            L2_regularization_penalty +=  Math.pow(weights[weight_i], 2);
        }  
        L2_regularization_penalty *= lambda / (2 * m);
        
	for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i++) {
            if (this.threshold != 0.0) {
                weights[feature_i] = weights[feature_i] + this.learningRate * error * sigmoid_activation_derivative(weighted_sum) * feature_vector[feature_i];
            } else {
                weights[feature_i] = weights[feature_i] + this.learningRate * error * feature_vector[feature_i];
            }
        }
        				     
    }

    public static double sigmoid_activation(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }
    
    public static double sigmoid_activation_derivative(double x) {
        return (Math.pow(Math.E,(-1*x)))/Math.pow((1 + Math.pow(Math.E,(-1*x))), 2);
    }
    
    public static double linear_activation(double x) {
        return x;
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

        double[] feature_vector = new double[this.nbOfBuckets];
        Set<String> feature_ngrams = text.ngrams;
        
        for (String ngram: feature_ngrams) {
            hashValue = hash(ngram);
            feature_vector[hashValue] += 1;
        }
        
        double weighted_sum = 0;
        for (int feature_i = 0; feature_i < this.nbOfBuckets; feature_i ++) {
            weighted_sum += feature_vector[feature_i] * this.weights[feature_i];
        }
        weighted_sum += bias;

        if (this.threshold != 0) {
            pr = sigmoid_activation(weighted_sum);
        } else {
            pr = weighted_sum;
        }
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
