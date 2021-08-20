/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Set;


/**
 * This class is a stub for naive bayes with feature hashing
 */
public class NaiveBayesFeatureHashing extends OnlineTextClassifier{

    public int logNbOfBuckets;
    private int nbOfBuckets;
    public int[][] counts; // counts[c][i]: The count of n-grams in e-mails of class c (spam: c=1) that hash to value i
    private int[] ngramCounts; //ngramCounts[c] the count of ngrams of class c (spam: c=1). Equal to sum over all columns of 'counts'.
    private int[] classCounts; //classCounts[c] the count of e-mails of class c (spam: c=1)
    private int seed;
    /* FILL IN HERE */

    /**
     * Initialize the naive Bayes classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses simple feature hashing: The features of this classifier are the hash values that n-grams
     * hash to.
     *
     * @param logNbOfBuckets The hash function hashes to the range [0,2^NbOfBuckets-1]
     * @param threshold The threshold for classifying something as positive (spam). Classify as spam if Pr(Spam|n-grams)>threshold)
     */
    public NaiveBayesFeatureHashing(int logNbOfBuckets, double threshold){
        this.logNbOfBuckets=logNbOfBuckets;
        this.threshold = threshold;
        this.nbOfBuckets=((int) Math.pow(2, logNbOfBuckets));

        this.counts = new int[2][this.nbOfBuckets];
        this.ngramCounts = new int[2];
    	this.classCounts = new int[2];
    	this.seed = (int) Math.random() * 1000;
        
        // Laplace estimation
        for (int c = 0; c <= 1; c ++) {
            this.classCounts[c] += 1;
            for (int i = 0; i < nbOfBuckets; i++) {
                this.counts[c][i] += 1;
            }
        }
        ngramCounts[0] = nbOfBuckets;
    	ngramCounts[1] = nbOfBuckets;
    	classCounts[0] = 1;
    	classCounts[1] = 1;
        
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
     * THIS METHOD IS REQUIRED
     *
     * @param labeledText is an incoming e-mail with a spam/ham label
     */
    @Override
    public void update(LabeledText labeledText){
        super.update(labeledText);

        int feature_label = labeledText.label;
        
        // Increment total spam/ham counter
        classCounts[feature_label]++;
        
        // Update ngramCounts counter
        ngramCounts[feature_label] += labeledText.text.ngrams.size();
        
        // Handle n-gram counters
        for (String ngram : labeledText.text.ngrams)
        	counts[feature_label][hash(ngram)]++;
    }


    /**
     * Uses the current model to make a prediction about the incoming e-mail belonging to class "1" (spam)
     * The prediction is the probability for the e-mail to be spam.
     * If the probability is larger than the threshold, then the e-mail is classified as spam.
     *
     * THIS METHOD IS REQUIRED
     *
     * @param text is an parsed incoming e-mail
     * @return the prediction
     */
    @Override
    public double makePrediction(ParsedText text) {
        double logJointProbSpam = logJointProb(text,1);
        double logJointProbHam = logJointProb(text,0);
        
        double logPr = logJointProbSpam - HelperFunctions.logSum(logJointProbSpam,logJointProbHam);
        
        // Convert log probability to probability
        double pr = Math.exp(logPr);
        
        if (pr < 0 || pr > 1) {
            System.out.println("FAILURE: makePrediction returned invalid probability score, which should be in [0,1]: Probability = " + pr);
        }
        
        return pr;
    }
    
    /**
     * Calculates the log of the joint prob. distribution P(Text, Class = c)
     * @param text
     * @param c
     * @return
     */
    public double logJointProb(ParsedText text, int c) {
        double result = 0;
        
        // ln(Pr[Text = given set of n-grams | S = c])
        for (String ngram : text.ngrams) {
        	int hashValue = hash(ngram);
        	result += Math.log((double) counts[c][hashValue]);
        }
        result -= text.ngrams.size() * Math.log((double) ngramCounts[c]);
        
        // ln(Pr[S = c])
        result += Math.log(classCounts[c]) - HelperFunctions.logSum(Math.log(classCounts[0]), Math.log(classCounts[1]));

        return result; 	
    }

    //public double logSum(double log_a, double log_b) {
    //    return log_a + Math.log(1 + Math.exp(log_b - log_a));
    //}

    /**
     * This runs your code.
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 7) {
            System.err.println("Usage: java NaiveBayesFeatureHashing <indexPath> <stopWordsPath> <logNbOfBuckets> <threshold> <outPath> <reportingPeriod> <maxN> [-writeOutAllPredictions]");
            throw new Error("Expected 7 or 8 arguments, got " + args.length + ".");
        }
        try {
            // parse input
            String indexPath = args[0];
            String stopWordsPath = args[1];
            int logNbOfBuckets = Integer.parseInt(args[2]);
            double threshold = Double.parseDouble(args[3]);
            String out = args[4];
            int reportingPeriod = Integer.parseInt(args[5]);
            int n = Integer.parseInt(args[6]);
            boolean writeOutAllPredictions = args.length>7 && args[7].equals("-writeOutAllPredictions");

            // initialize e-mail stream
            MailStream stream = new MailStream(indexPath, new EmlParser(stopWordsPath,n));

            // initialize learner
            NaiveBayesFeatureHashing nb = new NaiveBayesFeatureHashing(logNbOfBuckets, threshold);

            // generate output for the learning curve
            EvaluationMetric[] evaluationMetrics = new EvaluationMetric[5]; //ADD AT LEAST TWO MORE EVALUATION METRICS
            evaluationMetrics[0] = new Accuracy();
            evaluationMetrics[1] = new Recall();
            evaluationMetrics[2] = new Precision();
            evaluationMetrics[3] = new F1Score();
            evaluationMetrics[4] = new BalancedAccuracy();
            nb.makeLearningCurve(stream, evaluationMetrics, out+".nbfh", reportingPeriod, writeOutAllPredictions);

        } catch (FileNotFoundException e) {
            System.err.println(e.toString());
        }
    }


}
