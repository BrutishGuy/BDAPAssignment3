/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;
import java.util.Set;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

/**
 * This class is a stub for naive Bayes with count-min sketch
 */
public class NaiveBayesCountMinSketch extends OnlineTextClassifier{

    private int nbOfHashes;
    private int logNbOfBuckets;
    private int[][] hashAB; //hashAB[h][0] and hashAB[h][1] are the resp. constants 'a' and 'b' used in universal hashing for the h'th hash function. 

    private int[][][] counts; // counts[c][h][i]: The count of n-grams in e-mails of class c (spam: c=1)
                              // that hash to value i for the h'th hash function.
    private int[] classCounts; //classCounts[c] the count of e-mails of class c (spam: c=1)
    private int[] ngramCounts; //ngramCounts[c] the count of ngrams of class c (spam: c=1)
    private int nbOfBuckets;
    private int prime;
    private int seed;
    
    /* FILL IN HERE */

    /**
     * Initialize the naive Bayes classifier
     *
     * THIS CONSTRUCTOR IS REQUIRED, DO NOT CHANGE THE HEADER
     * You can write additional constructors if you wish, but make sure this one works
     *
     * This classifier uses the count-min sketch to estimate the conditional counts of the n-grams
     *
     * @param nbOfHashes The number of hash functions in the count-min sketch
     * @param logNbOfBuckets The hash functions hash to the range [0,2^NbOfBuckets-1]
     * @param threshold The threshold for classifying something as positive (spam). Classify as spam if Pr(Spam|n-grams)>threshold)
     */
    public NaiveBayesCountMinSketch(int nbOfHashes, int logNbOfBuckets, double threshold){
        this.nbOfHashes = nbOfHashes;
        this.logNbOfBuckets=logNbOfBuckets;
        this.threshold = threshold;
        this.prime = Primes.findLeastPrimeNumber(this.nbOfBuckets);
    	this.seed = (int) Math.random() * 1000;
        
        this.nbOfBuckets =((int) Math.pow(2, logNbOfBuckets));

        this.counts = new int[2][this.nbOfHashes][this.nbOfBuckets];
        this.classCounts = new int[2];
        
        // Init hashAB
    	hashAB = new int[this.nbOfHashes][2];
    	Random rand = new Random();
    	for (int i = 0; i < this.nbOfHashes; i++) {
    		hashAB[i][0] = rand.nextInt(this.prime - 1) + 1; // a != 0
    		hashAB[i][1] = rand.nextInt(this.prime);
    	}
    	
    	// Init counts
    	counts = new int[2][nbOfHashes][nbOfBuckets];
    	for (int c = 0; c < 2; c++)
    		for (int h = 0; h < nbOfHashes; h++)
    			for (int i = 0; i < nbOfBuckets; i++)
    				counts[c][h][i] = 1;
        
        // Init ngramCounts
    	ngramCounts = new int[2];
    	ngramCounts[0] = nbOfBuckets;
    	ngramCounts[1] = nbOfBuckets;
    	
    	// Init classCounts
    	classCounts = new int[2];
    	classCounts[0] = 1;
    	classCounts[1] = 1;
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
        	v = -1; // will cause system exception OutOfBounds
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
        
        // update classCounts
        classCounts[feature_label]++;
        
        // update ngramCounts
        ngramCounts[feature_label] += labeledText.text.ngrams.size();
        
        // update counts
        for (String ngram : labeledText.text.ngrams)
        	for (int h = 0; h < nbOfHashes; h++)
        		counts[feature_label][h][hash(ngram,h)]++;
        
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
        double pr = 0.0;
        
        List<String> ngramList = new ArrayList<String>(text.ngrams);

        // minCount[c][ngram] is the minimum count over all hash functions given class c and ngram
        int[][] minCount = new int[2][ngramList.size()];
        for (int i = 0; i < ngramList.size(); i++) {
        	
        	String ngram = ngramList.get(i);
        	int[] min = getMinCount(ngram);
        	minCount[0][i] = min[0];
        	minCount[1][i] = min[1];
        	
        }
        
        // from this point, similar to Feature Hashing
        double logJPDSpam = logJointProb(minCount[1],1);
        double logJPDHam = logJointProb(minCount[0],0);
        
        double logPr = logJPDSpam - HelperFunctions.logSum(logJPDSpam,logJPDHam);
        
        // Convert logPr to pr
        pr = Math.exp(logPr);
        
        // Testing
        if (pr < 0 || pr > 1)
        	System.out.println("FAILURE IN NB-CMS makePrediction(): pr = " + pr);
        
        
        
        return pr;
    }

    /**
     * Calculates the minimum count of ngram for both spam and ham.
     * @param ngram
     * @return An array with 2 elements, one minimum count for each class (ham and spam).
     */
    private int[] getMinCount(String ngram) {
    	int[] min = new int[2];
    	min[0] = ngramCounts[0];
    	min[1] = ngramCounts[1];
    	
    	for (int h = 0; h < nbOfHashes; h++) {
    		int hashValue = hash(ngram,h);
    		int countHam = counts[0][h][hashValue];
    		int countSpam = counts[1][h][hashValue];
    		
    		if (min[0] > countHam)
    			min[0] = countHam;
    		
    		if (min[1] > countSpam)
    			min[1] = countSpam;
    		
    	}

    	return min;
    }
    
    /**
     * Calculates the log of the joint prob. distribution P(Text, Class = c)
     * @param text
     * @param c
     * @return
     */
    public double logJointProb(int[] minCount, int c) {
        double result = 0;
        
        // ln(Pr[Text = given set of n-grams | S = c])
        for (int count : minCount) {
            // Note that probability P[ngram|c] = minCounts[ngram] / ngramCounts[c]
        	result += Math.log((double) count);
        }
        result -= minCount.length * Math.log((double) ngramCounts[c]);
        
        // ln(Pr[S = c])
        result += Math.log(classCounts[c]) - HelperFunctions.logSum(Math.log(classCounts[0]), Math.log(classCounts[1]));
    	
        
        return result;
    	
    }
    /**
     * This runs your code.
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 8) {
            System.err.println("Usage: java NaiveBayesCountMinSketch <indexPath> <stopWordsPath> <logNbOfBuckets> <nbOfHashes> <threshold> <outPath> <reportingPeriod> <maxN> [-writeOutAllPredictions]");
            throw new Error("Expected 8 or 9 arguments, got " + args.length + ".");
        }
        try {
            // parse input
            String indexPath = args[0];
            String stopWordsPath = args[1];
            int logNbOfBuckets = Integer.parseInt(args[2]);
            int nbOfHashes = Integer.parseInt(args[3]);
            double threshold = Double.parseDouble(args[4]);
            String out = args[5];
            int reportingPeriod = Integer.parseInt(args[6]);
            int n = Integer.parseInt(args[7]);
            boolean writeOutAllPredictions = args.length>8 && args[8].equals("-writeOutAllPredictions");

            // initialize e-mail stream
            MailStream stream = new MailStream(indexPath, new EmlParser(stopWordsPath,n));

            // initialize learner
            NaiveBayesCountMinSketch nb = new NaiveBayesCountMinSketch(nbOfHashes ,logNbOfBuckets, threshold);

            // generate output for the learning curve
            EvaluationMetric[] evaluationMetrics = new EvaluationMetric[5]; //ADD AT LEAST TWO MORE EVALUATION METRICS
            evaluationMetrics[0] = new Accuracy();
            evaluationMetrics[1] = new Recall();
            evaluationMetrics[2] = new Precision();
            evaluationMetrics[3] = new F1Score();
            evaluationMetrics[4] = new BalancedAccuracy();
            nb.makeLearningCurve(stream, evaluationMetrics, out+".nbcms", reportingPeriod, writeOutAllPredictions);

        } catch (FileNotFoundException e) {
            System.err.println(e.toString());
        }
    }
}
