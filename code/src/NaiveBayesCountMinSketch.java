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

/**
 * This class is a stub for naive Bayes with count-min sketch
 */
public class NaiveBayesCountMinSketch extends OnlineTextClassifier{

    private int nbOfHashes;
    private int logNbOfBuckets;
    private int[][][] counts; // counts[c][h][i]: The count of n-grams in e-mails of class c (spam: c=1)
                              // that hash to value i for the h'th hash function.
    private int[] classCounts; //classCounts[c] the count of e-mails of class c (spam: c=1)
    private int nbOfBuckets;
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

        this.nbOfBuckets =((int) Math.pow(2, logNbOfBuckets));

        this.counts = new int[2][this.nbOfHashes][this.nbOfBuckets];
        this.classCounts = new int[2];
        
        // Laplace estimation
        for (int c = 0; c <= 1; c ++) {
            this.classCounts[c] += 1;
            for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
                for (int i = 0; i < nbOfBuckets; i++) {
                    this.counts[c][hash_i][i] += 1;
                }
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
        
        // we differentiate out Murmur hashes by adjusting the seeds
        // this will give us unique hashes.
        // we keep the seed fixed rather than random for reproducibility.
        int seed = 1 + h * h;
        
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
        int hashValue;

        Set<String> feature_ngrams = labeledText.text.ngrams;
        
        for (String ngram: feature_ngrams) {
            for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
                hashValue = hash(ngram, hash_i);
                this.counts[feature_label][hash_i][hashValue] += 1;
            }
        }
        
        this.classCounts[feature_label] += 1;
        
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
        int hashValue;
        
        //this is okay not to use log sum formula, the counts and logs are not between 0 and 1
        double LogPrS = Math.log(this.classCounts[1]) - log_sum(Math.log(this.classCounts[0]), Math.log(this.classCounts[1]));
        double LogPrH = Math.log(this.classCounts[0]) - log_sum(Math.log(this.classCounts[0]), Math.log(this.classCounts[1]));
        
        // we now do the following formula:
        // log(Pr(S | text)) = log(Pr(S)) + sum_over_all_words_in_text(log(Pr(word|S)) - log(product over all words in text(Pr(word|S) + Pr(word|H))) 
        Set<String> feature_ngrams = text.ngrams;
        
        // initialize to 1 for multiplication
        double PrAllWordsinS = 0.0;
        double PrAllWordsNotS = 0.0;
        for (String ngram: feature_ngrams) {
            double[] PrAllWordsinSVec = new double[this.nbOfHashes];
            double[] PrAllWordsNotSVec = new double[this.nbOfHashes];
            for (int hash_i = 0; hash_i < this.nbOfHashes; hash_i++) {
                hashValue = hash(ngram, hash_i);
                PrAllWordsinSVec[hash_i] = this.counts[1][hash_i][hashValue];
                PrAllWordsNotSVec[hash_i] = this.counts[0][hash_i][hashValue];
            }
            double MinAllWordsinS = Arrays.stream(PrAllWordsinSVec).min().getAsDouble();
            double MinAllWordsNotS = Arrays.stream(PrAllWordsNotSVec).min().getAsDouble();
            PrAllWordsinS += Math.log((double)(MinAllWordsinS + 1)/(double)(this.classCounts[1] + 2));
            PrAllWordsNotS += Math.log((double)(MinAllWordsNotS + 1)/(double)(this.classCounts[0] + 2));

        }
        double prS = LogPrS + PrAllWordsinS;
        double prH = LogPrH + PrAllWordsNotS;
        
        double pr = Math.exp(prS)/(Math.exp(prS) + Math.exp(prH));        //System.out.println("Done----------------------------------------------------------");
        //System.out.println(Double.toString(pr));

        return pr;
    }

    public double log_sum(double log_a, double log_b) {
        return log_a + Math.log(1 + Math.exp(log_b - log_a));
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
