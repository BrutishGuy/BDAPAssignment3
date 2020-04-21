/**
 * Written by Victor Gueorguiev, 2020
 */


/**
 * This class calculates the F1 score
 */
public class F1Score implements EvaluationMetric {

    /**
     * Calculates the F1 score given the values of the contingency table
     *
     * @param TP Number of true positives
     * @param FP Number of false positives
     * @param TN Number of true negatives
     * @param FN Number of false negatives
     * @return evaluation evaluate
     */
    @Override
    public double evaluate(int TP, int FP, int TN, int FN) {
        double precision = ((double) TP)/(TP + FP);
        double recall = ((double) TP)/(TP+FN);
        return 2.0 * (precision * recall)/(precision + recall);
    }

    /**
     *
     * @return name of the evaluator
     */
    @Override
    public String name() {
        return "f1score";
    }
}

