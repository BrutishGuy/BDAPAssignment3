/**
 * Written by Victor Gueorguiev, 2020
 */


/**
 * This class calculates the balanced accuracy
 */
public class BalancedAccuracy implements EvaluationMetric {

    /**
     * Calculates the balanced accuracy given the values of the contingency table
     *
     * @param TP Number of true positives
     * @param FP Number of false positives
     * @param TN Number of true negatives
     * @param FN Number of false negatives
     * @return evaluation evaluate
     */
    @Override
    public double evaluate(int TP, int FP, int TN, int FN) {
        double recall = ((double) TP)/(TP + FN);
        double specificity = ((double) TN)/(TN+FP);
        return (recall + specificity)/2.0;
    }

    /**
     *
     * @return name of the evaluator
     */
    @Override
    public String name() {
        return "balancedaccuracy";
    }
}

