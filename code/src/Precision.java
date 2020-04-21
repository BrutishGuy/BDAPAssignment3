/**
 * Written by Victor Gueorguiev, 2020
 */


/**
 * This class calculates the precision
 */
public class Precision implements EvaluationMetric {

    /**
     * Calculates the precision given the values of the contingency table
     *
     * @param TP Number of true positives
     * @param FP Number of false positives
     * @param TN Number of true negatives
     * @param FN Number of false negatives
     * @return evaluation evaluate
     */
    @Override
    public double evaluate(int TP, int FP, int TN, int FN) {
        return ((double) TP)/(TP + FP);
    }

    /**
     *
     * @return name of the evaluator
     */
    @Override
    public String name() {
        return "precision";
    }
}

