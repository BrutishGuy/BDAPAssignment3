/**
 * Written by Victor Gueorguiev, 2020
 */


/**
 * This class calculates the recall score
 */
public class Recall implements EvaluationMetric {

    /**
     * Calculates the recall given the values of the contingency table
     *
     * @param TP Number of true positives
     * @param FP Number of false positives
     * @param TN Number of true negatives
     * @param FN Number of false negatives
     * @return evaluation evaluate
     */
    @Override
    public double evaluate(int TP, int FP, int TN, int FN) {
        return ((double) TP)/(TP+FN);
    }

    /**
     *
     * @return name of the evaluator
     */
    @Override
    public String name() {
        return "recall";
    }
}

