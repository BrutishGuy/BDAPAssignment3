/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */


/**
 * This class calculates the accuracy
 */
public class Accuracy implements EvaluationMetric {

    /**
     * Calculates the accuracy given the values of the contingency table
     *
     * @param TP Number of true positives
     * @param FP Number of false positives
     * @param TN Number of true negatives
     * @param FN Number of false negatives
     * @return evaluation evaluate
     */
    @Override
    public double evaluate(int TP, int FP, int TN, int FN) {
        return ((double) TP+TN)/(TP+TN+FP+FN);
    }

    /**
     *
     * @return name of the evaluator
     */
    @Override
    public String name() {
        return "acc";
    }
}
