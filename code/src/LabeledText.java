/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */


/**
 * This class represents a parsed text and its label
 */
public class LabeledText {

    public final ParsedText text;
    public final int label;

    /**
     * Initializes labeled text
     *
     * @param text The parsed text
     * @param label The label (0 or 1 for binary classes)
     */
    public LabeledText(ParsedText text, int label) {
        this.text = text;
        this.label = label;
    }

    @Override
    public String toString(){
        return label+"\t"+text;
    }
}
