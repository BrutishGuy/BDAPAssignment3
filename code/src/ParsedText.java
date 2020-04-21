/**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts, 2020
 */
import java.util.Set;


/**
 * This class represents a parsed text
 *
 * Optionally, you can make a subclass of Parsed Text that holds more features
 * that were extracted from the text, which can be used to learn better models.
 * If you do this, make sure that the classifiers does not require the
 * subclass. It should still be able to operate with n-grams alone.
 */
public class ParsedText {

    public final Set<String> ngrams;

    public ParsedText(Set<String> ngrams) {
        this.ngrams = ngrams;
    }

    @Override
    public String toString(){
        return ngrams.toString();
    }
}
