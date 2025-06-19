package com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelectors;

public enum ParentSelectionType {
    ROULETTE_WHEEL_PREFER_LARGE,
    ROULETTE_WHEEL_PREFER_SMALL,
    TOURNAMENT_PREFER_LARGE,
    TOURNAMENT_PREFER_SMALL,
    ELITES_PREFER_LARGE,
    ELITES_PREFER_SMALL;

    public static ParentSelectionType fromString(String type) throws IllegalArgumentException, NullPointerException{
        return ParentSelectionType.valueOf(type);
    }
}