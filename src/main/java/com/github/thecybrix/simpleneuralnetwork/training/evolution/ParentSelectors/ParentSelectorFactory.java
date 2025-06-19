package com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelectors;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;

public class ParentSelectorFactory<E extends MutableNeuralNetwork> {

    private final Map<String, ParentSelector<E>> custom = new HashMap<>();

    public ParentSelector<E> get(String key) {
        if (custom.containsKey(key.toUpperCase())) return custom.get(key.toUpperCase());
        return ParentSelectorFactory.createDefaultSelector(key);
    }

    public void register(String name, ParentSelector<E> selector) {
        custom.put(name.toUpperCase(), selector);
    }
    
    public boolean containsSelector(String name) throws NullPointerException {
        String upperase = Objects.requireNonNull(name, "Name is null.").toUpperCase();
        return Arrays.stream(ParentSelectionType.values()).map(Enum::name).anyMatch(x -> x.equals(upperase)) || custom.containsKey(upperase);
    }

    public String[] getSelectorNames() {
        return Stream.concat(
            Arrays.stream(ParentSelectionType.values()).map(Enum::name),
            custom.keySet().stream()
        ).toArray(String[]::new);
    }

    public static String[] getDefaultSelectorNames(){
        return Arrays.stream(ParentSelectionType.values()).map(Enum::name).toArray(String[]::new);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> createDefaultSelector(String name) {
        ParentSelectionType type = ParentSelectionType.fromString(name);
        return createDefaultSelector(type);
    }

    public static <E extends MutableNeuralNetwork> ParentSelector<E> createDefaultSelector(ParentSelectionType type) {
        switch (Objects.requireNonNull(type, "Selection type is null.")) {
            case ELITES_PREFER_LARGE:
                return new EliteSelector<>(Comparator.naturalOrder());
            case ELITES_PREFER_SMALL:
                return new EliteSelector<>(Comparator.reverseOrder());
            case ROULETTE_WHEEL_PREFER_LARGE:
                return new RouletteWheelSelector<>(false);
            case ROULETTE_WHEEL_PREFER_SMALL:
                return new RouletteWheelSelector<>(true);
            case TOURNAMENT_PREFER_LARGE:
                return new TournamentSelector<>(0.1f, Comparator.naturalOrder());
            case TOURNAMENT_PREFER_SMALL:
                return new TournamentSelector<>(0.1f, Comparator.reverseOrder());
            default:
                throw new IllegalArgumentException("Unknown selector type: " + type);
        }
    }
}

