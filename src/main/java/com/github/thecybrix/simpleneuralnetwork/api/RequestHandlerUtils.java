package com.github.thecybrix.simpleneuralnetwork.api;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.serialization.json.CustomGsonFactory;
import com.github.thecybrix.simpleneuralnetwork.training.evolution.ParentSelector;
import com.github.thecybrix.util.Fraction;
import com.github.thecybrix.util.Pair;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

final public class RequestHandlerUtils {
    final public static Gson GSON;

    static {
        GsonBuilder gsonBuilder = CustomGsonFactory.getInstance().newBuilder();
        gsonBuilder.enableComplexMapKeySerialization();
        gsonBuilder.serializeSpecialFloatingPointValues();
        gsonBuilder.setLenient();
        GSON = gsonBuilder.create();
    }

    private RequestHandlerUtils(){}

    public static enum ParentSelection {
        ROULETTE_WHEEL_PREFER_LARGE,
        ROULETTE_WHEEL_PREFER_SMALL,
        TOURNAMENT_PREFER_LARGE,
        TOURNAMENT_PREFER_SMALL,
        ELITES_PREFER_LARGE,
        ELITES_PREFER_SMALL;
    }

    

    public static <E extends MutableNeuralNetwork> ParentSelector<E> getParentSelector(ParentSelection selector) throws NullPointerException {
        switch (Objects.requireNonNull(selector)) {
            case ELITES_PREFER_LARGE:
                return ParentSelector.eliteSelection();
                
            case ELITES_PREFER_SMALL:
                return ParentSelector.eliteSelection(Comparator.reverseOrder());
            
            case ROULETTE_WHEEL_PREFER_LARGE:
                return ParentSelector.rouletteWheelSelection();
            
            case ROULETTE_WHEEL_PREFER_SMALL:
                return ParentSelector.rouletteWheelSelection(true);
            
            case TOURNAMENT_PREFER_LARGE:
                return ParentSelector.tournamentSelection(Fraction.of(10, 100));
            
            case TOURNAMENT_PREFER_SMALL:
                return ParentSelector.tournamentSelection(Fraction.of(10, 100), Comparator.reverseOrder());

            default:
                throw new NullPointerException();
        }
    }


    public static <T> T requireNonNull(T obj, String name) throws NoSuchElementException {
        if(obj == null) throw new NoSuchElementException("Field \"" + name + "\" not found.");
        return obj;
    }

    public static void requireField(JsonObject object, String field) throws NoSuchElementException {
        if(!object.has(field)) throw new NoSuchElementException("Missing field: \"" + field + "\"");
    }

    public static void requireFields(JsonObject object, String... fields) throws NoSuchElementException {
        boolean[] missing = new boolean[fields.length];
        byte missingCount = 0;

        for (int i = 0; i < fields.length; i++)
            if(!object.has(fields[i])){
                missing[i] = true;
                missingCount++;
            }

        if (missingCount == 0) return;

        StringBuilder errorMsg = new StringBuilder("Missing field");
        errorMsg.append(missingCount > 1 ? "s: " : ": ");

        int i = 0;
        while(!missing[i])
            i++;

        errorMsg.append("\"").append(fields[i]).append("\"");
        for (i = i + 1; i < missing.length; i++) {
            if(missing[i])
                errorMsg.append(", \"").append(fields[i]).append("\"");
        }

        throw new NoSuchElementException(errorMsg.toString());
    }

    public static String stackTraceToString(Exception e){
        StringWriter stackTrace = new StringWriter();
        e.printStackTrace(new PrintWriter(stackTrace));
        return stackTrace.toString();
    }

    public static Map<Object, Object> map(Object... entries){
        if(entries == null) throw new NullPointerException();

        HashMap<Object, Object> dict = new HashMap<>(entries.length);
        for (int i = 0; i < entries.length; i+=2)
            dict.put(entries[i], entries[i + 1]);

        return dict;
    }

    @SuppressWarnings("unchecked")
    public static <M extends Map<E, T>, E, T> M map(M map, Pair<E, T>... entries) throws NullPointerException{
        if(map == null) throw new NullPointerException("Map is null.");
        if(entries == null) throw new NullPointerException("Entries array is null.");

        for (Pair<E, T> pair : entries)
            map.put(pair.getFirst(), pair.getSecond());

        return map;
    }

    public static <E> List<E> unpackSuppliers(List<? extends Supplier<E>> list){
        return list.parallelStream()
                    .map(x -> x.get())
                    .collect(Collectors.toList());
    }

}
