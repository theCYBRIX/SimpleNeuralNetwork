package com.github.thecybrix.simpleneuralnetwork.serialization.json;

final public class JsonParsingTools {
    
    private JsonParsingTools(){}

    public static String missingFields(String className, String... fieldNames) {
        StringBuilder message = new StringBuilder("Unable to construct ")
                                .append(className)
                                .append(" from string. Missing Field")
                                .append(fieldNames.length > 1 ? "s: " : ": ");
        
        message.append(fieldNames[0]);
        for (int i = 1; i < fieldNames.length; i++)
                message.append(", ")
                       .append(fieldNames[i]);
        
        message.append(".");
        return message.toString();
    }

}
