package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.Arrays;
import java.util.Objects;

public final class PropertyType {
    final public static PropertyType
        STRING = new PropertyType("String"),
        BOOLEAN = new PropertyType("boolean"),
        BYTE = new PropertyType("byte"),
        SHORT = new PropertyType("short"),
        INTEGER = new PropertyType("int"),
        LONG = new PropertyType("long"),
        FLOAT = new PropertyType("float"),
        DOUBLE = new PropertyType("double"),
        ARRAY = new PropertyType("Array"),
        MAP = new PropertyType("Map"),
        OBJECT = new PropertyType("Object");
    
    final private String READABLE_NAME;

    private PropertyType(String typeName) throws IllegalArgumentException, NullPointerException {
        READABLE_NAME = validateTypeName(typeName);
    }

    private PropertyType(String typeName, String... genericTypes) throws IllegalArgumentException, NullPointerException {
        Objects.requireNonNull(typeName, "Type name is null.");
        
        if(genericTypes.length > 0){
            int arrayStartIndex = typeName.indexOf('[');
            if(arrayStartIndex >= 0){
                String arrayChars = typeName.substring(arrayStartIndex, typeName.length());
                typeName = typeName.substring(0, arrayStartIndex) + getGenericsString(genericTypes) + arrayChars;
            } else {
                typeName += getGenericsString(genericTypes);
            }
        }

        READABLE_NAME = validateTypeName(typeName);
    }

    private PropertyType(String typeName, PropertyType... genericTypes) throws IllegalArgumentException, NullPointerException {
        this(typeName, Arrays.stream(genericTypes).map(x -> x.toString()).toArray(String[]::new));
    }

    public String toString(){
        return READABLE_NAME;
    }

    public static PropertyType of(Class<?> classOfT) throws IllegalArgumentException, NullPointerException {
        return new PropertyType(classOfT.getSimpleName());
    }

    public static PropertyType of(Class<?> classOfT, Class<?>... genericTypes) throws IllegalArgumentException, NullPointerException {
        return new PropertyType(classOfT.getSimpleName(), Arrays.stream(genericTypes).map(x -> x.getSimpleName()).toArray(String[]::new));
    }

    public static PropertyType of(String typeName, String... genericTypes) throws IllegalArgumentException, NullPointerException {
        return new PropertyType(typeName, genericTypes);
    }

    public static PropertyType of(PropertyType typeName, PropertyType... genericTypes) throws IllegalArgumentException, NullPointerException {
        return new PropertyType(typeName.toString(), genericTypes);
    }

    public static PropertyType arrayOf(String typeName, int dimensions) throws IllegalArgumentException, NullPointerException {
        if(dimensions <= 0) throw new IllegalArgumentException("Dimensions is <= 0.");
        StringBuilder builder = new StringBuilder(typeName);
        for (int i = 0; i < dimensions; i++) {
            builder.append("[]");
        }
        return new PropertyType(builder.toString());
    }

    public static PropertyType arrayOf(String typeName) throws IllegalArgumentException, NullPointerException {
        return arrayOf(typeName, 1);
    }

    public static PropertyType arrayOf(PropertyType type) throws IllegalArgumentException, NullPointerException {
        return arrayOf(type.toString(), 1);
    }

    public static PropertyType arrayOf(PropertyType type, int dimensions) throws IllegalArgumentException, NullPointerException {
        return arrayOf(type.toString(), dimensions); 
    }

    public static PropertyType arrayOf(String typeName, PropertyType... genericTypes) throws IllegalArgumentException, NullPointerException {
        return arrayOf(new PropertyType(typeName, genericTypes), 1);
    }

    public static PropertyType arrayOf(String typeName, String... genericTypes) throws IllegalArgumentException, NullPointerException {
        return arrayOf(new PropertyType(typeName, genericTypes), 1);
    }

    public static PropertyType arrayOf(String typeName, int dimensions, PropertyType... genericTypes) throws IllegalArgumentException, NullPointerException {
        return arrayOf(new PropertyType(typeName, genericTypes), dimensions);
    }

    public static PropertyType arrayOf(String typeName, int dimensions, String... genericTypes) throws IllegalArgumentException, NullPointerException {
        return arrayOf(new PropertyType(typeName, genericTypes), dimensions);
    }

    public static PropertyType mapOf(PropertyType keyType, PropertyType valueType) throws IllegalArgumentException, NullPointerException {
        return of(MAP, keyType, valueType);
    }

    public static PropertyType mapOf(String keyType, String valueType) throws IllegalArgumentException, NullPointerException {
        return of(MAP.toString(), keyType, valueType);
    }

    private static String validateTypeName(String name) throws IllegalArgumentException, NullPointerException {
        if(name == null) throw new NullPointerException("Name is null.");
        if(name.isEmpty()) throw new IllegalArgumentException("Name has length 0.");
        int genericDepth = 0;
        int arrayDepth = 0;
        boolean lastWasLetter = false;
        char lastChar = name.charAt(0);
        int charIndex = -1;
        
        for(char c : name.toCharArray()){
            charIndex += 1;
            switch (c) {
                case '<':
                    if( !lastWasLetter || arrayDepth != 0 )
                        illegalNameFormatting(name, charIndex);
                    genericDepth += 1;
                    lastWasLetter = false;
                    break;

                case ',':
                    if( genericDepth == 0 || (!lastWasLetter && lastChar != '>' && lastChar != ']'))
                        illegalNameFormatting(name, charIndex);
                    lastWasLetter = false;
                    break;

                case '>':
                    if(genericDepth == 0 || !(lastWasLetter || lastChar == ']' || lastChar == '>') || arrayDepth != 0)
                        illegalNameFormatting(name, charIndex);
                    genericDepth -= 1;
                    lastWasLetter = false;
                    break;

                case '[':
                    if(!lastWasLetter && lastChar != ']' && lastChar != '>' || charIndex == 0)
                        illegalNameFormatting(name, charIndex);
                    arrayDepth += 1;
                    lastWasLetter = false;
                    break;

                case ']':
                    if(lastChar != '[')
                        illegalNameFormatting(name, charIndex);
                    arrayDepth -= 1;
                    break;

                case ' ':
                    if(genericDepth == 0 || lastChar != ',')
                        illegalNameFormatting(name, charIndex);
                    lastWasLetter = false;
                    break;
            
                default:

                    boolean isLetter = Character.isLetter(c);

                    if(!isLetter && (Character.isDigit(c) && (charIndex == 0 || (!lastWasLetter && !Character.isDigit(lastChar)))) || arrayDepth != 0 || !lastWasLetter && charIndex != 0 && lastChar != ' ' && lastChar != '<' )
                        illegalNameFormatting(name, charIndex);
                    lastWasLetter = isLetter;
            }
            lastChar = c;
        }

        if(genericDepth != 0 || arrayDepth != 0)
            illegalNameFormatting(name, charIndex);;

        return name;
    }

    private static <T> void validateGenericTypeArray(T[] genericTypes) throws IllegalArgumentException {
        if(genericTypes == null)
            throw new IllegalArgumentException("Generic types array is null.");
        if(genericTypes.length == 0)
            throw new IllegalArgumentException("Generic types array is empty.");
        if(Arrays.stream(genericTypes).anyMatch(x -> x == null))
            throw new IllegalArgumentException("Generic types array contains null.");
    }

    private static void illegalNameFormatting(String name, int charIndex) throws IllegalArgumentException {
        throw new IllegalArgumentException("\"" + name + "\" is not a valid type name. Error on character \"" + name.charAt(charIndex) + "\" at index " + charIndex + ".");
    }

    static private String getGenericsString(String[] genericTypes) throws IllegalArgumentException, NullPointerException {
        validateGenericTypeArray(genericTypes);
        for(String type : genericTypes)
            validateTypeName(type);
        return getGenericsStringUnchecked(genericTypes);
    }

    static private <T> String getGenericsStringUnchecked(T[] genericTypes){
        StringBuilder builder = new StringBuilder("<");
        builder.append(genericTypes[0]);
        for(int i = 1; i < genericTypes.length; i++)
            builder.append(", ").append(genericTypes[i]);
        builder.append(">");
        return builder.toString();
    }
}
