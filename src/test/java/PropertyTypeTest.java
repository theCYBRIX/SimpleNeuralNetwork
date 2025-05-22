import java.util.ArrayList;

import com.github.thecybrix.simpleneuralnetwork.server.PropertyType;

//Tests written using ChatGPT

public class PropertyTypeTest {
    public static void main(String[] args) {
        validTestCases();
        invalidTestCases();
    }

    private static void validTestCases(){
        try {
            ArrayList<PropertyType> validTypes = new ArrayList<>();
            PropertyType t;
        
            // 1. Primitive type
            t = PropertyType.of("int");
            validTypes.add(t);
        
            // 2. Class name in PascalCase
            t = PropertyType.of("MyClass");
            validTypes.add(t);
        
            // 3. Class name in camelCase
            t = PropertyType.of("myVariable1");
            validTypes.add(t);
        
            // 4. Generic type with one parameter
            t = PropertyType.of("List", "String");
            validTypes.add(t);
        
            // 5. Generic type with multiple parameters
            t = PropertyType.of("Map", "String", "Integer");
            validTypes.add(t);
        
            // 6. Nested generics
            t = PropertyType.of(PropertyType.of("Map"), PropertyType.of("String"), PropertyType.of("List", "Integer"));
            validTypes.add(t);
        
            // 7. Array of primitives
            t = PropertyType.arrayOf("int", 2);
            validTypes.add(t);
        
            // 8. Array of generics
            t = PropertyType.arrayOf("List", 3, PropertyType.of("String"));
            validTypes.add(t);
        
            // 9. Multi-dimensional array of custom types
            t = PropertyType.arrayOf(PropertyType.of("MyClass"), 3);
            validTypes.add(t);
        
            // 10. Map with arrays as key and value types
            t = PropertyType.mapOf(
                PropertyType.arrayOf("String", 1),
                PropertyType.arrayOf("Integer", 2)
            );
            validTypes.add(t);
        
            for(PropertyType type : validTypes)
                System.out.println("Correctly allowed valid type: " + type);
            System.out.println("Valid Tests Successful!");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Valid Test Cases Failed");
        }        
    }
    private static void invalidTestCases(){
        try {
            ArrayList<String> invalidTypes = new ArrayList<>();
            invalidTypes.add("");                     // 1. Empty string
            invalidTypes.add("123Invalid");           // 2. Starts with a number
            invalidTypes.add("Invalid@Type");         // 3. Contains special character
            invalidTypes.add("MyClass<>");            // 4. Invalid generic syntax
            invalidTypes.add("List<String,>");        // 5. Trailing comma in generic
            invalidTypes.add("[]");                   // 6. Invalid array type
            invalidTypes.add("String[");          // 7. Array declaration unfinished
            invalidTypes.add("MyClas]s<AnotherClass>"); // 8. Closing bracking within class name
            invalidTypes.add(null);                   // 9. Null string
            invalidTypes.add("Map<String");           // 10. Unterminated generic
            invalidTypes.add("Map<String, Integer>[][]<String>"); // 11. Invalid array parameter
            invalidTypes.add("Map<String, String><String>"); // 12. Invalid generic typing
        
            for (String invalid : invalidTypes) {
                try {
                    PropertyType.of(invalid);
                    throw new RuntimeException("Expected exception not thrown for invalid type: " + invalid);
                } catch (IllegalArgumentException | NullPointerException expected) {
                    System.out.println("Correctly threw exception for invalid type: " + invalid);
                }
            }
            System.out.println("Invalid Tests Successful!");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Invalid Test Cases Failed");
        }
    }
}
