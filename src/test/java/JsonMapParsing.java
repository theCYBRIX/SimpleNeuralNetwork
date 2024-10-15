import java.util.HashMap;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.serialization.json.CustomGsonFactory;
import com.google.gson.Gson;

public class JsonMapParsing extends TestingEnvironment {
    
    
    public static void main(String[] args) {
        final Gson GSON = CustomGsonFactory.getInstance().newBuilder().enableComplexMapKeySerialization().create();

        Map<String, Object> map = new HashMap<>();
        Map<String, double[]> values = new HashMap<>();

        int arraySize = (int) Math.round(Math.random() * 5);
        double[] v = new double[arraySize];
        for (int i = 0; i < arraySize; i++) {
            v[i] = Math.random();
        }

        values.put("array", v);
        map.put("title", "This is the title");
        map.put("values", values);

        String serialized = GSON.toJson(map);
        Map<?, ?> result = GSON.fromJson(serialized, Map.class);
        
        println(result.get("title") instanceof String);
        if(result.get("values") instanceof Map){
            println(true);
            println(((Map<?, ?>)result.get("values")).get("array").getClass());
        }
        

    }

    static class Generic<T>{
        T obj;

        public Generic(T obj) {
            this.obj = obj;
        }
    }
}
