package mjsd.simpleneuralnetwork.gson;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Objects;
import java.util.function.IntFunction;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class ArrayAdapter<E> extends TypeAdapter<E[]> {
    final private Class<E> TYPE_OF_E;
    final private IntFunction<E[]> ARRAY_PROVIDER;

    public ArrayAdapter(Class<E> typeOfE, IntFunction<E[]> arrayProvider) throws NullPointerException{
        TYPE_OF_E = Objects.requireNonNull(typeOfE);
        ARRAY_PROVIDER = Objects.requireNonNull(arrayProvider);
    }

    @Override
    public void write(JsonWriter out, E[] value) throws IOException {
        if(value == null){
            out.nullValue();
            return;
        }

        final Gson GSON = CustomGsonFactory.getInstance();

        out.beginArray();
        for(E obj : value)
            out.jsonValue(GSON.toJson(obj, TYPE_OF_E));
        out.endArray();
    }

    @Override
    public E[] read(JsonReader in) throws IOException, JsonParseException {
        ArrayList<E> items = new ArrayList<>();
        final Gson GSON = CustomGsonFactory.getInstance();
        
        try {
            in.beginArray();
            while(in.hasNext()){
                items.add(GSON.fromJson(in, TYPE_OF_E));
            }
            in.endArray();
        } catch (Exception e) {
            throw new JsonParseException("Unable to construct array from given string.", e);
        }

        return items.toArray(ARRAY_PROVIDER.apply(items.size()));
    }
    
}
