package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout;
import com.github.thecybrix.simpleneuralnetwork.core.NetworkLayout.NetworkLayer;
import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class NetworkLayoutAdapter extends TypeAdapter<NetworkLayout>{
    final private static String INPUT_LAYER = "inputs", OUTPUT_LAYER = "outputs", HIDDEN_LAYERS = "hiddenLayers";

    @Override
    public void write(JsonWriter out, NetworkLayout value) throws IOException {
        if(value == null){
            out.nullValue();
            return;
        }
        
        final Gson GSON = CustomGsonFactory.getInstance();
        List<NetworkLayer> hiddenLayers = value.getHiddenLayers();
        
        out.beginObject();
        out.name(INPUT_LAYER).jsonValue(GSON.toJson(value.getInputLayer(), NetworkLayer.class));
        out.name(OUTPUT_LAYER).jsonValue(GSON.toJson(value.getOutputLayer(), NetworkLayer.class));
        out.name(HIDDEN_LAYERS).jsonValue(GSON.toJson(hiddenLayers.toArray(new NetworkLayer[hiddenLayers.size()]), NetworkLayer[].class));
        out.endObject();
    }

    @Override
    public NetworkLayout read(JsonReader in) throws IOException {
        NetworkLayer inputs = null, outputs = null;
        NetworkLayer[] hiddenLayers = null;

        final Gson GSON = CustomGsonFactory.getInstance();

        try {
            in.beginObject();
            while(in.hasNext()){
                switch(in.peek()){
                    case NAME:
                        String var = in.nextName();
                        switch(var){
                            case INPUT_LAYER:
                                inputs = GSON.fromJson(in, NetworkLayer.class);
                                break;
                            case OUTPUT_LAYER:
                                outputs = GSON.fromJson(in, NetworkLayer.class);
                                break;
                            case HIDDEN_LAYERS:
                                hiddenLayers = GSON.fromJson(in, NetworkLayer[].class);
                                break;
                            default:
                                if(!in.isLenient()) throw new JsonParseException("Parser encountered an unexpected field. (" + var + ")");
                                in.skipValue();
                                break;
                        }
                        break;
                    default:
                        if(!in.isLenient()) throw new JsonParseException("Parser encountered an unexpected token. (" + in.peek() + ")");
                        in.skipValue();
                        break;
                }
            }
            in.endObject();

            if(inputs == null || outputs == null || hiddenLayers == null){
                ArrayList<String> missing = new ArrayList<>(3);
                if (inputs == null) missing.add("inputs");
                if (outputs == null) missing.add("outputs");
                if (outputs == null) missing.add("hiddenLayers");
                throw new JsonParseException(JsonParsingTools.missingFields(NetworkLayout.class.getSimpleName(), missing.toArray(String[]::new)));
            }
         
            return new NetworkLayout(inputs, outputs, hiddenLayers);

        } catch (Exception e) {
            e.printStackTrace();
            if(in.isLenient()) return null;
            throw new JsonParseException("Unable to construct " + NetworkLayout.class.getName() + " from string.");
        }
    }
    
}
