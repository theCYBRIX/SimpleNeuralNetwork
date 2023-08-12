package com.mjsd.simpleneuralnetwork.gson;

import java.io.IOException;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import com.mjsd.simpleneuralnetwork.NetworkLayout;
import com.mjsd.simpleneuralnetwork.NetworkLayout.NetworkLayer;

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

            if(inputs == null || outputs == null || hiddenLayers == null)
                throw new JsonParseException("Parser is unable to populate all necessary fields with given string.");
         
            return new NetworkLayout(inputs, outputs, hiddenLayers);

        } catch (Exception e) {
            e.printStackTrace();
            if(in.isLenient()) return null;
            throw new JsonParseException("Unable to construct " + NetworkLayout.class.getName() + " from string.");
        }
    }
    
}
