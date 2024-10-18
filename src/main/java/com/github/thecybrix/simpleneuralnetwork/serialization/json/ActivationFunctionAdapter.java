package com.github.thecybrix.simpleneuralnetwork.serialization.json;

import java.io.IOException;

import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunction;
import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions.Softmax;
import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

public class ActivationFunctionAdapter extends TypeAdapter<ActivationFunction>{

    @Override
    public void write(JsonWriter out, ActivationFunction value) throws IOException {
        out.value(value.toString());
    }

    @Override
    public ActivationFunction read(JsonReader in) throws IOException {
        String functionName = in.nextString();
        try{
            if(functionName.equals(Softmax.FUNCTION_NAME)) return new Softmax();
            return ActivationFunctions.valueOf(functionName);

        } catch (Exception e){
            if(in.isLenient()) return functionName.equalsIgnoreCase(Softmax.FUNCTION_NAME) ? new Softmax() : null;
            throw new JsonParseException("Parser failed to produce " + ActivationFunction.class.getName() + " from string.", e); 
        }
    }

}
