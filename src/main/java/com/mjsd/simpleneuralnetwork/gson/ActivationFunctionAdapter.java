package com.mjsd.simpleneuralnetwork.gson;

import java.io.IOException;

import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import com.mjsd.simpleneuralnetwork.ActivationFunctions;
import com.mjsd.simpleneuralnetwork.ActivationFunctions.Softmax;
import com.mjsd.simpleneuralnetwork.SimpleNeuralNetwork.ActivationFunction;

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
