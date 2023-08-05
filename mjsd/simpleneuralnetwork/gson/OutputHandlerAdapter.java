package mjsd.simpleneuralnetwork.gson;

import java.io.IOException;

import com.google.gson.JsonParseException;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import mjsd.simpleneuralnetwork.SimpleNeuralNetwork.OutputHandler;

public class OutputHandlerAdapter extends TypeAdapter<OutputHandler>{

    @Override
    public void write(JsonWriter out, OutputHandler value) throws IOException {
        out.value(value.toString());
    }

    @Override
    public OutputHandler read(JsonReader in) throws IOException {
        try {
            String providerName = in.nextString();
            if(providerName.equals(OutputHandler.NO_HANDLER_STRING)) return OutputHandler.NO_HANDLER;
            throw new JsonParseException("Unknown InputProvider. (" + providerName + ")");
        } catch (Exception e) {
            if(in.isLenient()) return null;
            throw new JsonParseException(e);
        }
    }

}
