import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.logging.ConsoleHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.api.JsonAPIServiceFactory;
import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizers;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.server.JsonIOHandler;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleTCPServer;

public class APIServerTest extends TestingTools {

    private static int port = 3050;
    private static int numInputs = 14,
                       numOutputs = 4,
                       numHiddenLayers = 1,
                       numNodesPerLayer = 28;

    private static SimpleTCPServer trainingServer;
    final private static String LOGGING_LEVELS = "{SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST}";

    public static void main(String[] args) {

        MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder();
        builder.withInputLayer(numInputs, InputNormalizers.BATCH)
               .withOutputLayer(numOutputs)
               .addHiddenLayers(numHiddenLayers, numNodesPerLayer, ActivationFunctions.ReLU);
        
        trainingServer = JsonAPIServiceFactory.createTCPServer(port, builder);
        trainingServer.start(SimpleTCPServer.class.getSimpleName());

        BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
        boolean keepActive = true;
        
        try {
            while(keepActive){
                String line = console.readLine();
                if (line == null || line.trim().isEmpty()) {
                    continue;
                }
                String[] input = line.trim().split("\\s+");
                println(Arrays.toString(input));
                switch (input[0]) {
                    case "exit":
                        keepActive = false;
                        break;

                    case "logging":
                        try {
                            if(input.length < 2 || input.length > 2){
                                println("Invalid arguments. Usage: logging " + LOGGING_LEVELS);
                                break;
                            }
                            String levelString = input[1].toUpperCase();
                            println(levelString);
                            Level level = Level.parse(levelString);
                            Logger logger = Logger.getLogger(SimpleTCPServer.class.getName());
                            if(logger.getHandlers().length > 0){
                                for(Handler handler : logger.getHandlers()){
                                    if(handler instanceof ConsoleHandler){
                                        handler.setLevel(level);
                                    }
                                }
                            }
                            logger = Logger.getLogger(JsonIOHandler.class.getName());
                            if(logger.getHandlers().length > 0){
                                for(Handler handler : logger.getHandlers()){
                                    if(handler instanceof ConsoleHandler){
                                        handler.setLevel(level);
                                    }
                                }
                            }
                            println("Logging level set to \"" + levelString + "\"");
                            
                        } catch (Exception e) {
                            println("Invalid arguments. " + e.getMessage() + "\nUsage: logging " + LOGGING_LEVELS);
                        }
                        break;
                    
                    case "clear":
                        clrscr();
                        break;
                
                    default:
                        println("Unknown command: \"" + String.join(" ", input) + "\"");
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            trainingServer.stop();
        }
        
    }
    
    //Source: https://stackoverflow.com/a/38365871
    public static void clrscr(){
        try {
            if (System.getProperty("os.name").contains("Windows"))
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            else
                Runtime.getRuntime().exec(new String[]{"clear"});
        } catch (IOException | InterruptedException ex) {}
    }
}
