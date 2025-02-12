import java.io.BufferedReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.api.APIIOHandler;
import com.github.thecybrix.simpleneuralnetwork.api.SimpleNNServer;
import com.github.thecybrix.simpleneuralnetwork.core.ActivationFunctions;
import com.github.thecybrix.simpleneuralnetwork.core.InputNormalizers;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetwork;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;

public class TrainingServerTest extends TestingEnvironment {

    private static int port = 3050;
    private static int numInputs = 14,
                       numOutputs = 4,
                       numHiddenLayers = 1,
                       numNodesPerLayer = 28;
                    //    networkSaveCount = 200;

    // private static String saveFile = "TestSaves\\testServerSave.json";

    private static SimpleNNServer<MutableNeuralNetwork> trainingServer;
    final private static String LOGGING_LEVELS = "{SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST}";

    public static void main(String[] args) {

        MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder();
        builder.withInputLayer(numInputs, InputNormalizers.BATCH)
               .withOutputLayer(numOutputs)
               .addHiddenLayers(numHiddenLayers, numNodesPerLayer, ActivationFunctions.ReLU);
        
        trainingServer = new SimpleNNServer<>(port, builder);
        trainingServer.start(SimpleNNServer.class.getSimpleName());

        BufferedReader console = new BufferedReader(System.console().reader());
        boolean keepActive = true;
        
        try {
            while(keepActive){
                String[] input = console.readLine().trim().split(" ");
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
                            Level level = Level.parse(levelString);
                            Logger logger = Logger.getLogger(SimpleNNServer.class.getName());
                            logger.getHandlers()[1].setLevel(level);
                            logger = Logger.getLogger(APIIOHandler.class.getName());
                            logger.getHandlers()[1].setLevel(level);
                            println("Logging level set to \"" + levelString + "\"");
                            
                        } catch (Exception e) {
                            println("Invalid arguments. " + e.getMessage() + "\nUsage: logging " + LOGGING_LEVELS);
                        }
                        break;
                    
                    // case "save":
                    //     try {
                    //         if(input.length > 2){
                    //             println("Invalid arguments. Usage: save [numNetworks]");
                    //             break;
                    //         }
                    //         int numNetworks = (input.length == 1) ? networkSaveCount : Integer.parseInt(input[1]);
                    //         saveToFile(saveFile, CustomGsonFactory.getInstance().toJson(trainingServer.getIoHandler().getBestNetworks(numNetworks), new TypeToken<List<MutableNeuralNetwork>>(){}.getType()));
                    //         println("Successfully saved " + networkSaveCount + " networks to \"" + saveFile + "\"");
                    //     } catch (NumberFormatException e){
                    //         println("Invalid arguments. " + e.getMessage() + "\nUsage: save [numNetworks]");
                    //     } catch (Exception e) {
                    //         e.printStackTrace();
                    //         println("Failed to save networks.");
                    //     }
                        
                    //     break;

                    // case "layout":
                    //     println(NetworkLayout.of(trainingServer.getIoHandler().getBestNetworks(1).get(0)));
                    //     break;
                    
                    case "clear":
                        clrscr();
                        break;
                
                    default:
                        println("Unknown command: \"" + input + "\"");
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
        //Clears Screen in java
        try {
            if (System.getProperty("os.name").contains("Windows"))
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            else
                Runtime.getRuntime().exec(new String[]{"clear"});
        } catch (IOException | InterruptedException ex) {}
    }
}
