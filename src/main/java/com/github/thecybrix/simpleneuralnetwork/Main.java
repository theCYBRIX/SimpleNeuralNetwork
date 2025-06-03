package com.github.thecybrix.simpleneuralnetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.Callable;
import java.util.logging.ConsoleHandler;
import java.util.logging.Formatter;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;

import com.github.thecybrix.simpleneuralnetwork.api.JsonAPIServiceFactory;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleStdioServer;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleTCPServer;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.ITypeConverter;
import picocli.CommandLine.Option;

@Command(name = "simpleneuralnetwork", mixinStandardHelpOptions = true, version = "Neural Framework 1.0", description = "Runs the neural framework in the specified mode.")
public class Main implements Callable<Integer> {

    enum Mode {
        TCP,
        STDIO
    }

    final private static int DEFAULT_PORT = 3050; 
    final private static String LOGGING_LEVELS = "SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST";

    final private static int ERR_INVALID_MODE = 2;
    
    @Option(names = {"-m", "--mode"}, description = "Mode to run: stdio or tcp.", converter=ModeConverter.class)
    private Mode mode = Mode.TCP;

    @Option(names = {"-p", "--port"}, description = "Port number on which to expose the TCP server.")
    private int port = DEFAULT_PORT;


    public static void main(String[] args) {
        CommandLine cmdLine = new CommandLine(new Main());
        int exitCode = cmdLine.execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        try {
            return handleMode(mode);
        } catch (Exception e) {
            System.err.println("An unexpected error occurred:");
            e.printStackTrace(System.err);
            return 1;
        }
    }

    private int handleMode(Mode mode) throws Exception {
        switch (mode) {
            case TCP:
                runTCPServer(port);
                return 0;
            
            case STDIO:
                runStdioServer();
                return 0;
        
            default:
                System.err.printf("Error: Invalid mode '%s'. Must be 'tcp' or 'stdio'.\n", mode);
                return ERR_INVALID_MODE;
        }
    }



    private static void runStdioServer() throws Exception {
        SimpleStdioServer stdioServer = JsonAPIServiceFactory.createStdioServer(new MutableNeuralNetworkBuilder());
        stdioServer.run();
    }

    private static void runTCPServer(int port) throws Exception {
        Logger logger = Logger.getLogger("");
        for(Handler handler : logger.getHandlers())
            logger.removeHandler(handler);

        ConsoleHandler consoleHandler = new ConsoleHandler();
        consoleHandler.setFormatter(new Formatter() {
            @Override
            public String format(LogRecord record) {
                return record.getMessage() + "\n";
            }
        });
        logger.addHandler(consoleHandler);
        logger.setLevel(Level.INFO);
        
        MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder();
        SimpleTCPServer server = JsonAPIServiceFactory.createTCPServer(port, builder);
        server.start(SimpleTCPServer.class.getSimpleName());

        BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
        boolean keepActive = true;
        
        try {
            while(keepActive){
                String line = console.readLine();
                if (line == null || line.trim().isEmpty()) {
                    continue;
                }
                String[] input = line.trim().split("\\s+");
                switch (input[0]) {
                    case "exit":
                        keepActive = false;
                        break;

                    case "logging":
                        if(logger == null){
                            println("There is no logger instantiated.");
                            break;
                        }

                        if(input.length == 1){
                            Level loggingLevel = logger.getLevel();
                            println(loggingLevel == null ? "[logging disabled]" : loggingLevel.getName());
                            break;
                        }

                        if(input.length > 2){
                            println("Too many arguments. Usage: 'logging {level}'. Valid levels are: " + LOGGING_LEVELS + "\n");
                            break;
                        }

                        String levelString = input[1].toUpperCase();
                        Level level;
                        
                        try{
                            level = Level.parse(levelString);
                        } catch (IllegalArgumentException e) {
                            println("Invalid logging level: '" + levelString + "'. Valid levels are: " + LOGGING_LEVELS + "\n");
                            break;
                        }

                        logger.setLevel(level);
                        
                        println("Logging level set to: '" + levelString + "'");
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
            server.stop();
        }
        
    }

    static void println(String string){
        System.out.println(string);
    }

    static void println(Object obj){
        println(obj.toString());
    }
    
    public static void clrscr(){
        try {
            if (System.getProperty("os.name").contains("Windows"))
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            else
                Runtime.getRuntime().exec(new String[]{"clear"});
        } catch (IOException | InterruptedException ex) {}
    }

    public static class ModeConverter implements ITypeConverter<Mode> {
        public ModeConverter(){}

        @Override
        public Mode convert(String value) throws Exception {
            return Mode.valueOf(value.toUpperCase());
        }
    }
}
