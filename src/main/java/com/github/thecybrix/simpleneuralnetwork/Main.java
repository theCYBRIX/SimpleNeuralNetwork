package com.github.thecybrix.simpleneuralnetwork;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.OptionalLong;
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

@Command(name = "simpleneuralnetwork", mixinStandardHelpOptions = true, version = "SimpleNeuralNetwork 1.0", description = "Runs the SimpleNerualNetwork in the specified mode.")
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

    @Option(names = {"--parent-pid"}, description = "PID of the Parent process; This process will exit if the parent terminates.", converter=PidConverter.class)
    private OptionalLong parentPid = OptionalLong.empty();

    @Option(names = {"--override-loggers"}, negatable = true, description = { "Enable or disable overrides for console logging handlers.", "Enabled by default." }, converter=PidConverter.class)
    private boolean overrideLoggers = true;

    @Option(names = { "-v", "--verbose" }, description = {
            "Specify multiple -v options to increase verbosity.",
            "For example, `-v -v -v` or `-vvv`" })
    private boolean[] verbosity = new boolean[0];


    private SimpleStdioServer stdioServer;
    private SimpleTCPServer tcpServer;

    public static void main(String[] args) {
        CommandLine cmdLine = new CommandLine(new Main());
        int exitCode = cmdLine.execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() throws Exception {
        parentPid.ifPresent(Main::bindToParentProcess);
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

    private void runStdioServer() throws Exception {
        Logger logger = Logger.getLogger("");
        logger.setLevel(interpretVerbosity(verbosity));
        if(overrideLoggers)
            overrideConsoleHandlers(logger);
        
        stdioServer = JsonAPIServiceFactory.createStdioServer(new MutableNeuralNetworkBuilder());
        stdioServer.run();
    }

    private void runTCPServer(int port) throws Exception {
        Logger logger = Logger.getLogger("");
        logger.setLevel(interpretVerbosity(verbosity));
        if(overrideLoggers)
            overrideConsoleHandlers(logger);
        
        MutableNeuralNetworkBuilder builder = new MutableNeuralNetworkBuilder();
        tcpServer = JsonAPIServiceFactory.createTCPServer(port, builder);
        tcpServer.start(SimpleTCPServer.class.getSimpleName());

        BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
        boolean keepActive = true;

        String helpMessage = String.join("\n\t",
            "\nValid Commands:",
            "close/exit - Closes the application.",
            "logging [level] - Prints the current logging level or sets it if a level is provided.",
            "clear/cls - Clear the terminal.",
            "help/? - Prints the list of valid commands.\n"
        );
        
        try {
            while(keepActive){
                String line = console.readLine();
                if (line == null || line.trim().isEmpty()) {
                    continue;
                }
                String[] input = line.trim().split("\\s+");
                switch (input[0]) {
                    case "close":
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
                            println(loggingLevel == null ? "[undefined]" : loggingLevel.getName());
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
                    
                    case "cls":
                    case "clear":
                        clrscr();
                        break;
                    
                    case "?":
                    case "help":
                        println(helpMessage);
                        break;
                
                    default:
                        println("Unknown command: \"" + String.join(" ", input) + "\"\nUse help/? to get a list of valid commands.\n");
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            tcpServer.stop();
        }
        
    }

    static void println(String string){
        System.err.println(string);
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

    public static Level interpretVerbosity(boolean[] verbosity){
        switch (verbosity.length) {
            case 1: return Level.CONFIG;
            case 2: return Level.FINE;
            case 3: return Level.FINER;
            case 4: return Level.FINEST;
            default: return Level.INFO;
        }
    }

    private static void bindToParentProcess(long parentPid){
        ProcessHandle parent = ProcessHandle.of(parentPid).orElse(null);

        if (parent == null || !parent.isAlive()) {
            System.exit(1);
        }

        parent.onExit().thenRun(() -> {
            System.exit(0);
        });
    }

    private static void overrideConsoleHandlers(Logger logger) {
        for(Handler handler : logger.getHandlers())
            if(handler instanceof ConsoleHandler)
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
    }

    public static class ModeConverter implements ITypeConverter<Mode> {
        public ModeConverter(){}

        @Override
        public Mode convert(String value) throws Exception {
            return Mode.valueOf(value.toUpperCase());
        }
    }

    public static class PidConverter implements ITypeConverter<OptionalLong> {
        public PidConverter(){}

        @Override
        public OptionalLong convert(String value) throws Exception {
            return OptionalLong.of(Long.parseLong(value));
        }
    }
}
