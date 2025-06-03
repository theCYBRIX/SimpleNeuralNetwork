package com.github.thecybrix.simpleneuralnetwork.util;

import java.io.IOException;
import java.util.Objects;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class LoggingHelper {

    final private Logger logger;

    private ConsoleHandler consoleHandler = null;
    private FileHandler fileHandler = null;

    public LoggingHelper(Logger logger) throws NullPointerException{
        this.logger = Objects.requireNonNull(logger, "Logger is null.");
    }

    public LoggingHelper(String loggerName) throws NullPointerException{
        this.logger = Logger.getLogger(loggerName);
    }

    public void enableConsoleLogging(Level level){
        enableConsoleLogging(level, null);
    }

    public void enableConsoleLogging(Level level, Formatter formatter){
        consoleHandler = new ConsoleHandler();
        consoleHandler.setFormatter(formatter == null ? new PrintlnFormatter() : formatter);
        consoleHandler.setLevel(level);
        logger.addHandler(consoleHandler);
    }

    public void disableConsoleLogging(){
        if(consoleHandler != null) {
            logger.removeHandler(consoleHandler);
            consoleHandler = null;
        }
    }

    public void enableFileLogging(String filePath, boolean append, Level level) throws IOException, SecurityException{
        enableFileLogging(filePath, append, level, null);
    }

    public void enableFileLogging(String filePath, boolean append, Level level, Formatter formatter) throws IOException, SecurityException{
        fileHandler = new FileHandler(filePath, append);
        fileHandler.setFormatter(formatter == null ? new SimpleFormatter() : formatter);
        fileHandler.setLevel(level);
        logger.addHandler(fileHandler);
    }

    public void disableFileLogging() {
        if(fileHandler != null) {
            logger.removeHandler(fileHandler);
            fileHandler = null;
        }
    }

    public void setLevel(Level level){
        logger.setLevel(level);
    }

    public void setConsoleLevel(Level level) throws IllegalStateException{
        if(consoleHandler == null) throw new IllegalStateException("Console handler is not enabled.");
        consoleHandler.setLevel(level);
    }

    public void setFileLevel(Level level){
        if(fileHandler == null) throw new IllegalStateException("File handler is not enabled.");
        fileHandler.setLevel(level);
    }

    public ConsoleHandler getConsoleHandler() {
        return consoleHandler;
    }

    public FileHandler getFileHandler() {
        return fileHandler;
    }

    static class PrintlnFormatter extends Formatter{
        @Override
        public String format(LogRecord record) {
            return record.getMessage() + "\n";
        }
    }
}
