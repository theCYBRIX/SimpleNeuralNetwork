package com.github.thecybrix.simpleneuralnetwork.util;

public class StopWatch {
    final private static float NANO_PER_SEC = 1_000_000_000.0f;
    final private static float NANO_PER_MILLIS = 1_000_000.0f;

    private boolean running = false;
    private long startTime;
    private long endTime;
    private long duration;

    public void start() throws IllegalStateException {
        if(running)
            throw new IllegalStateException("StopWatch is already running.");
        
        startTime = System.nanoTime();
        running = true;
    }

    public void stop() throws IllegalStateException {
        if(!running)
            throw new IllegalStateException("StopWatch is not running.");
        
        endTime = System.nanoTime();
        running = false;
        duration = endTime - startTime;
    }

    public boolean isRunning() {
        return running;
    }

    private long getDuration(){
        if(running)
            duration = System.nanoTime() - startTime;
        return duration;
    }

    public long getSeconds(){
        return (long)(getDuration() / NANO_PER_SEC);
    }

    public long getMillis(){
        return (long)(getDuration() / NANO_PER_MILLIS);
    }

    public long getNano(){
        return getDuration();
    }

    public double getSecondsExact(){
        return getDuration() / NANO_PER_SEC;
    }

    public double getMillisExact(){
        return getDuration() / NANO_PER_MILLIS;
    }
}
