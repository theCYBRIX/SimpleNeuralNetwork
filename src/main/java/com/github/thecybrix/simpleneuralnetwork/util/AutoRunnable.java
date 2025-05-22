package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ThreadFactory;

public interface AutoRunnable extends Runnable {
	
    default public Thread start() {
		return AutoRunnable.start(this, null);
	}
	
    default public Thread start(String threadName) {
		return AutoRunnable.start(this, threadName);
	}
    
	default public Thread start(ThreadFactory threadFactory) throws NullPointerException {
		Thread newThread = Objects.requireNonNull(threadFactory, "ThreadFactory is null.").newThread(this);
		newThread.start();
		return newThread;
	}

	public void stop();

	public boolean isRunning();

	public Optional<Thread> getRunningThread();
    
	private static Thread start(Runnable task, String name){
		Thread newThread = (name != null) ? new Thread(task, name) : new Thread(task);
		newThread.start();
		return newThread;
	}
}
