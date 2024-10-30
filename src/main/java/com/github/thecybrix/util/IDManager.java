package com.github.thecybrix.util;

import java.util.LinkedList;

public class IDManager {
    private int counter = 0;
    private LinkedList<Integer> availableIDs = new LinkedList<>();
    
    public synchronized int getNextID(){
        return (availableIDs.size() > 0) ? availableIDs.removeFirst() : counter++;
    }

    public synchronized void releaseID(int id){
        availableIDs.addLast(id);
    }

}
