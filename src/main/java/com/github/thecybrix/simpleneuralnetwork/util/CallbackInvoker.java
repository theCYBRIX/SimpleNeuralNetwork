package com.github.thecybrix.simpleneuralnetwork.util;

import java.util.List;
import java.util.function.Consumer;

public interface CallbackInvoker<E> {

	public List<Consumer<E>> getCallbackList();

	default public void attachCallback(Consumer<E> callback){
		getCallbackList().add(callback);
	}

	default public boolean detachCallback(Consumer<E> callback){
		return getCallbackList().remove(callback);
	}

	default public void detachAllCallbacks(){
		getCallbackList().clear();
	}

    default public void processCallbacks(E parameter){
        for(Consumer<E> callback : getCallbackList())
            callback.accept(parameter);
    }
}
