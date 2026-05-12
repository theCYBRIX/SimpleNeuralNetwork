package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.HashMap;
import java.util.Map;

import com.github.thecybrix.simpleneuralnetwork.util.Endianness;

public class SessionContext {
    private Map<Class<?>, SessionComponent> sessionComponents = new HashMap<>();
    private Endianness endianness = Endianness.BIG_ENDIAN;

    public void add(SessionComponent component) throws IllegalStateException {
        if (sessionComponents.containsKey(component.getClass())){
            throw new IllegalStateException("SessionComponent of type " + component.getClass() + " is already defined.");
        }

        sessionComponents.put(component.getClass(), component);
    }

    public <E extends SessionComponent> E get(Class<E> type) throws IllegalStateException {
        if (!sessionComponents.containsKey(type)){
            throw new IllegalStateException("Component type is not defined");
        }

        return type.cast(sessionComponents.get(type));
    }

    /**
     * Removes the specified component from the context.
     * @param component The component to remove
     * @return {@code true} if the component was present, otherwise {@code false}
     */
    public boolean remove(SessionComponent component) {
        return (sessionComponents.remove(component.getClass()) != null);
    }

    public Endianness getEndianness(){
        return endianness;
    }

    public void setEndianness(Endianness endianness) {
        this.endianness = endianness;
    }
}
