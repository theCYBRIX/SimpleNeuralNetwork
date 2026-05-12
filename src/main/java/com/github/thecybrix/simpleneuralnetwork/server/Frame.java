package com.github.thecybrix.simpleneuralnetwork.server;

import java.util.Objects;

public class Frame {
    protected int encodingType;
    protected int requestID;
    protected byte[] payload;

    public Frame() {}

    /**
     * Creates a new Frame with the same properties as the template.
     * @param template
     */
    public Frame(Frame template) throws NullPointerException {
        this(
            Objects.requireNonNull(template, "Template frame is null.").encodingType,
            template.requestID,
            template.payload
        );
    }

    /**
     * Creates a new Frame with the same type and requestID as the template, but with a different specified payload.
     * @param template
     * @param payload The payload to add to this frame.
     * @throws NullPointerException
     */
    public Frame(Frame template, byte[] payload) throws NullPointerException {
        this(Objects.requireNonNull(template, "Template is null.").encodingType, template.requestID);
        this.payload = payload != null ? payload : new byte[0];
    }
    
    public Frame(int type, int requestID) {
        this.encodingType = type;
        this.requestID = requestID;
        this.payload = new byte[0];
    }

    public Frame(int type, int requestID, byte[] payload) {
        this.encodingType = type;
        this.requestID = requestID;
        this.payload = payload;
    }

    public int getEncodingType() {
        return encodingType;
    }

    public int getRequestID() {
        return requestID;
    }
    
    public byte[] getPayload() {
        return payload;
    }
    
}
