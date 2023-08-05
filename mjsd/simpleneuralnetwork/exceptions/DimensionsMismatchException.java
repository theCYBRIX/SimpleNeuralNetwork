package mjsd.simpleneuralnetwork.exceptions;


public class DimensionsMismatchException extends RuntimeException {
    /**
     * Indicates that the number of layers, or number of nodes per layer between two Neural Networks are not equal.
     */
    public DimensionsMismatchException(){ super(); }
    /**
     * Indicates that the number of layers, or number of nodes per layer between two Neural Networks are not equal.
     */
    public DimensionsMismatchException(String msg){ super(msg); }
}
