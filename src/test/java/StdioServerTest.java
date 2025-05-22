import com.github.thecybrix.simpleneuralnetwork.api.JsonAPIServiceFactory;
import com.github.thecybrix.simpleneuralnetwork.core.MutableNeuralNetworkBuilder;
import com.github.thecybrix.simpleneuralnetwork.server.SimpleStdioServer;

public class StdioServerTest {
    public static void main(String[] args) {
        SimpleStdioServer stdioServer = JsonAPIServiceFactory.createStdioServer(new MutableNeuralNetworkBuilder(), () -> {});
        stdioServer.run();
    }
}
