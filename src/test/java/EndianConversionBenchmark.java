
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareInputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianAwareOutputStream;
import com.github.thecybrix.simpleneuralnetwork.util.EndianConverter;
import com.github.thecybrix.simpleneuralnetwork.util.Endianness;
import com.github.thecybrix.simpleneuralnetwork.util.StopWatch;

import java.io.*;
import java.util.Random;

public class EndianConversionBenchmark {

    private static final int SAMPLES = 10_000_000;
    private static final Endianness ENDIANNESS = Endianness.LITTLE_ENDIAN;

    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_RESET = "\u001B[0m";

    private static void printResult(String label, double time1, double time2) {
        String fastColor = time1 <= time2 ? ANSI_GREEN : ANSI_RED;
        String slowColor = time1 > time2 ? ANSI_GREEN : ANSI_RED;

        System.out.printf("%-10s Static: %s%.3fms%s  Stream: %s%.3fms%s%n",
                label,
                fastColor, time1, ANSI_RESET,
                slowColor, time2, ANSI_RESET
        );
    }

    public static void main(String[] args) {
        try {
            new EndianConversionBenchmark().benchmarkEndianConversionMethods();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    
    @SuppressWarnings({ "resource", "unused" })
    public void benchmarkEndianConversionMethods() throws IOException {
        Random rand = new Random();

        StopWatch staticTimer = new StopWatch();
        StopWatch streamTimer = new StopWatch();

        // Setup output/input streams
        ByteArrayOutputStream outBuffer = new ByteArrayOutputStream();
        EndianAwareOutputStream streamOut = new EndianAwareOutputStream(outBuffer, ENDIANNESS);

        ByteArrayInputStream inBuffer;
        EndianAwareInputStream streamIn;

        EndianConverter converter = new EndianConverter(ENDIANNESS);

        
        // === INT ===
        staticTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            int val = rand.nextInt();
            byte[] b = converter.intToBytes(val);

            ByteArrayOutputStream tempOut = new ByteArrayOutputStream();
            tempOut.write(b);
            tempOut.flush();

            ByteArrayInputStream tempIn = new ByteArrayInputStream(tempOut.toByteArray());
            byte[] read = tempIn.readNBytes(4);
            int val2 = converter.bytesToInt(read);
        }
        staticTimer.stop();

        streamTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            outBuffer.reset();
            int val = rand.nextInt();
            streamOut.writeInt(val);
            streamOut.flush();

            inBuffer = new ByteArrayInputStream(outBuffer.toByteArray());
            streamIn = new EndianAwareInputStream(inBuffer, ENDIANNESS);
            int val2 = streamIn.readInt();
        }
        streamTimer.stop();

        printResult("int", staticTimer.getMillisExact(), streamTimer.getMillisExact());

        // === LONG ===
        staticTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            long val = rand.nextLong();
            byte[] b = converter.longToBytes(val);

            ByteArrayOutputStream tempOut = new ByteArrayOutputStream();
            tempOut.write(b);
            tempOut.flush();

            ByteArrayInputStream tempIn = new ByteArrayInputStream(tempOut.toByteArray());
            byte[] read = tempIn.readNBytes(8);
            long val2 = converter.bytesToLong(read);
        }
        staticTimer.stop();

        streamTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            outBuffer.reset();
            long val = rand.nextLong();
            streamOut.writeLong(val);
            streamOut.flush();

            inBuffer = new ByteArrayInputStream(outBuffer.toByteArray());
            streamIn = new EndianAwareInputStream(inBuffer, ENDIANNESS);
            long val2 = streamIn.readLong();
        }
        streamTimer.stop();

        printResult("long", staticTimer.getMillisExact(), streamTimer.getMillisExact());

        // === FLOAT ===
        staticTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            float val = rand.nextFloat();
            byte[] b = converter.floatToBytes(val);

            ByteArrayOutputStream tempOut = new ByteArrayOutputStream();
            tempOut.write(b);
            tempOut.flush();

            ByteArrayInputStream tempIn = new ByteArrayInputStream(tempOut.toByteArray());
            byte[] read = tempIn.readNBytes(4);
            float val2 = converter.bytesToFloat(read);
        }
        staticTimer.stop();

        streamTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            outBuffer.reset();
            float val = rand.nextFloat();
            streamOut.writeFloat(val);
            streamOut.flush();

            inBuffer = new ByteArrayInputStream(outBuffer.toByteArray());
            streamIn = new EndianAwareInputStream(inBuffer, ENDIANNESS);
            float val2 = streamIn.readFloat();
        }
        streamTimer.stop();

        printResult("float", staticTimer.getMillisExact(), streamTimer.getMillisExact());

        // === DOUBLE ===
        staticTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            double val = rand.nextDouble();
            byte[] b = converter.doubleToBytes(val);

            ByteArrayOutputStream tempOut = new ByteArrayOutputStream();
            tempOut.write(b);
            tempOut.flush();

            ByteArrayInputStream tempIn = new ByteArrayInputStream(tempOut.toByteArray());
            byte[] read = tempIn.readNBytes(8);
            double val2 = converter.bytesToDouble(read);
        }
        staticTimer.stop();

        streamTimer.start();
        for (int i = 0; i < SAMPLES; i++) {
            outBuffer.reset();
            double val = rand.nextDouble();
            streamOut.writeDouble(val);
            streamOut.flush();

            inBuffer = new ByteArrayInputStream(outBuffer.toByteArray());
            streamIn = new EndianAwareInputStream(inBuffer, ENDIANNESS);
            double val2 = streamIn.readDouble();
        }
        streamTimer.stop();

        printResult("double", staticTimer.getMillisExact(), streamTimer.getMillisExact());
    }
}

