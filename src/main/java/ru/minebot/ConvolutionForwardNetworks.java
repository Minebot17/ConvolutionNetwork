package ru.minebot;

import ru.minebot.Convolution.ConvolutionNetwork;
import ru.minebot.Convolution.FloatMatrix;
import ru.minebot.Forward.ForwardNetwork;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class ConvolutionForwardNetworks {
    private ConvolutionNetwork convolutionNetwork;
    private ForwardNetwork forwardNetwork;

    public ConvolutionForwardNetworks(int[] firstSize, int[] filtersCount, int filtersSize, int padding, int stride, float learnSpeed, float momentum, int[] layersNeuronsCount, boolean withBias){
        convolutionNetwork = new ConvolutionNetwork(firstSize, filtersCount, filtersSize, padding, stride, learnSpeed, momentum);
        forwardNetwork = Utils.generateSimpleForwardNetwork(learnSpeed, momentum, layersNeuronsCount, withBias);
    }

    public ConvolutionForwardNetworks(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".cfn");
        if (!saveFile.exists())
            throw new IOException("Save don't exists");

        try(ZipInputStream zin = new ZipInputStream(new FileInputStream(saveFile))) {
            ZipEntry entry;
            String fileName;
            Files.createDirectory(Paths.get(directory));
            while ((entry = zin.getNextEntry()) != null) {
                fileName = entry.getName();
                FileOutputStream fout = new FileOutputStream(directory + "/" + fileName);
                for (int c = zin.read(); c != -1; c = zin.read()) {
                    fout.write(c);
                }
                fout.flush();
                zin.closeEntry();
                fout.close();
            }
        }
        catch(Exception ex){
            System.out.println(ex.getMessage());
        }

        convolutionNetwork = new ConvolutionNetwork(name + "/" + name);
        forwardNetwork = new ForwardNetwork(name + "/" + name);
        Utils.delete(new File(directory));
    }

    public void save(String name) throws IOException {
        if (!Files.exists(Paths.get("./saves")))
            Files.createDirectory(Paths.get("./saves"));

        String directory = "./saves/" + name;
        File saveFile = new File( directory + ".cfn");
        if (saveFile.exists())
            saveFile.delete();

        Files.createDirectory(Paths.get(directory));
        convolutionNetwork.save(name + "/" + name);
        forwardNetwork.save(name + "/" + name);

        try(ZipOutputStream zout = new ZipOutputStream(new FileOutputStream(directory + ".cfn"));
            FileInputStream fisInfo = new FileInputStream(directory + "/" + name + ".fn");
            FileInputStream fisWeights = new FileInputStream(directory + "/" + name + ".cn");) {

            ZipEntry infoEntry = new ZipEntry(name + ".fn");
            zout.putNextEntry(infoEntry);
            byte[] buffer0 = new byte[fisInfo.available()];
            fisInfo.read(buffer0);
            zout.write(buffer0);
            zout.closeEntry();

            ZipEntry weightsEntry = new ZipEntry(name + ".cn");
            zout.putNextEntry(weightsEntry);
            byte[] buffer1 = new byte[fisWeights.available()];
            fisWeights.read(buffer1);
            zout.write(buffer1);
            zout.closeEntry();
        }
        Utils.delete(new File(directory));
    }

    public void setInput(List<FloatMatrix> input) throws Exception {
        float[] result = convolutionNetwork.convolution(input);
        forwardNetwork.setInputIn(result);
    }

    public void correctWeight(byte require) throws Exception {
        float[] requireResult = new float[10];
        requireResult[require] = 1;

        convolutionNetwork.correctWeight(forwardNetwork.correctWeight(requireResult));
    }

    public float getMeanSquaredError(byte require) throws Exception {
        float[] result = new float[10];
        result[require] = 1;
        return forwardNetwork.getMeanSquaredError(result);
    }

    public float iterateEra(){
        return forwardNetwork.iterateEra();
    }

    public float[] getResult(){
        return forwardNetwork.getResult();
    }
}
