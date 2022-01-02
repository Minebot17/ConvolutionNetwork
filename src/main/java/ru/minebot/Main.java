package ru.minebot;

import de.javagl.mnist.reader.MnistDecompressedReader;
import ru.minebot.Convolution.ConvolutionNetwork;
import ru.minebot.Forward.ForwardNetwork;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        ConvolutionForwardNetworks network = null;
        Scanner sc = new Scanner(System.in);
        System.out.println("Load or create new [new/name]:");
        String name = sc.nextLine();
        if (name.equals("new"))
            network = new ConvolutionForwardNetworks(new int[] { 28, 28, 1 }, new int[] { 10, 20 }, 3, 1, 1, 0.001f, 0.6f, new int[]{ 720, 480, 320, 10 }, true);
        else
            network = new ConvolutionForwardNetworks(name);

        System.out.println("Train? [y/n]: ");

        if (sc.nextLine().equals("y")) {
            System.out.println("Fixed eras? [y/n]: ");

            if (sc.nextLine().equals("y")) {
                System.out.println("Era count: ");
                int eraCount = Integer.parseInt(sc.nextLine());

                for (int i = 0; i < eraCount; i++) {
                    iterateEra(network, i);
                }
            }
            else {
                int count = 0;
                System.out.println("Start? [y/n]: ");
                while (sc.nextLine().equals("y")){
                    iterateEra(network, count);
                    System.out.println("Next era? [y/n]: ");
                    count++;
                }
            }

            System.out.println("Train is end. Save? [name/n]: ");
            String saveAnswer = sc.nextLine();
            if (!saveAnswer.equals("n"))
                network.save(saveAnswer);
        }
        while (true) {
            System.out.println("Continue? [y/exit]: ");
            if (sc.nextLine().equals("y")) {
                File inputFile = new File("./input.png");
                if (!inputFile.exists())
                    break;

                BufferedImage image = centerImage(ImageIO.read(inputFile));
                network.setInput(Utils.prepareImage(image));

                float max = -99999;
                int maxIndex = -1;
                float[] result = network.getResult();
                for (int i = 0; i < result.length; i++)
                    if (result[i] > max) {
                        max = result[i];
                        maxIndex = i;
                    }
                System.out.println("Answer: " + maxIndex);
            } else
                break;
        }
    }

    private static BufferedImage centerImage(BufferedImage image){
        int[] coords = new int[4]; // up right down left

        for(int y = 0; y < image.getHeight(); y++) {
            boolean isBreak = false;
            for (int x = 0; x < image.getWidth(); x++)
                if (image.getRGB(x, y) != -1) {
                    coords[0] = y;
                    isBreak = true;
                    break;
                }
            if (isBreak)
                break;
        }

        for(int x = image.getWidth() - 1; x >= 0; x--) {
            boolean isBreak = false;
            for (int y = 0; y < image.getHeight(); y++)
                if (image.getRGB(x, y) != -1) {
                    coords[1] = x;
                    isBreak = true;
                    break;
                }
            if (isBreak)
                break;
        }

        for(int y = image.getHeight() - 1; y >= 0; y--) {
            boolean isBreak = false;
            for (int x = 0; x < image.getWidth(); x++)
                if (image.getRGB(x, y) != -1) {
                    coords[2] = y;
                    isBreak = true;
                    break;
                }
            if (isBreak)
                break;
        }

        for(int x = 0; x < image.getWidth(); x++) {
            boolean isBreak = false;
            for (int y = 0; y < image.getHeight(); y++)
                if (image.getRGB(x, y) != -1) {
                    coords[3] = x;
                    isBreak = true;
                    break;
                }
            if (isBreak)
                break;
        }

        int xDelta = Math.abs(coords[1] - coords[3]);
        int yDelta = Math.abs(coords[2] - coords[0]);
        int xOffset = (image.getWidth() - xDelta)/2;
        int yOffset = (image.getHeight() - yDelta)/2;
        BufferedImage result = new BufferedImage(28, 28, Image.SCALE_DEFAULT);
        for(int x = 0; x < result.getWidth(); x++)
            for(int y = 0; y < result.getHeight(); y++)
                result.setRGB(x, y, 0xffffff);
        for(int x = coords[3]; x <= coords[1]; x++)
            for(int y = coords[0]; y <= coords[2]; y++)
                result.setRGB(xOffset + (x - coords[3]), yOffset + (y - coords[0]), image.getRGB(x, y));
        return result;
    }

    private static void iterateEra(ConvolutionForwardNetworks network, int era) throws Exception {
        MnistDecompressedReader mnistReader = new MnistDecompressedReader();
        mnistReader.readDecompressedTraining(Paths.get("./"), mnistEntry ->
        {
            //System.out.println("Read entry " + mnistEntry);
            BufferedImage image = mnistEntry.createImage();
            try {
                network.setInput(Utils.prepareImage(image));
                network.correctWeight(mnistEntry.getLabel());
                System.out.println("Iteration: " + mnistEntry.getIndex() + " Error: " + (network.getMeanSquaredError(mnistEntry.getLabel()) * 100f) + "%");

            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        System.out.println("Era: " + era + " Error: " + (network.iterateEra() * 100f) + "%");
    }

    private static void readMnist() throws IOException {
        MnistDecompressedReader mnistReader = new MnistDecompressedReader();
        mnistReader.readDecompressedTraining(Paths.get("./"), mnistEntry ->
        {
            //System.out.println("Read entry " + mnistEntry);
            BufferedImage image = mnistEntry.createImage();
            try {
                ImageIO.write(image, "png", new File("./train_images/" + mnistEntry.getIndex() + "_" + mnistEntry.getLabel() + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        mnistReader.readDecompressedTesting(Paths.get("./"), mnistEntry ->
        {
            //System.out.println("Read entry " + mnistEntry);
            BufferedImage image = mnistEntry.createImage();
            try {
                ImageIO.write(image, "png", new File("./test_images/" + mnistEntry.getIndex() + "_" + mnistEntry.getLabel() + ".png"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }
}
