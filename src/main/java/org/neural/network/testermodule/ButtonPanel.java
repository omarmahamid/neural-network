package org.neural.network.testermodule;

import org.neural.network.mathlib.algebra.Matrix;
import org.neural.network.mathlib.algebra.Vector;
import org.neural.network.neuralnetlib.io.NetworkIO;
import org.neural.network.neuralnetlib.io.Resource;
import org.neural.network.neuralnetlib.options.cost.CrossEntropyCostFunction;
import org.neural.network.neuralnetlib.options.regularization.L2Regularization;
import org.neural.network.neuralnetlib.trainer.StochasticGradientDescentTrainer;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.util.Random;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;

/**
 * Panel holding all the buttons and interacting with neural network.
 *
 * */
public class ButtonPanel extends JPanel {

    private final JLabel[] labels;
    private final Frame frame;
    private final JButton predictButton;
    private StochasticGradientDescentTrainer sgdt;

    /**
     * Initializes all buttons and labels.
     *
     * @param frame the parent frame
     */
    public ButtonPanel(Frame frame) {
        super(new GridLayout(12, 3));
        this.frame = frame;
        this.labels = new JLabel[10];

        for (int i = 0; i < 10; i++) {
            add(new JLabel(i + " ", JLabel.RIGHT));
            labels[i] = new JLabel("", JLabel.CENTER);
            add(labels[i]);
            add(new JLabel("%"));
        }
        JButton resetButton = new JButton("Reset");
        JButton trainButton = new JButton("Train");
        JButton saveButton = new JButton("Save");
        JButton loadButton = new JButton("Load");
        JButton testButton = new JButton("Test");
        predictButton = new JButton("Predict");

        resetButton.addActionListener((ActionEvent ae) -> resetButtonActionPerformed());

        trainButton.addActionListener((ActionEvent ae) -> trainButtonActionPerformed());

        saveButton.addActionListener((ActionEvent ae) -> saveButtonActionPerformed());

        loadButton.addActionListener((ActionEvent ae) -> {
            try {
                loadButtonActionPerformed();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        testButton.addActionListener((ActionEvent ae) -> testButtonActionPerformed());
        predictButton.addActionListener((ActionEvent ae) -> predictButtonActionPerformed());

        add(loadButton);
        add(saveButton);
        add(resetButton);
        add(trainButton);
        add(testButton);
        add(predictButton);

        Graphics2D g = frame.getGraphics2D();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE);
        setLabels(new double[10]);
        frame.repaint();
    }

    /**
     * Trains the network using MNIST data.
     */
    private void trainButtonActionPerformed() {
        if (sgdt == null) {
            sgdt = new StochasticGradientDescentTrainer(frame.getNet(), new CrossEntropyCostFunction(), new L2Regularization());

            sgdt.setTrainingData(MNISTLoader.importData(new Resource("train-images-idx3-ubyte.gz").getAbsoluteFileName()),
                    MNISTLoader.importData(new Resource("train-labels-idx1-ubyte.gz").getAbsoluteFileName()));

            sgdt.setTestData(MNISTLoader.importData(new Resource("t10k-images-idx3-ubyte.gz").getAbsoluteFileName()),
                    MNISTLoader.importData(new Resource("t10k-labels-idx1-ubyte.gz").getAbsoluteFileName()));
        }
        sgdt.train(2, 0.5, 5.0, 10, true);
    }

    /**
     * Testing the network by iterating through some test images and diplaying
     * calculated results.
     */
    private void testButtonActionPerformed() {

        Random rand = new Random();

        Vector[] data = MNISTLoader.importData(new Resource("t10k-images-idx3-ubyte.gz").getAbsoluteFileName());

        new Thread(() -> {
            System.out.println("\n---Testing Network---");
            for (int j = 0; j < data.length; j += rand.nextInt(200) + 400) {
                try {
                    drawImageFromVector(data[j]);
                    Thread.sleep(1000);
                } catch (InterruptedException ex) {
                }
            }
            System.out.println("finished");
            frame.setPredicted(true);
        }).start();
    }

    /**
     * Saves the current neural network.
     */
    private void saveButtonActionPerformed() {
        NetworkIO.saveNetwork(new Resource("network.dat").getAbsoluteFileName(), frame.getNet());
    }

    /**
     * Loads some saved neural network.
     */
    private void loadButtonActionPerformed() throws Exception {
        frame.setNet(NetworkIO.loadNetwork(new Resource("network.dat").getAbsoluteFileName()));
    }

    /**
     * Resets the drawing panel and the labels.
     */
    public void resetButtonActionPerformed() {
        Graphics2D g = frame.getGraphics2D();
        g.setBackground(Color.WHITE);
        g.clearRect(0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE);
        setLabels(new double[10]);
        frame.setPredicted(false);
        frame.repaint();
    }

    /**
     * Sets the probabilities as label content.
     *
     * @param output calculated probabilities
     */
    public void setLabels(double[] output) {
        double sum = 0;
        double max = Double.NEGATIVE_INFINITY;
        for (double d : output) {
            sum += d;
            if (d > max) {
                max = d;
            }
        }
        for (int i = 0; i < 10; i++) {
            if (sum != 0) {
                labels[i].setText(String.format("%.2f", output[i] / sum * 100));
            } else {
                labels[i].setText(String.format("%.2f", output[i] * 100));
            }
            if (output[i] == max && sum != 0) {
                labels[i].setForeground(new Color(0f, 0.8f, 0f));
            } else {
                labels[i].setForeground(Color.BLACK);
            }
        }
    }

    /**
     * Generates a double array representing the drawing. White pixels are
     * encoded as 0. Black pixels are encoded as 1. Grey pixels are values
     * inbetween.
     *
     * @param image drawing to represent
     * @return double array with image data
     */
    public double[] calcInput(BufferedImage image) {
        double[] input = new double[784];
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                input[i * 28 + j] = 1.0 - Math.abs((image.getRGB(j, i) + 1.0) / 16777215.0);
            }
        }
        return input;
    }

    /**
     * Draws image from vector. The vector contains values between 0 and 1. That
     * is convertet into different grey colors and then painted out.
     *
     * @param vector vector to be shown on panel
     */
    private void drawImageFromVector(Vector vector) {
        resetButtonActionPerformed();
        Graphics2D g = frame.getGraphics2D();
        double[] va = vector.getArray();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int grey = 255 - (int) (va[i * 28 + j] * 255);
                g.setColor(new Color(grey, grey, grey, 255));
                g.fillRect(j * 25, i * 25, 25, 25);
            }
        }
        frame.repaint();
        setLabels(frame.getNet().feedforward(new Matrix(vector)).getCols()[0].getArray());
    }

    /**
     * Counts rows of white pixels in current drawing.
     *
     * @param image image to analyze
     * @param i1 start index
     * @param i2 direchtion. either -1 or 1
     * @param horizontal count rows if true. count cols if false.
     * @return number of rows with white pixels
     */
    private int getWhitespaceInImage(BufferedImage image, int i1, int i2, boolean horizontal) {
        int cut = i1;
        for (int i = i1; i >= 0 && i < Frame.DRAW_SIZE; i += i2) {
            boolean white = true;
            for (int j = 0; j < Frame.DRAW_SIZE; j++) {
                if ((horizontal && new Color(image.getRGB(j, i)).getRed() != 255)
                        || (!horizontal && new Color(image.getRGB(i, j)).getRed() != 255)) {
                    white = false;
                    break;
                }
            }

            if (white || i2 < 0) {
                cut = i + 1;
            }
            if (!white) {
                break;
            }
        }
        return cut;
    }

    /**
     * Calculates the center of an image with the highest pixel density.
     *
     * @param image image to analyze
     * @return vector containing y and x value of center
     */
    private Vector centerOfMassOfPixels(BufferedImage image) {
        Vector result = null;
        int iCount = 0, jCount = 0;
        int iVal = 0, jVal = 0;
        for (int i = 0; i < image.getHeight(); i++) {
            for (int j = 0; j < image.getWidth(); j++) {
                if (new Color(image.getRGB(j, i)).getRed() != 255) {
                    iVal += i;
                    jVal += j;
                    iCount++;
                    jCount++;
                }
            }
        }
        if(iCount != 0 && jCount != 0) {
            result = new Vector(iVal / iCount, jVal / jCount);
        }
        return result;
    }

    /**
     * Translates the image towards some point.
     *
     * @param image image to translate
     * @param iC new center y
     * @param jC new center x
     * @param iM image center y
     * @param jM image center x
     */
    private void moveTowardsPoint(BufferedImage image, int iC, int jC, int iM, int jM) {
        int iDiff = iM - iC;
        int jDiff = jM - jC;

        if (iDiff > 0) {
            for (int i = image.getHeight() - 1; i >= 0; i--) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (i - iDiff >= 0) {
                        image.setRGB(j, i, image.getRGB(j, i - iDiff));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        } else if (iDiff < 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (i - iDiff < image.getHeight()) {
                        image.setRGB(j, i, image.getRGB(j, i - iDiff));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        }
        if (jDiff > 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = image.getWidth() - 1; j >= 0; j--) {
                    if (j - jDiff >= 0) {
                        image.setRGB(j, i, image.getRGB(j - jDiff, i));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        } else if (jDiff < 0) {
            for (int i = 0; i < image.getHeight(); i++) {
                for (int j = 0; j < image.getWidth(); j++) {
                    if (j - jDiff < image.getWidth()) {
                        image.setRGB(j, i, image.getRGB(j - jDiff, i));
                    } else {
                        image.setRGB(j, i, Color.WHITE.getRGB());
                    }
                }
            }
        }
    }

    /**
     * Predicts the digit drawn by using the neural network to calculate.
     */
    public void predictButtonActionPerformed() {
        if (frame.isPredicted()) {
            return;
        }
        Thread t1 = new Thread(() -> {
            try {
                int nCut = getWhitespaceInImage(frame.getImage(), 0, 1, true);
                int wCut = getWhitespaceInImage(frame.getImage(), 0, 1, false);
                int eCut = getWhitespaceInImage(frame.getImage(), Frame.DRAW_SIZE - 1, -1, false);
                int sCut = getWhitespaceInImage(frame.getImage(), Frame.DRAW_SIZE - 1, -1, true);

                int width = eCut - wCut;
                int height = sCut - nCut;

                if (width <= 0 || height <= 0) {
                    return;
                }

                BufferedImage cutImage = new BufferedImage(eCut - wCut, sCut - nCut, BufferedImage.TYPE_INT_ARGB_PRE);
                cutImage.getGraphics().drawImage(frame.getImage().getSubimage(wCut, nCut, width, height), 0, 0, null);

                resetButtonActionPerformed();

                int offset = (int) (0.14285714285714285714 * Frame.DRAW_SIZE);
                if (width == height) {
                    frame.getGraphics2D().drawImage(cutImage, offset, offset, Frame.DRAW_SIZE - offset, Frame.DRAW_SIZE - offset, 0, 0, width, height, this);
                } else if (width > height) {
                    double scaledWidth = (Frame.DRAW_SIZE - (2.0 * offset)) / width;
                    int space = (int) ((width - height) * scaledWidth / 2.0);
                    frame.getGraphics2D().drawImage(cutImage, offset, offset + space, Frame.DRAW_SIZE - offset, Frame.DRAW_SIZE - offset - space, 0, 0, width, height, this);
                } else if (width < height) {
                    double scaledHeight = (Frame.DRAW_SIZE - (2.0 * offset)) / height;
                    int space = (int) ((height - width) * scaledHeight / 2.0);
                    frame.getGraphics2D().drawImage(cutImage, offset + space, offset, Frame.DRAW_SIZE - offset - space, Frame.DRAW_SIZE - offset, 0, 0, width, height, this);
                }
                frame.repaint();
                Thread.sleep(400);

                double[] cop = centerOfMassOfPixels(frame.getImage()).getArray();

                moveTowardsPoint(frame.getImage(), (int) cop[0], (int) cop[1], Frame.DRAW_SIZE / 2, Frame.DRAW_SIZE / 2);
                frame.repaint();
                Thread.sleep(400);

                double[] vec = new double[784];
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        int sum = 0;
                        for (int k = 0; k < 25; k++) {
                            for (int l = 0; l < 25; l++) {
                                sum += new Color(frame.getImage().getRGB(j * 25 + k, i * 25 + l)).getRed();
                            }
                        }
                        vec[i * 28 + j] = 1 - sum / 625.0 / 255.0;
                    }
                }
                drawImageFromVector(new Vector(vec));
                frame.setPredicted(true);
            } catch (InterruptedException ex) {
            }
        });
        t1.start();
    }

    /**
     * Gets the predict button.
     *
     * @return predict button
     */
    public JButton getPredictButton() {
        return predictButton;
    }
}