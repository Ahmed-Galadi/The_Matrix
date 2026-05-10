import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;

public class DrawPredict extends JPanel {
    // Network dimensions
    static final int INPUT = 784;
    static final int HIDDEN = 128;
    static final int OUTPUT = 10;

    // Weights and biases loaded from C++
    static double[][] W1 = new double[HIDDEN][INPUT]; // hidden weights: 128x784
    static double[] B1 = new double[HIDDEN];          // hidden biases
    static double[][] W2 = new double[OUTPUT][HIDDEN];// output weights: 10x128
    static double[] B2 = new double[OUTPUT];          // output biases

    // Drawing area
    private BufferedImage canvas = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY);
    private Graphics2D g2d = canvas.createGraphics();
    private int lastX, lastY;
    private JLabel predictionLabel;
    private boolean drawing = false;

    public DrawPredict() {
        setPreferredSize(new Dimension(280, 280));
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 280, 280);
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(12, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

        addMouseListener(new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                lastX = e.getX();
                lastY = e.getY();
                drawing = true;
            }
            public void mouseReleased(MouseEvent e) {
                drawing = false;
                predict();
            }
        });
        addMouseMotionListener(new MouseMotionAdapter() {
            public void mouseDragged(MouseEvent e) {
                if (drawing) {
                    int x = e.getX(), y = e.getY();
                    g2d.drawLine(lastX, lastY, x, y);
                    lastX = x;
                    lastY = y;
                    repaint();
                }
            }
        });
    }

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(canvas, 0, 0, null);
    }

    // Preprocess drawing and run forward pass
    private void predict() {
    // 1. Find bounding box of drawn digit
    int minX = 280, minY = 280, maxX = 0, maxY = 0;
    boolean hasInk = false;
    for (int y = 0; y < 280; y++) {
        for (int x = 0; x < 280; x++) {
            int rgb = canvas.getRGB(x, y);
            int gray = (rgb >> 16) & 0xff;
            if (gray < 200) { // dark pixel = ink
                hasInk = true;
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (!hasInk) {
        predictionLabel.setText("Predicted: ? (draw something)");
        return;
    }

    // 2. Add padding (20% of bounding box size)
    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    int padX = (int)(width * 0.2);
    int padY = (int)(height * 0.2);
    minX = Math.max(0, minX - padX);
    minY = Math.max(0, minY - padY);
    maxX = Math.min(279, maxX + padX);
    maxY = Math.min(279, maxY + padY);

    // 3. Make the crop square (centered)
    int cropSize = Math.max(maxX - minX + 1, maxY - minY + 1);
    int centerX = (minX + maxX) / 2;
    int centerY = (minY + maxY) / 2;
    int halfSize = cropSize / 2;
    int newMinX = Math.max(0, centerX - halfSize);
    int newMinY = Math.max(0, centerY - halfSize);
    int newMaxX = Math.min(279, newMinX + cropSize - 1);
    int newMaxY = Math.min(279, newMinY + cropSize - 1);
    // Adjust if overflow
    if (newMaxX - newMinX < cropSize - 1) newMinX = Math.max(0, newMaxX - cropSize + 1);
    if (newMaxY - newMinY < cropSize - 1) newMinY = Math.max(0, newMaxY - cropSize + 1);

    // 4. Extract the cropped region and scale to 20x20 (MNIST-style centering)
    BufferedImage cropped = canvas.getSubimage(newMinX, newMinY,
                                  newMaxX - newMinX + 1, newMaxY - newMinY + 1);
    BufferedImage centered = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
    Graphics2D gc = centered.createGraphics();
    gc.setColor(Color.WHITE);
    gc.fillRect(0, 0, 28, 28);
    // Draw the digit in the center 20x20 region
    int innerSize = 20;
    int offsetX = (28 - innerSize) / 2;
    int offsetY = (28 - innerSize) / 2;
    gc.drawImage(cropped, offsetX, offsetY, offsetX + innerSize, offsetY + innerSize,
                 0, 0, cropped.getWidth(), cropped.getHeight(), null);
    gc.dispose();

    // 5. Convert to double array (MNIST format: white digit on black background)
    double[] input = new double[INPUT];
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int rgb = centered.getRGB(x, y);
            int gray = (rgb >> 16) & 0xff;
            // MNIST: 255 = white background, 0 = black ink
            // We want ink = 1.0, background = 0.0
            input[y * 28 + x] = 1.0 - (gray / 255.0);
        }
    }

    // 6. Hidden layer
    double[] hidden = new double[HIDDEN];
    for (int h = 0; h < HIDDEN; h++) {
        double sum = 0;
        for (int i = 0; i < INPUT; i++) {
            sum += input[i] * W1[h][i];
        }
        hidden[h] = Math.max(0, sum + B1[h]); // ReLU
    }

    // 7. Output layer
    double[] logits = new double[OUTPUT];
    double maxLogit = Double.NEGATIVE_INFINITY;
    int predicted = -1;
    for (int o = 0; o < OUTPUT; o++) {
        double sum = 0;
        for (int h = 0; h < HIDDEN; h++) {
            sum += hidden[h] * W2[o][h];
        }
        logits[o] = sum + B2[o];
        if (logits[o] > maxLogit) {
            maxLogit = logits[o];
            predicted = o;
        }
    }

    // 8. Softmax for confidence
    double sumExp = 0;
    for (int o = 0; o < OUTPUT; o++) {
        logits[o] = Math.exp(logits[o] - maxLogit);
        sumExp += logits[o];
    }
    double confidence = logits[predicted] / sumExp * 100;
    predictionLabel.setText(String.format("Predicted: %d (%.1f%%)", predicted, confidence));
}

    // Clear the canvas
    public void clear() {
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 280, 280);
        g2d.setColor(Color.BLACK);
        repaint();
        predictionLabel.setText("Predicted: -");
    }

    // Load a matrix from text file (first line: rows cols)
    private static double[][] loadMatrix(String filename) throws IOException {
        Scanner sc = new Scanner(new File(filename));
        int rows = sc.nextInt();
        int cols = sc.nextInt();
        double[][] mat = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                mat[i][j] = sc.nextDouble();
        sc.close();
        return mat;
    }
    private static double[] loadVector(String filename) throws IOException {
    Scanner sc = new Scanner(new File(filename));
    int rows = sc.nextInt();
    int cols = sc.nextInt();
    int size = rows * cols;                    // 1*128 = 128 or 128*1 = 128
    double[] vec = new double[size];
    for (int i = 0; i < size; i++) {
        vec[i] = sc.nextDouble();
    }
    sc.close();
    return vec;
}

    public static void main(String[] args) throws Exception {
        // Load trained weights from C++ files
        W1 = loadMatrix("hidden_weights.txt");
        B1 = loadVector("hidden_biases.txt");
        W2 = loadMatrix("output_weights.txt");
        B2 = loadVector("output_biases.txt");

        System.out.println("W1 shape: " + W1.length + " x " + W1[0].length);
        System.out.println("W2 shape: " + W2.length + " x " + W2[0].length);
        System.out.println("B1 length: " + B1.length);
        System.out.println("B2 length: " + B2.length);
        // Fix W1 if it's (784 x 128) instead of (128 x 784)
        if (W1.length == INPUT && W1[0].length == HIDDEN) {
            double[][] fixed = new double[HIDDEN][INPUT];
            for (int i = 0; i < HIDDEN; i++)
                for (int j = 0; j < INPUT; j++)
                    fixed[i][j] = W1[j][i];
        W1 = fixed;
        }

        // Fix W2 if it's (128 x 10) instead of (10 x 128)
        if (W2.length == HIDDEN && W2[0].length == OUTPUT) {
            double[][] fixed = new double[OUTPUT][HIDDEN];
            for (int i = 0; i < OUTPUT; i++)
                for (int j = 0; j < HIDDEN; j++)
                    fixed[i][j] = W2[j][i];
            W2 = fixed;
        }
        JFrame frame = new JFrame("Digit Recognizer");
        DrawPredict panel = new DrawPredict();
        JButton clearBtn = new JButton("Clear");
        panel.predictionLabel = new JLabel("Predicted: -", SwingConstants.CENTER);
        clearBtn.addActionListener(e -> panel.clear());
        JPanel bottom = new JPanel(new BorderLayout());
        bottom.add(panel.predictionLabel, BorderLayout.CENTER);
        bottom.add(clearBtn, BorderLayout.EAST);
        frame.add(panel, BorderLayout.CENTER);
        frame.add(bottom, BorderLayout.SOUTH);
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}