import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.awt.image.Kernel;
import java.awt.image.ConvolveOp;

public class DrawPredict extends JPanel {
    // Network dimensions
    static final int INPUT = 784;
    static final int HIDDEN = 128;
    static final int OUTPUT = 10;

    // Weights and biases loaded from C++
    static double[][] W1 = new double[HIDDEN][INPUT];
    static double[] B1 = new double[HIDDEN];
    static double[][] W2 = new double[OUTPUT][HIDDEN];
    static double[] B2 = new double[OUTPUT];

    // Drawing area
    private BufferedImage canvas = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY);
    private Graphics2D g2d = canvas.createGraphics();
    private int lastX, lastY;
    private JLabel predictionLabel;
    private boolean drawing = false;

    // Save components
    private JTextField labelField;
    private JButton saveButton;

    public DrawPredict() {
        setPreferredSize(new Dimension(280, 280));
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 280, 280);
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(30, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));

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

    // Preprocess the canvas exactly as we do for prediction → returns the 28x28 normalized array
    private double[] preprocessCanvas() {
        // 1. Find bounding box
        int minX = 280, minY = 280, maxX = 0, maxY = 0;
        boolean hasInk = false;
        for (int y = 0; y < 280; y++) {
            for (int x = 0; x < 280; x++) {
                int rgb = canvas.getRGB(x, y);
                int gray = (rgb >> 16) & 0xff;
                if (gray < 200) {
                    hasInk = true;
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }
        if (!hasInk) return null;   // nothing drawn

        // 2. Padding (20%)
        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        int padX = (int)(width * 0.2);
        int padY = (int)(height * 0.2);
        minX = Math.max(0, minX - padX);
        minY = Math.max(0, minY - padY);
        maxX = Math.min(279, maxX + padX);
        maxY = Math.min(279, maxY + padY);

        // 3. Square crop (centered)
        int cropSize = Math.max(maxX - minX + 1, maxY - minY + 1);
        int centerX = (minX + maxX) / 2;
        int centerY = (minY + maxY) / 2;
        int halfSize = cropSize / 2;
        int newMinX = Math.max(0, centerX - halfSize);
        int newMinY = Math.max(0, centerY - halfSize);
        int newMaxX = Math.min(279, newMinX + cropSize - 1);
        int newMaxY = Math.min(279, newMinY + cropSize - 1);
        if (newMaxX - newMinX < cropSize - 1) newMinX = Math.max(0, newMaxX - cropSize + 1);
        if (newMaxY - newMinY < cropSize - 1) newMinY = Math.max(0, newMaxY - cropSize + 1);

        // 4. Extract the cropped region and scale to 20x20 with anti‑aliasing
        BufferedImage cropped = canvas.getSubimage(newMinX, newMinY,
                              newMaxX - newMinX + 1, newMaxY - newMinY + 1);
        // Create a temporary high‑quality scaled version
        BufferedImage scaled20 = new BufferedImage(20, 20, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D gs = scaled20.createGraphics();
        gs.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        gs.drawImage(cropped, 0, 0, 20, 20, null);
        gs.dispose();

        // Place into the center of 28x28 white background
        BufferedImage centered = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D gc = centered.createGraphics();
        gc.setColor(Color.WHITE);
        gc.fillRect(0, 0, 28, 28);
        int offsetX = (28 - 20) / 2;   // 4
        int offsetY = (28 - 20) / 2;   // 4
        gc.drawImage(scaled20, offsetX, offsetY, null);
        gc.dispose();

        // // Tiny 3x3 Gaussian blur to mimic MNIST anti‑aliasing
        // float[][] kernelData = {
        //     {0.075f, 0.125f, 0.075f},
        //     {0.125f, 0.200f, 0.125f},
        //     {0.075f, 0.125f, 0.075f}
        // };
        // float[] kernelArray = new float[9];
        // int idx = 0;
        // for (int r = 0; r < 3; r++)
        // for (int c = 0; c < 3; c++)
        //     kernelArray[idx++] = kernelData[r][c];
        // Kernel kernel = new Kernel(3, 3, kernelArray);
        // ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
        // centered = op.filter(centered, null);

        // 5. Convert to double array (ink = 1.0, background = 0.0)
        double[] input = new double[INPUT];
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int rgb = centered.getRGB(x, y);
                int gray = (rgb >> 16) & 0xff;
                input[y * 28 + x] = 1.0 - (gray / 255.0);
            }
        }
        return input;
    }

    private void predict() {
        double[] input = preprocessCanvas();
        if (input == null) {
            predictionLabel.setText("Predicted: ? (draw something)");
            return;
        }

        // Hidden layer
        double[] hidden = new double[HIDDEN];
        for (int h = 0; h < HIDDEN; h++) {
            double sum = 0;
            for (int i = 0; i < INPUT; i++) {
                sum += input[i] * W1[h][i];
            }
            hidden[h] = Math.max(0, sum + B1[h]);
        }

        // Output layer
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

        double sumExp = 0;
        for (int o = 0; o < OUTPUT; o++) {
            logits[o] = Math.exp(logits[o] - maxLogit);
            sumExp += logits[o];
        }
        double confidence = logits[predicted] / sumExp * 100;
        predictionLabel.setText(String.format("Predicted: %d (%.1f%%)", predicted, confidence));
        // Print probabilities for all digits to the console
        System.out.print("Digit probabilities: ");
        for (int o = 0; o < OUTPUT; o++) {
            System.out.printf("%d: %.1f%%  ", o, logits[o] * 100);
        }
        System.out.println();
    }

    // Save the current drawing with the label from the text field
    private void saveDrawing() {
        String labelText = labelField.getText().trim();
        if (labelText.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter a digit (0-9) in the label field.", "No Label", JOptionPane.WARNING_MESSAGE);
            return;
        }
        int label;
        try {
            label = Integer.parseInt(labelText);
            if (label < 0 || label > 9) throw new NumberFormatException();
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Label must be a single digit 0-9.", "Invalid Label", JOptionPane.WARNING_MESSAGE);
            return;
        }

        double[] input = preprocessCanvas();
        if (input == null) {
            JOptionPane.showMessageDialog(this, "Please draw a digit first.", "Empty Canvas", JOptionPane.WARNING_MESSAGE);
            return;
        }

        // Append to mydata.csv
        try (FileWriter fw = new FileWriter("mydata.csv", true);
             PrintWriter pw = new PrintWriter(fw)) {
            pw.print(label);
            for (int i = 0; i < INPUT; i++) {
                pw.print(",");
                pw.print(input[i]);
            }
            pw.println();
            JOptionPane.showMessageDialog(this, "Saved digit " + label + " to mydata.csv");
        } catch (IOException ex) {
            JOptionPane.showMessageDialog(this, "Error saving: " + ex.getMessage(), "Save Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    public void clear() {
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, 280, 280);
        g2d.setColor(Color.BLACK);
        repaint();
        predictionLabel.setText("Predicted: -");
    }

    // ==================== FILE LOADING (unchanged) ====================
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
        int size = rows * cols;
        double[] vec = new double[size];
        for (int i = 0; i < size; i++) {
            vec[i] = sc.nextDouble();
        }
        sc.close();
        return vec;
    }

    public static void main(String[] args) throws Exception {
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

        // Bottom panel: prediction, label input, save, clear
        panel.predictionLabel = new JLabel("Predicted: -", SwingConstants.CENTER);
        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> panel.clear());

        // Label input
        JLabel labelPrompt = new JLabel("Label:");
        panel.labelField = new JTextField("0", 2);   // default to 0
        panel.labelField.setMaximumSize(new Dimension(40, 25));

        JButton saveBtn = new JButton("Save");
        saveBtn.addActionListener(e -> panel.saveDrawing());

        JPanel bottom = new JPanel();
        bottom.setLayout(new FlowLayout(FlowLayout.CENTER, 10, 5));
        bottom.add(panel.predictionLabel);
        bottom.add(labelPrompt);
        bottom.add(panel.labelField);
        bottom.add(saveBtn);
        bottom.add(clearBtn);

        frame.add(panel, BorderLayout.CENTER);
        frame.add(bottom, BorderLayout.SOUTH);
        frame.pack();
        frame.setSize(400, 400); 
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}