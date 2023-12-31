package org.neural.network.testermodule;

import java.awt.Dimension;
import java.awt.Graphics;
import javax.swing.JPanel;

/**
 * The panel where drawing is performed.
 *
 */
public class DrawPanel extends JPanel {

    private final Frame frame;

    /**
     * Initializes drawing space.
     *
     * @param frame parent frame
     */
    public DrawPanel(Frame frame) {
        super();
        this.frame = frame;
        setPreferredSize(new Dimension(Frame.DRAW_SIZE, Frame.DRAW_SIZE));

        addMouseListener(frame.getMouse());
        addMouseMotionListener(frame.getMouse());
    }

    /**
     * Paints the panel on screen.
     *
     * @param grphcs graphics to paint at
     */
    @Override
    protected void paintComponent(Graphics grphcs) {
        super.paintComponent(grphcs);
        grphcs.drawImage(frame.getImage(), 0, 0, Frame.DRAW_SIZE, Frame.DRAW_SIZE, this);
    }
}