using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CRNNnet;
using System.Drawing;

namespace CRNN.gui
{
    public class Utility
    {
        public static FloatArray ImageToFloatArray(Image image)
        {
            using (Bitmap bitmap = new Bitmap(image))
            {
                int row = bitmap.Height, col = bitmap.Width;
                int frame = row * col;
                float[] data = new float[row * col * 3];
                for (int r = 0; r < row; ++r)
                {
                    for (int c = 0; c < col; ++c)
                    {
                        var pix = bitmap.GetPixel(c, r);
                        data[0 * frame + r * col + row] = pix.R / 255.0f * pix.A;
                        data[1 * frame + r * col + row] = pix.G / 255.0f * pix.A;
                        data[2 * frame + r * col + row] = pix.B / 255.0f * pix.A;
                    }
                }
                return new FloatArray(new int[] { row, col, 3 }, data);
            }
        }
    }
}