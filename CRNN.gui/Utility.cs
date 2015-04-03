using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using CRNNnet;
using System.Drawing;
using System.IO;
using System.Net;

namespace CRNN.gui
{
    public class Utility
    {
        public static FloatArray ImageToFloatArray(Image image)
        {
            const float div = 255.0f * 255.0f;
            using (Bitmap bitmap = new Bitmap(image))
            {
                int row = bitmap.Height, col = bitmap.Width;
                int frame = row * col;
                float[] data = new float[3 * frame];
                for (int r = 0; r < row; ++r)
                {
                    for (int c = 0; c < col; ++c)
                    {
                        var pix = bitmap.GetPixel(c, r);
                        data[0 * frame + r * col + c] = pix.R * pix.A / div;
                        data[1 * frame + r * col + c] = pix.G * pix.A / div;
                        data[2 * frame + r * col + c] = pix.B * pix.A / div;
                    }
                }
                return new FloatArray(new int[] { row, col, 3 }, data);
            }
        }
        public static String PromoteLine(String msg)
        {
            Console.Write(msg + ":");
            return Console.ReadLine();
        }

        public static void Run(string[] args, params Func<string[], bool>[] fns)
        {
            foreach (var fn in fns)
            {
                if (fn(args))
                {
                    break;
                }
            }
        }

        public static String PostData(String url, byte[] data)
        {
            HttpWebRequest req = (HttpWebRequest)HttpWebRequest.Create(url);
            req.Method = "POST";
            using (var stream = req.GetRequestStream())
            {
                stream.Write(data, 0, data.Length);
            }
            using (var reader = new StreamReader(req.GetResponse().GetResponseStream()))
            {
                return reader.ReadToEnd();
            }
        }
    }
}