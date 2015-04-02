using CRNNnet;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace CRNN.gui
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            MyMain(Environment.GetCommandLineArgs());
        }

        static void MyMain(string[] args)
        {
            string filename = "";
            if (args.Length != 2)
            {
                Console.Write("Model:");
                filename = Console.ReadLine();
            }
            else
            {
                filename = args[1];
            }
            if (File.Exists(filename))
            {
                var json = File.ReadAllText(filename);
                Network network = new Network(json, "predict");
                while (true)
                {
                    Console.Write("Image:");
                    var imagename = Console.ReadLine();
                    var image = Bitmap.FromFile(imagename);
                    var arr = Utility.ImageToFloatArray(image);
                    network.set_input(arr);
                    var ans = network.forward();
                    string chr = network.translate(ans.arg_max());
                    Console.WriteLine(chr);
                }
                //network.set_input()
                //Network.TrainAndTestNetwork(filename);
            }
            //Application.Run(new MainForm());
        }
    }
}
