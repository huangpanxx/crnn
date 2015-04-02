using CRNNnet;
using System;
using System.Collections.Generic;
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
                Console.Write("model:");
                filename = Console.ReadLine();
            }
            else
            {
                filename = args[1];
            }
            if (File.Exists(filename))
            {
                Network.TestNetwork(filename);
            }
            //Application.Run(new MainForm());
        }
    }
}
