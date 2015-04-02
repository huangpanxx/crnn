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
            //Application.EnableVisualStyles();
            //Application.SetCompatibleTextRenderingDefault(false);
            //Application.Run(new MainForm());
            MyMain();
        }

        static void MyMain()
        {
            Console.Write("model:");
            string filename = Console.ReadLine();
            if (File.Exists(filename))
            {
                Network.TestNetwork(filename);
            }
        }
    }
}
