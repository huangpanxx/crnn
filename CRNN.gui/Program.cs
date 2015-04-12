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
            Utility.Run(args, RunServer, TestAndTrain);
        }

        static bool TestAndTrain(string[] args)
        {
            if (args.Length != 2 || !args[1].EndsWith(".json"))
                return false;
            var filename = args[1];
            var json = File.ReadAllText(filename);
            using (Network network = new Network(json, "predict"))
            {
                while (true)
                {
                    Console.Write("IMAGE:");
                    var path = Console.ReadLine();
                    if (!File.Exists(path)) continue;
                    using (var image = Image.FromFile(path))
                    {
                        var data = Utility.ImageToFloatArray(image);
                        network.SetInput(data);
                        for (int i = 0; i < 20; ++i)
                        {
                            using (var signal = network.Forward())
                            {
                                int k = signal.ArgMax();
                                var s = network.Translate(k);
                                if (s == "eof") break;
                                Console.Write(s);
                            }
                        }
                        Console.WriteLine();
                    }
                }
            }
        }

        static bool RunServer(string[] args)
        {
            if (args.Length != 1) return false;
            var filename = Utility.PromoteLine("MODEL");
            var json = File.ReadAllText(filename);
            using (CaptchaEngine engine = new CaptchaEngine())
            {
                engine.LoadModel(json, "predict");
                CaptchaServer server = new CaptchaServer(engine);
                server.Start(7500);
            }
            return true;
        }
    }
}
