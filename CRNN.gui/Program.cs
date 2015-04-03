﻿using CRNNnet;
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
            if (args.Length != 1) return false;
            while (true)
            {
                var filename = Utility.PromoteLine("MODEL");
                Network.TrainAndTestNetwork(filename);
            }
        }

        static bool RunServer(string[] args)
        {
            if (args.Length != 2 || !args[1].EndsWith(".json"))
                return false;
            var filename = args[1];
            var json = File.ReadAllText(filename);
            CaptchaEngine engine = new CaptchaEngine();
            engine.LoadModel(json, "predict");
            CaptchaServer server = new CaptchaServer(engine);
            server.Start(7500);
            return true;
        }
    }
}
