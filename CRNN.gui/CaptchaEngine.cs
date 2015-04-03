using CRNNnet;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CRNN.gui
{
    public class CaptchaEngine
    {
        Network _network = null;
        public CaptchaEngine()
        {
        }

        public void LoadModel(String json, String plan = "predict")
        {
            _network = new Network(json, plan);
        }

        public Captcha ReadCaptcha(Image image, int maxLen = 10)
        {
            if (_network == null)
            {
                throw new Exception("You must load model first!");
            }
            DateTime now = DateTime.Now;
            //var dims = _network.input_dims();
            //int rows = dims[0], cols = dims[1], channels = dims[2];
            var data = Utility.ImageToFloatArray(image);
            _network.SetInput(data);
            Captcha cap = new Captcha();
            for (int i = 0; i < maxLen; ++i)
            {
                var arr = _network.Forward();
                int k = arr.ArgMax();
                string s = _network.Translate(k);
                if (s == "eof") { break; }
                float prob = arr.At(k);
                cap.Labels.Add(new Captcha.Label(s, prob));
            }
            cap.Time = (float)(DateTime.Now - now).TotalMilliseconds;
            return cap;
        }
    }
}
