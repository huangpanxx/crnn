using CRNNnet;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace CRNN.gui
{
    public class CaptchaEngine : IDisposable
    {
        Network _network = null;
        public CaptchaEngine()
        {
        }

        public void LoadModel(String json, String plan = "predict")
        {
            _network = new Network(json, plan);
        }

        Tuple<float, List<Captcha.Label>> AddLabel(Tuple<float, List<Captcha.Label>> tp, Captcha.Label label)
        {
            var ls = tp.Item2.ToList();
            ls.Add(label);
            return new Tuple<float, List<Captcha.Label>>(tp.Item1 * label.Prob, ls);
        }

        List<Tuple<float, List<Captcha.Label>>> AddLabel(List<Tuple<float, List<Captcha.Label>>> ls, Captcha.Label label)
        {
            if (ls.Count == 0)
            {
                return new List<Tuple<float, List<Captcha.Label>>>()
                {
                    new Tuple<float,List<Captcha.Label>>(label.Prob, new List<Captcha.Label>()
                    {
                        label
                    })
                };
            }
            return ls.Select(x =>
            {
                if (x.Item2.Last().Name == "eof")
                    return x;
                return AddLabel(x, label);
            }).ToList();
        }
        

        List<Tuple<float, List<Captcha.Label>>> AddLabels(List<Tuple<float, List<Captcha.Label>>> ls, List<Captcha.Label> labels)
        {
            List<Tuple<float, List<Captcha.Label>>> res = new List<Tuple<float, List<Captcha.Label>>>();
            foreach (var label in labels)
            {
                var nl = AddLabel(ls, label);
                res.AddRange(nl);
            }
            return res;
        }


        public List<Captcha> ReadCaptcha(Image image, int size = 5, int top = 5, int maxLen = 10)
        {
            if (_network == null) throw new Exception("You must load model first!");
            lock (_network)
            {
                DateTime now = DateTime.Now;
                using (var data = Utility.ImageToFloatArray(image))
                {
                    _network.SetInput(data);
                    var set = new List<Tuple<float, List<Captcha.Label>>>();

                    for (int i = 0; i < maxLen; ++i)
                    {
                        using (var arr = _network.Forward())
                        {
                            //predict all labels
                            List<int> idxes = Utility.SortByValue(arr);
                            List<Captcha.Label> labels = new List<Captcha.Label>();
                            for (int j = 0; j < Math.Min(top, idxes.Count); ++j)
                            {
                                int k = idxes[j];
                                string s = _network.Translate(k);
                                float prob = arr.At(k);
                                labels.Add(new Captcha.Label(s, prob));
                            }
                            //update set
                            set = AddLabels(set, labels);
                            set.Sort();
                            set.Reverse();
                            set = set.Take(size).ToList();
                            //continue?
                            bool isContinue = false;
                            foreach (var seq in set)
                            {
                                if (seq.Item2.Last().Name != "eof")
                                {
                                    isContinue = true;
                                    break;
                                }
                            }
                            if (!isContinue) break;
                        }
                    }
                    var time = (float)(DateTime.Now - now).TotalMilliseconds;
                    return set.Select(x =>
                    {
                        var cap = new Captcha();
                        cap.Labels.AddRange(x.Item2.Take(x.Item2.Count - 1));
                        cap.Time = time;
                        return cap;
                    }).ToList();
                }
            }
        } 

        public void Dispose()
        {
            if (_network != null)
            {
                _network.Dispose();
            }
        }
    }
}
