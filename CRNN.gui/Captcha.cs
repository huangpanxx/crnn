using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CRNN.gui
{
    public class Captcha
    {
        public class Label
        {
            public String Name { get; private set; }
            public float Prob { get; private set; }
            public Label(String name, float prob)
            {
                this.Name = name;
                this.Prob = prob;
            }
        }
        public Captcha()
        {
            this.Labels = new List<Label>();
            this.Time = 0;
        }

        public float Time { get; set; }

        public List<Label> Labels { get; private set; }

        public float Confidence
        {
            get
            {
                float conf = 1.0f;
                foreach (var l in Labels)
                {
                    conf *= l.Prob;
                }
                return conf;
            }
        }

        public override string ToString()
        {
            var code = String.Concat(Labels.Select(x => x.Name));
            float conf = Confidence;
            const String TEMPLATE = "{\"code\":\"{{CODE}}\",\"time\":{{TIME}},\"confidence\":{{CONFIDENCE}}}";
            return TEMPLATE.Replace("{{CODE}}", code)
                .Replace("{{TIME}}", Time.ToString("F3"))
                .Replace("{{CONFIDENCE}}", conf.ToString("F3"));
        } 
    } 
} 
