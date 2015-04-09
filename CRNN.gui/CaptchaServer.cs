using CRNNnet;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;

namespace CRNN.gui
{
    public class CaptchaServer
    {
        HttpListener _listener;
        CaptchaEngine _engine;

        public CaptchaServer(CaptchaEngine engine)
        {
            _listener = new HttpListener();
            _engine = engine;
        }

        public void Start(int port)
        {
            _listener.Prefixes.Add(String.Format("http://*:{0}/", port));
            _listener.Start();
            Console.WriteLine("Serve on port {0}.", port);
            while (true)
            {
                //single thread
                try
                {
                    var ctx = _listener.GetContext();
                    HandleSafe(ctx);
                }
                catch { }
                //ThreadPool.QueueUserWorkItem(new WaitCallback
                //(x => HandleSafe(x as HttpListenerContext)), ctx);
            }
        }

        private void HandleSafe(HttpListenerContext ctx)
        {
            try { Handle(ctx); }
            catch (Exception e)
            {
                using (var sw = new StreamWriter(ctx.Response.OutputStream))
                {
                    sw.WriteLine("{0}", e.Message);
                }
            }
            finally
            {
                try { ctx.Response.Close(); }
                catch { }
            }
        }

        private void Handle(HttpListenerContext ctx)
        {
            using (var stream = ctx.Request.InputStream)
            {
                using (var image = Bitmap.FromStream(stream))
                {
                    var captcha = _engine.ReadCaptcha(image);
                    using (var sr = new StreamWriter(ctx.Response.OutputStream))
                    {
                        sr.WriteLine(captcha.ToString());
                    }
                }
            }
        }
    }
}
