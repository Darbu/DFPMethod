using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DFPMethod
{
    class Program
    {
        static void Main(string[] args)
        {
            int n = 7, M = 1000;
            double eps1 = 1e-3, eps2 = 1e-3;
            DFPMethod dfp_m = new DFPMethod();
            Func<Matrix<double>, double> aimFunc = delegate (Matrix<double> x)
            {
                return (x[0, 0] - 5) * (x[0, 0] - 5) + (x[1, 0] - 3) * (x[1, 0] - 3) + (x[2, 0] - 1) * (x[2, 0] - 1) + (x[3, 0] - 4) * (x[3, 0] - 4) + (x[4, 0] - 4) * (x[4, 0] - 4) + (x[5, 0] - 5) * (x[5, 0] - 5) + (x[6, 0] - 6) * (x[6, 0] - 6);
            };
            Matrix<double> x_0 = CreateMatrix.Dense<double>(n, 1);
            Matrix<double> result = CreateMatrix.Dense<double>(n, 1);

            Stopwatch sw = new Stopwatch();
            sw.Start();
            result = dfp_m.DavidonFletcherPowellMethod(aimFunc, x_0, eps1, eps2, M, n);
            sw.Stop();
            TimeSpan ts = sw.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
            ts.Hours, ts.Minutes, ts.Seconds,
            ts.Milliseconds / 10);
            Console.WriteLine("RunTime " + elapsedTime);
            Console.WriteLine("Answer");
            for (int i = 0; i < n; i++)
                Console.WriteLine(result[i, 0]);
        }
    }
}
