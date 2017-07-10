using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DFPMethod
{
    public class DFPMethod
    {
        private double h;
        double maxAbs(Matrix<double> m, int n)
        {
            double max = 0;
            int i;
            for (i = 0; i < n; i++)
                if (Math.Abs(m[i, 0]) > max)
                    max = Math.Abs(m[i, 0]);
            return max;
        }
        //нахождение градиента функции
        //не параллельный вариант
        //x - точка, в которой ищем градиент
        //grad - сюда записываем ответ
        //aimFunc - берем градиент этой функции в точке x
        //n - размерность вектора x
        void gradFunc(Matrix<double> x, int n, Matrix<double> grad, Func<Matrix<double>, double> aimFunc)
        {
            int i; double H = 0.01;
            Matrix<double> tempXAfter = CreateMatrix.Dense<double>(n, 1);
            Matrix<double> tempXBefore = CreateMatrix.Dense<double>(n, 1);
            for (i = 0; i < n; i++)
            {
                x.CopyTo(tempXAfter);
                x.CopyTo(tempXBefore);
                tempXAfter[i, 0] = x[i, 0] + H;
                tempXBefore[i, 0] = x[i, 0] - H;
                grad[i, 0] = (aimFunc(tempXAfter) - aimFunc(tempXBefore)) / (2 * H);
            }
        }

        //нахождение градиента функции
        //параллельный вариант
        //x - точка, в которой ищем градиент
        //grad - сюда записываем ответ
        //aimFunc - берем градиент этой функции в точке x
        void gradFuncParallel(Matrix<double> x, int n, Matrix<double> grad, Func<Matrix<double>, double> aimFunc)
        {
            double H = 0.01;

            Parallel.For(0, n, (i, state) => {
                Matrix<double> tempXAfter = CreateMatrix.Dense<double>(n, 1);
                Matrix<double> tempXBefore = CreateMatrix.Dense<double>(n, 1);

                x.CopyTo(tempXAfter);
                x.CopyTo(tempXBefore);
                tempXAfter[i, 0] = x[i, 0] + H;
                tempXBefore[i, 0] = x[i, 0] - H;
                grad[i, 0] = (aimFunc(tempXAfter) - aimFunc(tempXBefore)) / (2 * H);
            });
        }


        //сделать m единичной матрицей
        //n - размерность матрицы
        void makeIdentity(Matrix<double> m, int n)
        {
            int i, j;
            for (i = 0; i < n; i++)
                for (j = 0; j < n; j++)
                    if (i == j)
                        m[i, j] = 1;
                    else
                        m[i, j] = 0;
        }

        //одномерная минимизация
        //a_k, b_k - границы
        //func - целевая функция
        //l - точность
        double bisectionMethod(double a_k, double b_k, Func<double, double> func, double l)
        {
            int k = 0;
            double L_2k = b_k - a_k;
            double x_c_k, f_x_c_k, y_k, z_k, f_y_k, f_z_k;
            while (L_2k > l)
            {
                x_c_k = (a_k + b_k) / 2;
                f_x_c_k = func(x_c_k);

                y_k = a_k + L_2k / 4;
                z_k = b_k - L_2k / 4;
                f_y_k = func(y_k);
                f_z_k = func(z_k);

                if (f_y_k < f_x_c_k)
                {
                    b_k = x_c_k;
                }
                else
                {
                    if (f_z_k < f_x_c_k)
                    {
                        a_k = x_c_k;
                        x_c_k = z_k;
                    }
                    else
                    {
                        a_k = y_k;
                        b_k = z_k;
                    }
                }
                L_2k = b_k - a_k;
                k = k + 1;
            }
            return (a_k + b_k) / 2;
        }
        /*
         * Метод Девидона-Флетчера-Пауэлла
         * n - кол-во переменных
         * M - максимальное количество итераций
         * aimFunc - целевая функция
         * x_0 - начальная точка
         * eps1 - максимальное значение остановки по градиенту
         * eps2 - максимальное значение остановки по шагу и значению целевой функции
         */
        public Matrix<double> DavidonFletcherPowellMethod(Func<Matrix<double>, double> aimFunc, Matrix<double> x_0, double eps1, double eps2, int M, int n)
        {
            Matrix<double> x_k_plus_1 = null,
                grad_k_plus_1 = CreateMatrix.Dense<double>(n, 1),
                delt_g_k = CreateMatrix.Dense<double>(n, 1),
                delt_x_k = CreateMatrix.Dense<double>(n, 1),
                A_k_c = CreateMatrix.Dense<double>(n, n);

            Matrix<double> res = CreateMatrix.Dense<double>(n, 1);
            int k = 0, commonK = 0;
            double t_k = 0;
            Matrix<double> x_k = CreateMatrix.Dense<double>(n, 1);
            x_0.CopyTo(x_k);
            Matrix<double> A_k = CreateMatrix.DenseIdentity<double>(n);
            Matrix<double> grad_k = CreateMatrix.Dense<double>(n, 1);

            gradFuncParallel(x_k, n, grad_k, aimFunc);

            while (maxAbs(grad_k, n) > eps1)
            {

                if (x_k_plus_1 != null)
                {
                    gradFuncParallel(x_k_plus_1, n, grad_k_plus_1, aimFunc);
                }

                if (maxAbs(grad_k, n) < eps1)
                {
                    res = x_k;
                    break;
                }
                if (commonK >= M)
                {
                    res = x_k;
                    break;
                }
                else if (k == n)
                {
                    A_k = CreateMatrix.DenseIdentity<double>(n);
                    k = 0;
                }
                else if (k >= 1)
                {
                    delt_g_k = grad_k_plus_1 - grad_k;
                    delt_x_k = x_k_plus_1 - x_k;

                    A_k_c = delt_x_k.TransposeAndMultiply(delt_x_k) / delt_x_k.TransposeThisAndMultiply(delt_g_k)[0, 0];

                    A_k_c = A_k_c - A_k.Multiply(delt_g_k).TransposeAndMultiply(delt_g_k).Multiply(A_k) / delt_g_k.TransposeThisAndMultiply(A_k).Multiply(delt_g_k)[0, 0];

                    A_k = A_k + A_k_c;
                }

                Func<double, double> funcForDFP = delegate (double t_k_arg)
                {
                    return aimFunc(x_k - t_k_arg * A_k.Multiply(grad_k));
                };

                if (x_k_plus_1 != null)
                {
                    x_k_plus_1.CopyTo(x_k);
                    grad_k_plus_1.CopyTo(grad_k);
                }

                t_k = bisectionMethod(-1, 10, funcForDFP, 0.0001);

                if (x_k_plus_1 == null)
                {
                    x_k_plus_1 = CreateMatrix.Dense<double>(n, 1);
                }
                x_k_plus_1 = x_k - t_k * A_k.Multiply(grad_k);


                if (maxAbs(x_k_plus_1 - x_k, n) < eps2 && Math.Abs(aimFunc(x_k_plus_1) - aimFunc(x_k)) < eps2)
                {
                    res = x_k_plus_1;
                    break;
                }
                else
                {
                    k++;
                    commonK++;
                }
            }
            res = x_k;
            return res;
        }
    }
}