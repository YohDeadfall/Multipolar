using System;
using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class SigmoidLayer
    {
        public readonly int Size;

        public readonly float[] Output;

        public readonly float[] Gradient;

        public SigmoidLayer(int size)
        {
            Size = size;
            Output = new float[size];
            Gradient = new float[size];
        }

        public void Feed(float[] input)
        {
            Assert(input.Length == Size);

            fixed (float* p_input = &input[0])
            {
                Feed(p_input);
            }
        }

        public void ComputeGradient(float[] previous)
        {
            Assert(previous.Length == Size);

            fixed (float* p_previous = &previous[0])
            {
                ComputeGradient(p_previous);
            }
        }

        private void Feed(float* p_input)
        {
            fixed (float* p_output = &Output[0])
            {
                for (var i = 0; i < Output.Length; i++)
                {
                    p_output[i] = (float)(1 / (1 + Math.Exp(-p_input[i])));
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            fixed (float* p_output = &Output[0])
            fixed (float* p_gradient = &Gradient[0])
            {
                for (var i = 0; i < Gradient.Length; i++)
                {
                    var err_wrt_out = p_previous[i];
                    var out_wrt_in = p_output[i] * (1 - p_output[i]);

                    p_gradient[i] = err_wrt_out * out_wrt_in;
                }
            }
        }
    }
}
