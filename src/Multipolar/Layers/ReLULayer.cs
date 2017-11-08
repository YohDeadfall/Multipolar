using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class ReLULayer
    {
        public readonly int Size;

        public readonly float Factor;

        public readonly float[] Output;

        public readonly float[] InputGradient;

        public ReLULayer(int size, float factor = 0)
        {
            Size = size;
            Factor = factor;
            Output = new float[size];
            InputGradient = new float[size];
        }

        public ReLULayer(int height, int width, int channels, float factor = 0) 
            : this(height * width * channels, factor)
        {
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

        public void ComputeGradient(float[,,] previous)
        {
            Assert(previous.Length == Size);

            fixed (float* p_previous = &previous[0, 0, 0])
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
                    var value = p_input[i];

                    if (value < 0)
                    {
                        p_output[i] = value * Factor;
                    }
                    else
                    {
                        p_output[i] = value;
                    }
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            fixed (float* p_output = &Output[0])
            fixed (float* p_gradient = &InputGradient[0])
            {
                for (var i = 0; i < InputGradient.Length; i++)
                {
                    var value = p_previous[i];

                    if (p_output[i] < 0)
                    {
                        p_gradient[i] = value * Factor;
                    }
                    else
                    {
                        p_gradient[i] = value;
                    }
                }
            }
        }
    }
}
