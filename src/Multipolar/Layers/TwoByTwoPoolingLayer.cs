using static System.Diagnostics.Debug;

namespace Multipolar.Layers
{
    public unsafe class TwoByTwoPoolingLayer
    {
        public readonly (int Height, int Width, int Channels) InputDimensions;

        public readonly (int Height, int Width, int Channels) OutputDimensions;

        public readonly float[,,] Selection;

        public readonly float[,,] Output;

        public readonly float[,,] InputGradient;

        public TwoByTwoPoolingLayer((int height, int width, int channels) outputDimensions)
        {
            var (height, width, channels) = outputDimensions;

            InputDimensions = (height * 2, width * 2, channels);
            OutputDimensions = outputDimensions;
            Selection = new float[height * 2, width * 2, channels];
            Output = new float[height, width, channels];
            InputGradient = new float[height * 2, width * 2, channels];
        }

        public void Feed(float[] input)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0])
            {
                Feed(p_input);
            }
        }

        public void Feed(float[,,] input)
        {
            Assert(input.Length == InputGradient.Length);

            fixed (float* p_input = &input[0, 0, 0])
            {
                Feed(p_input);
            }
        }

        public void ComputeGradient(float[] previous)
        {
            Assert(previous.Length == Output.Length);

            fixed (float* p_previous = &previous[0])
            {
                ComputeGradient(p_previous);
            }
        }

        private void Feed(float* p_input)
        {
            fixed (float* p_selection = Selection)
            fixed (float* p_output = Output)
            {
                var output_size_y = OutputDimensions.Width * OutputDimensions.Channels;
                var output_size_x = OutputDimensions.Channels;

                var input_size_y = InputDimensions.Width * InputDimensions.Channels;
                var input_size_x = InputDimensions.Channels;

                for (var i_output = 0; i_output < Output.Length; i_output++)
                {
                    var output_y = i_output / output_size_y;
                    var output_x = i_output % output_size_y / output_size_x;
                    var output_c = i_output % output_size_x;

                    var i_input0 = (2 * output_y * input_size_y) + (2 * output_x * input_size_x) + (output_c);
                    var i_input1 = i_input0 + input_size_y;
                    var i_input2 = i_input0 + input_size_x;
                    var i_input3 = i_input0 + input_size_y + input_size_x;

                    var selected_input = i_input0;

                    if (p_input[i_input1] > p_input[selected_input])
                    {
                        selected_input = i_input1;
                    }

                    if (p_input[i_input2] > p_input[selected_input])
                    {
                        selected_input = i_input2;
                    }

                    if (p_input[i_input3] > p_input[selected_input])
                    {
                        selected_input = i_input3;
                    }

                    p_selection[i_input0] = 0;
                    p_selection[i_input1] = 0;
                    p_selection[i_input2] = 0;
                    p_selection[i_input3] = 0;
                    p_selection[selected_input] = 1;

                    p_output[i_output] = p_input[selected_input];
                }
            }
        }

        private void ComputeGradient(float* p_previous)
        {
            fixed (float* p_input_g = &InputGradient[0, 0, 0])
            fixed (float* p_selection = &Selection[0, 0, 0])
            fixed (float* p_output = &Output[0, 0, 0])
            {
                for (var i_input = 0; i_input < InputGradient.Length; i_input++)
                {
                    var i_output = i_input / 4;

                    p_input_g[i_input] = p_selection[i_input] * p_previous[i_output] * p_output[i_output];
                }
            }
        }
    }
}
