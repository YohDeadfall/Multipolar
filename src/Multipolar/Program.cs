using Multipolar.Layers;
using Multipolar.Primitives;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using static System.Console;

namespace Multipolar
{
    public static class Program
    {
        static readonly string folder = Environment.ExpandEnvironmentVariables(@"%USERPROFILE%\Downloads\MNIST\");
        const string training_images_file = "train-images.idx3-ubyte";
        const string training_labels_file = "train-labels.idx1-ubyte";
        const string test_images_file = "t10k-images.idx3-ubyte";
        const string test_labels_file = "t10k-labels.idx1-ubyte";

        private static void Test_static_basic()
        {
            var input = new float[] { 0.05f, 0.10f };

            var target = new float[] { 0.01f, 0.99f };

            var eta = 0.5f;

            var fc1 = new FullyConnectedLayer(2, 2);
            fc1.Biases[0] = 0.35f;
            fc1.Biases[1] = 0.35f;
            fc1.Weights[0] = 0.15f;
            fc1.Weights[1] = 0.25f;
            fc1.Weights[2] = 0.20f;
            fc1.Weights[3] = 0.30f;

            var sig1 = new SigmoidLayer(2);

            var fc2 = new FullyConnectedLayer(2, 2);
            fc2.Biases[0] = 0.60f;
            fc2.Biases[1] = 0.60f;
            fc2.Weights[0] = 0.40f;
            fc2.Weights[1] = 0.50f;
            fc2.Weights[2] = 0.45f;
            fc2.Weights[3] = 0.55f;

            var sig2 = new SigmoidLayer(2);

            for (var i = 0; i < 10_000; i++)
            {
                fc1.Feed(input);
                sig1.Feed(fc1.Output);
                fc2.Feed(sig1.Output);
                sig2.Feed(fc2.Output);

                var error = new float[2]
                {
                    sig2.Output[0] - target[0],
                    sig2.Output[1] - target[1],
                };

                sig2.ComputeGradient(error);
                fc2.ComputeGradient(sig2.Gradient);
                sig1.ComputeGradient(fc2.InputGradient);
                fc1.ComputeGradient(sig1.Gradient);

                fc2.Optimize(sig1.Output, eta);
                fc1.Optimize(input, eta);
            }

            WriteLine(sig2.Output[0]);
            WriteLine(sig2.Output[1]);
        }

        private static async Task Test_MNIST_basic()
        {
            var eta = 0.001f;

            var training_images = await IDXFileReader.Load(Path.Combine(folder, training_images_file)) as byte[,,];
            var training_labels = await IDXFileReader.Load(Path.Combine(folder, training_labels_file)) as byte[];
            var test_images = await IDXFileReader.Load(Path.Combine(folder, test_images_file)) as byte[,,];
            var test_labels = await IDXFileReader.Load(Path.Combine(folder, test_labels_file)) as byte[];

            // Build network

            var input = new float[28, 28, 1];

            var fc1 = new FullyConnectedLayer(784, 1024);
            var act1 = new ReLULayer(1024);
            var fc2 = new FullyConnectedLayer(1024, 10);
            var act2 = new SoftmaxLayer(10);

            // Initialize tensors

            var randomWeightDistribution = new NormalDistribution(0, 0.1f);

            fc1.Weights.Fill(randomWeightDistribution);
            fc1.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            fc2.Weights.Fill(randomWeightDistribution);
            fc2.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            // Loop through images to train

            var image_ct = training_images.GetLength(0);
            var batch_sz = 100;
            var batch_ct = image_ct / batch_sz;
            var target = new float[10];

            var stats = (accuracy: 0f, squareLoss: 0f, crossEntropyLoss: 0f);

            for (var i_batch = 0; i_batch < batch_ct; i_batch++)
            {
                if (i_batch > 0)
                {
                    SetCursorPosition(0, CursorTop - 4);
                }

                // Train

                for (var j_batch = 0; j_batch < batch_sz; j_batch++)
                {
                    var i_input = i_batch * batch_sz + j_batch;

                    // Initialize data

                    for (var x = 0; x < 28; x++)
                    {
                        for (var y = 0; y < 28; y++)
                        {
                            input[x, y, 0] = training_images[i_input, x, y] / 255f;
                        }
                    }

                    Array.Clear(target, 0, 10);

                    target[training_labels[i_input]] = 1;

                    // Feed forward

                    fc1.Feed(input);
                    act1.Feed(fc1.Output);
                    fc2.Feed(act1.Output);
                    act2.Feed(fc2.Output);

                    // Calculate loss

                    var prediction = act2.Output;

                    var correct = act2.Output.MaxIndex() == training_labels[i_input];

                    var softmaxLoss = target.Zip(prediction, (p, q) => q - p).ToArray();
                    var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                    var crossEntropyLoss = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                    stats.accuracy += correct ? 1 : 0;
                    stats.squareLoss += squareLoss.Sum();
                    stats.crossEntropyLoss += crossEntropyLoss.Average();

                    // Gradient

                    act2.ComputeGradient(softmaxLoss);
                    fc2.ComputeGradient(act2.InputGradient);
                    act1.ComputeGradient(fc2.InputGradient);
                    fc1.ComputeGradient(act1.InputGradient);

                    // Optimize

                    fc2.Optimize(act1.Output, eta);
                    fc1.Optimize(input, eta);

                    // UI

                    SetCursorPosition(0, CursorTop);
                    Write($"{i_input + 1} training iterations");
                }

                WriteLine();
                WriteLine($"- accuracy        : {(stats.accuracy / batch_sz).ToString("F9")}");
                WriteLine($"- squared error   : {(stats.squareLoss / batch_sz).ToString("F9")}");
                WriteLine($"- x-entropy loss  : {(stats.crossEntropyLoss / batch_sz).ToString("F9")}");

                stats = default;
            }

            WriteLine();

            var testCount = 10_000;

            for (var i = 0; i < testCount; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    for (var k = 0; k < 28; k++)
                    {
                        input[j, k, 0] = test_images[i, j, k] / 255f;
                    }
                }

                fc1.Feed(input);
                act1.Feed(fc1.Output);
                fc2.Feed(act1.Output);
                act2.Feed(fc2.Output);

                // Calculate loss

                Array.Clear(target, 0, 10);
                target[test_labels[i]] = 1;

                var prediction = act2.Output;
                var correct = act2.Output.MaxIndex() == test_labels[i];

                var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                var crossEntropyLoss = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                stats.accuracy += correct ? 1 : 0;
                stats.squareLoss += squareLoss.Sum();
                stats.crossEntropyLoss += crossEntropyLoss.Average();

                // no gradient optimization for testing

                SetCursorPosition(0, CursorTop);
                Write($"{i + 1} test iterations");
            }

            WriteLine();

            WriteLine($"- correctness       : {(stats.accuracy / testCount).ToString("F9")}");
            WriteLine($"- squared error     : {(stats.squareLoss / testCount).ToString("F9")}");
            WriteLine($"- x-entropy loss    : {(stats.crossEntropyLoss / testCount).ToString("F9")}");
        }

        private static async Task Test_MNIST_convolutional(int? train = null, bool test = true)
        {
            var eta = 0.0001f;

            var training_images_task = IDXFileReader.Load(Path.Combine(folder, training_images_file));
            var training_labels_task = IDXFileReader.Load(Path.Combine(folder, training_labels_file));

            var test_images_task = Task.FromResult<Array>(default);
            var test_labels_task = Task.FromResult<Array>(default);

            if (test)
            {
                test_images_task = IDXFileReader.Load(Path.Combine(folder, test_images_file));
                test_labels_task = IDXFileReader.Load(Path.Combine(folder, test_labels_file));
            }

            // Build network

            var input = new float[28, 28, 1];

            var conv1 = new Convolutional2DLayer(28, 28, 1, 5, 5, 32);
            var relu1 = new ReLULayer(28, 28, 32);
            var pool1 = new TwoByTwoPoolingLayer((14, 14, 32));

            var conv2 = new Convolutional2DLayer(14, 14, 32, 5, 5, 64);
            var relu2 = new ReLULayer(14, 14, 64);
            var pool2 = new TwoByTwoPoolingLayer((7, 7, 64));

            var fc1 = new FullyConnectedLayer(7 * 7 * 64, 1024);
            var relu3 = new ReLULayer(1024);

            var dropout = new DropoutLayer(1024, 0.5f);

            var fc2 = new FullyConnectedLayer(1024, 10);
            var softmax = new SoftmaxLayer(10);

            // Initialize tensors

            var randomWeightDistribution = new NormalDistribution(0, 0.1f);

            conv1.Kernel.Fill(randomWeightDistribution);
            conv1.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            conv2.Kernel.Fill(randomWeightDistribution);
            conv2.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            fc1.Weights.Fill(randomWeightDistribution);
            fc1.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            fc2.Weights.Fill(randomWeightDistribution);
            fc2.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            // Loop through images to train

            await Task.WhenAll(training_images_task, training_labels_task);

            var training_images = training_images_task.Result as byte[,,];
            var training_labels = training_labels_task.Result as byte[];

            var input_ct = training_images.GetLength(0);
            var batch_sz = 100;
            var batch_ct = (train ?? input_ct) / batch_sz;
            var target = new float[10];

            var stats = (accuracy: 0f, squareLoss: 0f, crossEntropyLoss: 0f);

            var totalStopwatch = TimeSpan.Zero;
            var batchStopwatch = new Stopwatch();

            for (var i_batch = 0; i_batch < batch_ct; i_batch++)
            {
                if (i_batch > 0)
                {
                    SetCursorPosition(0, CursorTop - 6);
                }

                batchStopwatch.Restart();

                // Train

                for (var j_batch = 0; j_batch < batch_sz; j_batch++)
                {
                    var i_input = i_batch * batch_sz + j_batch;

                    // Initialize data

                    for (var j = 0; j < 28; j++)
                    {
                        for (var k = 0; k < 28; k++)
                        {
                            input[j, k, 0] = training_images[i_input, j, k] / 255f;
                        }
                    }

                    Array.Clear(target, 0, 10);

                    target[training_labels[i_input]] = 1;

                    // Feed forward

                    conv1.Feed(input);
                    relu1.Feed(conv1.Output);
                    pool1.Feed(relu1.Output);

                    conv2.Feed(pool1.Output);
                    relu2.Feed(conv2.Output);
                    pool2.Feed(relu2.Output);

                    fc1.Feed(pool2.Output);
                    relu3.Feed(fc1.Output);

                    dropout.Feed(relu3.Output);

                    fc2.Feed(dropout.Output);
                    softmax.Feed(fc2.Output);

                    // Calculate loss

                    var prediction = softmax.Output;

                    var softmaxLoss = target.Zip(prediction, (p, q) => q - p).ToArray();

                    var correct = softmax.Output.MaxIndex() == training_labels[i_input];
                    var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                    var crossEntropy = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                    stats.accuracy += correct ? 1 : 0;
                    stats.squareLoss += squareLoss.Sum();
                    stats.crossEntropyLoss += crossEntropy.Average();

                    // Gradient

                    softmax.ComputeGradient(softmaxLoss);
                    fc2.ComputeGradient(softmax.InputGradient);

                    dropout.ComputeGradient(fc2.InputGradient);

                    relu3.ComputeGradient(dropout.InputGradient);
                    fc1.ComputeGradient(relu3.InputGradient);

                    pool2.ComputeGradient(fc1.InputGradient);
                    relu2.ComputeGradient(pool2.InputGradient);
                    conv2.ComputeGradient(relu2.InputGradient);

                    pool1.ComputeGradient(conv2.InputGradient);
                    relu1.ComputeGradient(pool1.InputGradient);
                    conv1.ComputeGradient(relu1.InputGradient);

                    // Optimize

                    fc2.Optimize(dropout.Output, eta);
                    fc1.Optimize(pool2.Output, eta);
                    conv2.Optimize(pool1.Output, eta);
                    conv1.Optimize(input, eta);

                    // UI

                    SetCursorPosition(0, CursorTop);
                    Write($"{i_input + 1} training iterations");
                }

                batchStopwatch.Stop();

                totalStopwatch += batchStopwatch.Elapsed;

                WriteLine();
                WriteLine($"- accuracy        : {(stats.accuracy / batch_sz).ToString("F9")}".PadRight(BufferWidth));
                WriteLine($"- squared error   : {(stats.squareLoss / batch_sz).ToString("F9")}".PadRight(BufferWidth));
                WriteLine($"- x-entropy loss  : {(stats.crossEntropyLoss / batch_sz).ToString("F9")}".PadRight(BufferWidth));
                WriteLine($"- elapsed         : {batchStopwatch.Elapsed} (total: {totalStopwatch})".PadRight(BufferWidth));
                WriteLine($"- est. remaining  : {(totalStopwatch / (i_batch + 1)) * (batch_ct - (i_batch + 1))}".PadRight(BufferWidth));

                stats = default;
            }

            WriteLine();

            if (!test)
            {
                return;
            }

            await Task.WhenAll(test_images_task, test_labels_task);

            var test_images = test_images_task.Result as byte[,,];
            var test_labels = test_labels_task.Result as byte[];

            var testCount = 10_000;

            for (var i = 0; i < testCount; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    for (var k = 0; k < 28; k++)
                    {
                        input[j, k, 0] = test_images[i, j, k] / 255f;
                    }
                }

                conv1.Feed(input);
                relu1.Feed(conv1.Output);
                pool1.Feed(relu1.Output);

                conv2.Feed(pool1.Output);
                relu2.Feed(conv2.Output);
                pool2.Feed(relu2.Output);

                fc1.Feed(pool2.Output);
                relu3.Feed(fc1.Output);

                //dropout.Feed(relu3.Output);

                fc2.Feed(relu3.Output);
                softmax.Feed(fc2.Output);

                // Calculate loss

                Array.Clear(target, 0, 10);
                target[test_labels[i]] = 1;

                var prediction = softmax.Output;
                var correct = softmax.Output.MaxIndex() == test_labels[i];

                var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                var crossEntropyLoss = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                stats.accuracy += correct ? 1 : 0;
                stats.squareLoss += squareLoss.Sum();
                stats.crossEntropyLoss += crossEntropyLoss.Average();

                // no gradient optimization for testing

                SetCursorPosition(0, CursorTop);
                Write($"{i + 1} test iterations");
            }

            WriteLine();

            WriteLine($"- accuracy        : {(stats.accuracy / testCount).ToString("F9")}");
            WriteLine($"- squared error   : {(stats.squareLoss / testCount).ToString("F9")}");
            WriteLine($"- x-entropy loss  : {(stats.crossEntropyLoss / testCount).ToString("F9")}");
        }

        private static async Task Test_MNIST_convolutional_codelab(int? train = null, bool test = true)
        {
            var eta = 0.0001f;

            var training_images_task = IDXFileReader.Load(Path.Combine(folder, training_images_file));
            var training_labels_task = IDXFileReader.Load(Path.Combine(folder, training_labels_file));

            var test_images_task = Task.FromResult<Array>(default);
            var test_labels_task = Task.FromResult<Array>(default);

            if (test)
            {
                test_images_task = IDXFileReader.Load(Path.Combine(folder, test_images_file));
                test_labels_task = IDXFileReader.Load(Path.Combine(folder, test_labels_file));
            }

            // Build network

            var input = new float[28, 28, 1];

            var conv1 = new Convolutional2DLayer(28, 28, 1, 5, 5, 4);
            var relu1 = new ReLULayer(28, 28, 4);
            var pool1 = new TwoByTwoPoolingLayer((14, 14, 4));

            var conv2 = new Convolutional2DLayer(14, 14, 4, 5, 5, 8);
            var relu2 = new ReLULayer(14, 14, 8);
            var pool2 = new TwoByTwoPoolingLayer((7, 7, 8));

            var conv3 = new Convolutional2DLayer(7, 7, 8, 4, 4, 12);
            var relu3 = new ReLULayer(7, 7, 12);

            var fc1 = new FullyConnectedLayer(7 * 7 * 12, 200);
            var relu4 = new ReLULayer(200);

            var fc2 = new FullyConnectedLayer(200, 10);
            var softmax = new SoftmaxLayer(10);

            // Initialize tensors

            var randomWeightDistribution = new NormalDistribution(0, 0.1f);

            conv1.Kernel.Fill(randomWeightDistribution);
            conv1.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            conv2.Kernel.Fill(randomWeightDistribution);
            conv2.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            conv3.Kernel.Fill(randomWeightDistribution);
            conv3.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            fc1.Weights.Fill(randomWeightDistribution);
            fc1.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            fc2.Weights.Fill(randomWeightDistribution);
            fc2.Biases.Fill(Enumerable.Repeat(0.1f, int.MaxValue));

            // Loop through images to train

            await Task.WhenAll(training_images_task, training_labels_task);

            var training_images = training_images_task.Result as byte[,,];
            var training_labels = training_labels_task.Result as byte[];

            var imageCount = training_images.GetLength(0);
            var batchSize = 100;
            var batches = (train ?? imageCount) / batchSize;
            var target = new float[10];

            var stats = (accuracy: 0f, squareLoss: 0f, crossEntropyLoss: 0f);

            var totalStopwatch = new Stopwatch();
            var batchStopwatch = new Stopwatch();

            totalStopwatch.Start();

            for (var batch = 0; batch < batches; batch++)
            {
                if (batch > 0)
                {
                    SetCursorPosition(0, CursorTop - 6);
                }

                batchStopwatch.Restart();

                for (var i = batch * batchSize; i < batch * batchSize + batchSize; i++)
                {
                    for (var j = 0; j < 28; j++)
                    {
                        for (var k = 0; k < 28; k++)
                        {
                            input[j, k, 0] = training_images[i, j, k] / 255f;
                        }
                    }

                    conv1.Feed(input);
                    relu1.Feed(conv1.Output);
                    pool1.Feed(relu1.Output);

                    conv2.Feed(pool1.Output);
                    relu2.Feed(conv2.Output);
                    pool2.Feed(relu2.Output);

                    conv3.Feed(pool2.Output);
                    relu3.Feed(conv3.Output);

                    fc1.Feed(relu3.Output);
                    relu4.Feed(fc1.Output);

                    fc2.Feed(relu4.Output);
                    softmax.Feed(fc2.Output);

                    // Calculate loss

                    Array.Clear(target, 0, 10);

                    target[training_labels[i]] = 1;

                    var prediction = softmax.Output;
                    var correct = softmax.Output.MaxIndex() == training_labels[i];

                    var softmaxLoss = target.Zip(prediction, (p, q) => q - p).ToArray();
                    var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                    var crossEntropy = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                    stats.accuracy += correct ? 1 : 0;
                    stats.squareLoss += squareLoss.Sum();
                    stats.crossEntropyLoss += crossEntropy.Average();

                    softmax.ComputeGradient(softmaxLoss);
                    fc2.ComputeGradient(softmax.InputGradient);

                    relu4.ComputeGradient(fc2.InputGradient);
                    fc1.ComputeGradient(relu4.InputGradient);

                    relu3.ComputeGradient(fc1.InputGradient);
                    conv3.ComputeGradient(relu3.InputGradient);

                    pool2.ComputeGradient(conv3.InputGradient);
                    relu2.ComputeGradient(pool2.InputGradient);
                    conv2.ComputeGradient(relu2.InputGradient);

                    pool1.ComputeGradient(conv2.InputGradient);
                    relu1.ComputeGradient(pool1.InputGradient);
                    conv1.ComputeGradient(relu1.InputGradient);

                    fc2.Optimize(relu4.Output, eta);
                    fc1.Optimize(relu3.Output, eta);
                    conv3.Optimize(relu2.Output, eta);
                    conv2.Optimize(relu1.Output, eta);
                    conv1.Optimize(input, eta);

                    SetCursorPosition(0, CursorTop);
                    Write($"{i + 1} training iterations");
                }

                batchStopwatch.Stop();

                WriteLine();
                WriteLine($"- accuracy        : {(stats.accuracy / batchSize).ToString("F9")}");
                WriteLine($"- squared error   : {(stats.squareLoss / batchSize).ToString("F9")}");
                WriteLine($"- x-entropy loss  : {(stats.crossEntropyLoss / batchSize).ToString("F9")}");
                WriteLine($"- elapsed         : {batchStopwatch.Elapsed} (total: {totalStopwatch.Elapsed})");
                WriteLine($"- est. remaining  : {(totalStopwatch.Elapsed / (batch + 1)) * (batches - (batch + 1))}");

                stats = default;
            }

            WriteLine();

            if (!test)
            {
                return;
            }

            await Task.WhenAll(test_images_task, test_labels_task);

            var test_images = test_images_task.Result as byte[,,];
            var test_labels = test_labels_task.Result as byte[];

            var testCount = 10_000;

            for (var i = 0; i < testCount; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    for (var k = 0; k < 28; k++)
                    {
                        input[j, k, 0] = test_images[i, j, k] / 255f;
                    }
                }

                conv1.Feed(input);
                relu1.Feed(conv1.Output);
                pool1.Feed(relu1.Output);

                conv2.Feed(pool1.Output);
                relu2.Feed(conv2.Output);
                pool2.Feed(relu2.Output);

                conv3.Feed(pool2.Output);
                relu3.Feed(conv3.Output);

                fc1.Feed(relu3.Output);
                relu4.Feed(fc1.Output);

                fc2.Feed(relu4.Output);
                softmax.Feed(fc2.Output);

                // Calculate loss

                Array.Clear(target, 0, 10);
                target[test_labels[i]] = 1;

                var prediction = softmax.Output;
                var correct = softmax.Output.MaxIndex() == test_labels[i];

                var squareLoss = target.Zip(prediction, (p, q) => (float)(Math.Pow(p - q, 2) / 2)).ToArray();
                var crossEntropyLoss = target.Zip(prediction, (p, q) => -(float)(p * Math.Log(q))).ToArray();

                stats.accuracy += correct ? 1 : 0;
                stats.squareLoss += squareLoss.Sum();
                stats.crossEntropyLoss += crossEntropyLoss.Average();

                // no gradient optimization for testing

                SetCursorPosition(0, CursorTop);
                Write($"{i + 1} test iterations");
            }

            WriteLine();

            WriteLine($"- accuracy        : {(stats.accuracy / testCount).ToString("F9")}");
            WriteLine($"- squared error   : {(stats.squareLoss / testCount).ToString("F9")}");
            WriteLine($"- x-entropy loss  : {(stats.crossEntropyLoss / testCount).ToString("F9")}");
        }

        private static Task Test_MNIST_convolutional_perf() => Test_MNIST_convolutional(1000, false);

        public static async Task Main(string[] args)
        {
            CursorVisible = false;

            // Used for profiling...
            await Test_MNIST_convolutional_perf();

            //Test_static_basic();

            //await Test_MNIST_basic();

            //await Test_MNIST_convolutional();

            //await Test_MNIST_convolutional_codelab();

            return;
        }
    }
}