using System;
using System.Collections;
using System.Collections.Generic;

namespace Multipolar.Primitives
{
    public class NormalDistribution : IEnumerable<float>, IEnumerator<float>
    {
        private readonly Random random;

        public float Mean { get; }

        public float StandardDeviation { get; }

        public NormalDistribution(float mean, float standardDeviation)
        {
            Mean = mean;
            StandardDeviation = standardDeviation;

            random = new Random();
        }

        public float Current { get; private set; }

        object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            var u1 = 1.0 - random.NextDouble();
            var u2 = 1.0 - random.NextDouble();
            var s = Math.Sqrt(-2 * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2);

            Current = Mean + StandardDeviation * (float)s;

            return true;
        }

        public void Reset()
        {
            throw new NotSupportedException();
        }

        public void Dispose()
        {
            // no-op
        }

        public IEnumerator<float> GetEnumerator() => this;

        IEnumerator IEnumerable.GetEnumerator() => this;
    }
}
