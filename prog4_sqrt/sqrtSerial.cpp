#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>

void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void sqrtAVX2(int N, float initialGuess, float values[], float output[])
{
    // __m256 kThreshold = _mm256_set1_ps(0.00001f);
    // __m256 one = _mm256_set1_ps(1.f);
    // __m256 three = _mm256_set1_ps(3.f);
    // __m256 zpf = _mm256_set1_ps(0.5f);
    // __m256 sign_bit = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < N; i+=8)
    {
        __m256 x = _mm256_load_ps(&values[i]);
        __m256 res = _mm256_sqrt_ps(x);
        _mm256_storeu_ps(&output[i], res);
    }
}

