#include <iostream>
#include <ginac/ginac.h>
#include <array>
#include <functional>

using num_t = GiNaC::numeric;
using ex = GiNaC::ex;

template <unsigned int N>
class Integrator{

    protected:
    // Current step size
    num_t h;

    // Current variable value
    num_t t;

    // Current value of y
    std::array<num_t, N> y;

    // Runge-Kutta-Fehlberg parameters
    std::array<num_t, 6> 
        C{"0", "1/4", "3/8", "12/13", "1", "1/2"},
        B{"16/135", "0", "6656/12825", "28561/56430", "-9/50", "2/55"},
        Bp{"25/216", "0", "1408/2565", "2197/4104", "-1/5", "0"}
    ;

    std::array<std::array<num_t, 5>, 5> A{
        "1/4", "0", "0", "0", "0",
        "3/32", "9/32", "0", "0" ,"0", 
        "1932/2197", "-7200/2197", "7296/2197", "0", "0", 
        "439/216", "-8", "3680/513", "-845/4104", "0", 
        "-8/27", "2", "-3544/2565", "1859/4104", "-11/40"};

    std::array<std::array<num_t, N>, 6> K;

    // The function to evaluate
    std::function<std::array<num_t, N>(num_t, std::array<num_t, N>)> f;

    public:
    Integrator(
            num_t t0,
            std::array<num_t, N> y0,
            std::function<std::array<num_t, N>(num_t, std::array<num_t, N>)> f
            ) 
        : t(t0),
          y(y0),
          f(f)
    {
        h = num_t("0.0001");

        for ( int i = 0; i < C.size(); i++ ) {
            for ( int j = 0; j < N; j++ ) {
                K[i][j] = 0;
            }
        }
    }

    void integrate(num_t tf) {
        num_t rel_err{"1e-5"};

        while ( t <= tf ) {
            std::array<num_t, N> dy, dyp;
            dy.fill("0");
            dyp.fill("0");

            for ( int i = 0; i < C.size(); i++ ) {
                std::array<num_t, N> par;
                par.fill("0");

                for ( int j = 0; j < i; j++ ) {
                    for ( int n = 0; n < N; n++ ) {
                        par[n] += A[i-1][j]*K[j][n];
                    }
                }

                auto curr_y{y};
                for ( int n = 0; n < N; n++ ) {
                    curr_y[n] += par[n]*h;
                }

                K[i] = f(t+C[i]*h, curr_y);

                for ( int n = 0; n < N; n++ ) {
                    dy[n] += B[i]*K[i][n];
                    dyp[n] += Bp[i]*K[i][n];
                }
            }

            for ( int n = 0; n < N; n++ ) {
                dy[n] *= h;
                dyp[n] *= h;
            }

            //for ( auto x: dy ) {
            //    std::cout << x << " ";
            //};
            //std::cout << std::endl;
            t += h;

            std::cout << t << " ";
            for ( int n = 0; n < N; n++ ) {
                y[n] += dy[n];
                std::cout << y[n] << " ";
            }
            std::cout << std::endl;
        }
    }
};

std::array<num_t, 2> fun(num_t t, std::array<num_t, 2> y) {
    return std::array<num_t, 2>{y[1], (t*t-9)*y[0]};
}

int main() {
    std::array<num_t, 2> y0{1, 0};
    num_t t0{0};

    GiNaC::Digits = 50;

    Integrator<2> javi(t0, y0, fun);
    javi.integrate(10);
    return 0;
}

