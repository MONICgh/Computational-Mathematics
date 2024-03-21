#include <iostream>
#include <omp.h>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

int N = 900;
int NB = 50;
int MAX_ITER = 100;
int THREAD_COUNT = 8;

const int SEED = 1;
const double H = (double)(1) / (double)(N + 1);
const double EPS = 1e-3;

// Алгоритм 11.1
auto problem_finding_function(vector<vector<double>>& u, const vector<vector<double>>& f, int& iter) {
    auto start = chrono::high_resolution_clock::now();

    double dmax = 0;
    do {
        dmax = 0;
        for (int i = 1; i < N + 1; i++) {
            for (int j = 1; j < N + 1; j++) {
                double temp = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - H * H * f[i][j]);
                double dm = fabs(temp - u[i][j]);
                if (dm > dmax) dmax = dm;
            }
        }
        iter++;
        if (iter >= MAX_ITER) break;
    } while (dmax > EPS);

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::duration<double>>(end - start);
}


// Алгоритм 11.6
auto problem_finding_function_parallel(vector<vector<double>>& u, const vector<vector<double>>& f, int& iter) {
    auto start = chrono::high_resolution_clock::now();

    omp_lock_t dmax_lock;
    omp_init_lock(&dmax_lock);

    vector<double> dm_all(N + 1, 0.0);
    double dmax = 0;
    do {
        dmax = 0;
        for (int nx = 0; nx < NB; nx++) {

            int i, j, i_in, j_in;
            double temp, dm_local;

            #pragma omp parallel for shared(u, f, dm_all, nx, NB, N) private(i, j, i_in, j_in, temp, dm_local) num_threads(THREAD_COUNT)
            for (int i = 0; i < nx + 1; i++) {
                j = nx - i;
                for (i_in = i * NB + 1; i_in <= min(N, (i + 1) * NB); i_in++) {
                    for (j_in = j * NB + 1; j_in <= min(N, (j + 1) * NB); j_in++) {
                        temp = u[i_in][j_in];
                        u[i_in][j_in] = 0.25 * (u[i_in - 1][j_in] + u[i_in + 1][j_in] + u[i_in][j_in - 1] + u[i_in][j_in + 1] - H * H * f[i_in][j_in]);
                        dm_local = fabs(temp - u[i_in][j_in]);
                        if (dm_local > dm_all[i_in]) {
                            dm_all[i_in] = dm_local;
                        }
                    }
                }
            }
        }
        for (int nx = NB - 2; nx > -1; nx--) {

            int i, j, i_in, j_in;
            double temp, dm_local;

            #pragma omp parallel for shared(u, f, dm_all, nx, NB, N) private(i, j, i_in, j_in, temp, dm_local) num_threads(THREAD_COUNT)
            for (int i = 0; i < nx + 1; i++) {
                int j = 2 * (NB - 1) - nx - i;
                for (i_in = i * NB + 1; i_in <= min(N, (i + 1) * NB); i_in++) {
                    for (j_in = j * NB + 1; j_in <= min(N, (j + 1) * NB); j_in++) {
                        temp = u[i_in][j_in];
                        u[i_in][j_in] = 0.25 * (u[i_in - 1][j_in] + u[i_in + 1][j_in] + u[i_in][j_in - 1] + u[i_in][j_in + 1] - H * H * f[i_in][j_in]);
                        dm_local = fabs(temp - u[i_in][j_in]);
                        if (dm_local > dm_all[i_in]) {
                            dm_all[i_in] = dm_local;
                        }
                    }
                }
            }
        }
        int sz = 100, i, j;
        double d;
        
        #pragma omp parallel for shared(dm_all, dmax, N, sz, dmax_lock) private(i, j, d) default(none) num_threads(THREAD_COUNT)
        for (i = 1; i <= N; i += sz) {
            d = 0;
            for (j = i; j < min(i + sz, N + 1); j++)
                if (d < dm_all[j]) d = dm_all[j];

            omp_set_lock(&dmax_lock);
            if (dmax < d) dmax = d;
            omp_unset_lock(&dmax_lock);
        }
        iter++;
        if (iter >= MAX_ITER) break;
    } while ( dmax > EPS ); 

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::duration<double>>(end - start);
}

void generate_start_values(vector<vector<double>>& u) {
    
    mt19937 gen(SEED);
    uniform_real_distribution<double> rand(0.0, 1.0);
    
    double x, y;
    for (int i = 0; i < N + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            x = (double)i / (double)(N + 1);
            y = (double)j / (double)(N + 1);
            if (y == 0) u[i][j] = 100 - 200 * x;
            else if (x == 0) u[i][j] = 100 - 200 * y;
            else if (y == 1) u[i][j] = -100 + 200 * x;
            else if (x == 1) u[i][j] = -100 + 200 * y;
            else u[i][j] = rand(gen);
        }
    }
}

void run () {
    vector<vector<double>> u(N + 2, vector<double> (N + 2)), f(N + 2, vector<double> (N + 2));
    generate_start_values(u);
    int k1 = 0;
    auto t1 = problem_finding_function(u, f, k1);

    cout << N << ' ' << NB << ' ' << THREAD_COUNT << '\n';
    cout << k1 << ' ' << fixed << setprecision(3) << t1.count() << '\n';

    generate_start_values(u);
    int k2 = 0;
    auto t2 = problem_finding_function_parallel(u, f, k2);

    cout <<  k2 << ' ' << t2.count() << '\n' << '\n';
    // time in seconds, k - iterations count
}

int main () {

    // N = 900; NB = 50; MAX_ITER = 100; THREAD_COUNT = 8;
    run();

    return 0;
} 