#pragma once
// Three-macro test harness.
//
// Deliberately not GoogleTest: adding a dependency that has to be fetched and
// built would mean "run the tests" is something you do once and then stop
// doing. These tests build with a single g++ command and no network, so they
// stay in reach on any machine, including one mid-setup.

#include <cmath>
#include <cstdio>
#include <string>

inline int& test_failures() { static int n = 0; return n; }
inline int& test_checks()   { static int n = 0; return n; }

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++test_checks();                                                       \
        if (!(cond)) {                                                         \
            ++test_failures();                                                 \
            std::printf("  FAIL %s:%d  CHECK(%s)\n", __FILE__, __LINE__,       \
                        #cond);                                                \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++test_checks();                                                       \
        const double va__ = (double)(a), vb__ = (double)(b);                   \
        if (!(std::fabs(va__ - vb__) <= (double)(tol))) {                      \
            ++test_failures();                                                 \
            std::printf("  FAIL %s:%d  %s (%.6g) !~ %s (%.6g), tol %.3g\n",    \
                        __FILE__, __LINE__, #a, va__, #b, vb__, (double)(tol));\
        }                                                                      \
    } while (0)

#define SECTION(name) std::printf("- %s\n", name)

inline int test_summary(const char* suite) {
    if (test_failures() == 0) {
        std::printf("\n%s: all %d checks passed\n", suite, test_checks());
        return 0;
    }
    std::printf("\n%s: %d of %d checks FAILED\n", suite, test_failures(),
                test_checks());
    return 1;
}
