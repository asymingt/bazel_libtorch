// Minimal assertion helpers for the smoke tests: CHECK() records a failure and
// keeps going, so one run reports every broken check.
#pragma once

#include <cstdio>

namespace test {

inline int& Failures() {
  static int failures = 0;
  return failures;
}

// Returns the process exit code, after printing a summary.
inline int Finish(const char* name) {
  if (Failures() != 0) {
    std::fprintf(stderr, "%s: %d check(s) failed\n", name, Failures());
    return 1;
  }
  std::printf("%s: all checks passed\n", name);
  return 0;
}

}  // namespace test

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                   #cond);                                                \
      ++::test::Failures();                                               \
    }                                                                     \
  } while (false)
