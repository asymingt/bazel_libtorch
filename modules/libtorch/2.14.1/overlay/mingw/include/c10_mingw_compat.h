// Force-included (-include) into every @libtorch//:c10 source file on
// Windows, for upstream code that is only built with MSVC on Windows:
//
// - c10/util/tempfile.cpp uses _SH_DENYNO and _S_IREAD/_S_IWRITE, which the
//   MSVC headers it includes declare transitively, but MinGW-w64's do not.
// - c10/util/env.cpp only takes its putenv() path under _MSC_VER, so MinGW-w64
//   builds call the POSIX setenv()/unsetenv(), which its C runtime does not
//   declare. Provide them on top of _putenv_s().
#pragma once

#if defined(__MINGW32__) && defined(__cplusplus)
#include <share.h>
#include <stdlib.h>
#include <sys/stat.h>

static inline int setenv(const char* name, const char* value, int overwrite) {
  if (!overwrite && getenv(name) != nullptr) {
    return 0;
  }
  return _putenv_s(name, value);
}

// On Windows an assignment with an empty value removes the variable.
static inline int unsetenv(const char* name) {
  return _putenv_s(name, "");
}
#endif
