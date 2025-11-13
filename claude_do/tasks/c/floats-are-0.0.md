---
description: replace all literal floats/doubles written as integers to floating point (e.g. 1.0f/1.0)
---

You are a Software Developer with many years of experience in writing C code.

Check this repository for potential implicit float/double to integer conversion during
assignment and comparison. In particular:

- if a variable's data type is double or float and assigned or compared
  to a literal number, that number must be written as double or float, respectively.

  For example:
  ```
  double a = 0.0; // Correct
  float b = 1.0f; // Correct
  double c = 0; // Incorrect
  float d = 2; // Incorrect

  if (a == 0) {} // Incorrect, should be a == 0.0 instead
  if (b < 1) {} // Incorrect, should be b < 1.0f instead
  ```

- if a variable's data type is double or float and it is changed with a literal number,
  that number must be written as double or float, respectively:

  For example:
  ```
  double a = 0.0;

  fabs(a + 1.0); // Correct
  fmod(a/3); // Incorrect, should be a/3.0  instead
  ```

Check all source files in this repository and fix them as required.
