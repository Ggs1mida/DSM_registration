#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static double PYTHAG(double a, double b);
int dsvd(double **a, int m, int n, double *w, double **v);