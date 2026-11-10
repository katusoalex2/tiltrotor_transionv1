#include <stdio.h>
#include <math.h>
#include "pid_controller.h"

int main(void) {
    double state[12] = {0.0, 0.0, -100.0, 10.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0};
    double ref[3] = {100.0, 12.0, 0.5};
    double integral[2] = {0.0, 0.0};
    double omega, vtail;
    double integral_new[2];

    pid_controller_step(state, ref, integral, &omega, &vtail, integral_new);

    printf("TEST1 %.10f %.10f %.10f %.10f\n", omega, vtail, integral_new[0], integral_new[1]);

    double state2[12] = {0.0, 0.0, -95.0, 8.0, 2.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.0};
    double ref2[3] = {100.0, 10.0, 0.3};
    double integral2[2] = {0.1, -0.05};

    pid_controller_step(state2, ref2, integral2, &omega, &vtail, integral_new);

    printf("TEST2 %.10f %.10f %.10f %.10f\n", omega, vtail, integral_new[0], integral_new[1]);

    return 0;
}
