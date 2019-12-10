#ifndef RAND_H
#include <cstdlib>
#include <cstdint>
#include <assert.h>

#define MY_RAND_MAX 0x7fffffff

void init_rand_state();
uint32_t xorshift128plus(uint64_t* state);
extern uint64_t rand_state[2];
#define RAND() xorshift128plus(&rand_state[0])

#define RAND_H
#endif