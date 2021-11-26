#include "transpose-operator-tester.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/transpose.h>

TEST(TRANSPOSE_X32_E2E, transpose_e2e_2D_0_1) {
  TransposeOperatorTester()
      .num_dims(2)
      .set_shape({1024, 768})
      .set_perm({1, 0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_2D_1_0) {
  TransposeOperatorTester()
      .num_dims(2)
      .set_shape({1024, 768})
      .set_perm({1, 0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p0) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({4,3,2})
      .set_perm({0,1,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p1) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({4,3,2})
      .set_perm({0,2,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p2) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({4,3,2})
      .set_perm({1,0,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p3) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({8,6,4})
      .set_perm({1,2,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p4) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({1024,768,7})
      .set_perm({2,0,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p5) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({1024,768,7})
      .set_perm({2,1,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p6) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({8,5,3})
      .set_perm({1,2,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_3D_p7) {
  TransposeOperatorTester()
      .num_dims(3)
      .set_shape({4,3,2})
      .set_perm({0,2,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_1_2_3) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,1,2,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_1_3_2) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,1,3,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_2_1_3) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,2,1,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_2_3_1) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,2,3,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_3_1_2) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,3,1,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_0_3_2_1) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({0,3,2,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_0_2_3) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,0,2,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_0_3_2) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,0,3,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_2_0_3) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,2,0,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_2_3_0) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,2,3,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_3_0_2) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,3,0,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_1_3_2_0) {
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({1,3,2,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_0_1_3){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,0,1,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_0_3_1){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,0,3,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_1_0_3){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,1,0,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_1_3_0){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,1,3,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_3_0_1){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,3,0,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_2_3_1_0){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({2,3,1,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_0_1_2){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,0,1,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_0_2_1){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,0,2,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_1_0_2){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,1,0,2})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_1_2_0){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,1,2,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_2_0_1){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,2,0,1})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_4D_3_2_1_0){
  TransposeOperatorTester()
      .num_dims(4)
      .set_shape({4,3,2,5})
      .set_perm({3,2,1,0})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_5D_3_2_1_0){
  TransposeOperatorTester()
      .num_dims(5)
      .set_shape({4,3,2,5,6})
      .set_perm({4,2,1,0,3})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_6D_3_2_1_0){
  TransposeOperatorTester()
      .num_dims(6)
      .set_shape({4,3,2,5,6,13})
      .set_perm({5,2,1,0,3,4})
      .TestX32();
}

TEST(TRANSPOSE_X32_E2E, transpose_e2e_6D_3_2_1_0_7){
  TransposeOperatorTester()
      .num_dims(6)
      .set_shape({7,11,37,41,3,13})
      .set_perm({5,1,0,2,3,4})
      .TestX32();
}
