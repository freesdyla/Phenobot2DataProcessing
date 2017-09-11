//
// vector3d.cpp
//     Class providing common math operations for 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#include "vector3d.h"
#include <math.h>

Vector3d::Vector3d() {
  x = 0; y = 0; z = 0;
}

Vector3d::Vector3d(float a, float b, float c) {
  x = a; y = b; z = c;
}

bool Vector3d::operator==(const Vector3d &rhs) const {
  if((x == rhs.x) && (y == rhs.y) && (z == rhs.z))
    return true;
  return false;
}

Vector3d& Vector3d::operator=(const Vector3d& other) {
  x = other.x; y = other.y; z = other.z;
//  nx_ = other.nx_; ny_ = other.ny_; nz_ = other.nz_; 
  return *this;
}

// nicely formatted output
std::ostream& operator<<(std::ostream& strm, const Vector3d& vec) {
  return strm << "(" << vec.x << "," << vec.y << "," << vec.z << ")";
}

// Euclidean norm
float Vector3d::norm() const {
  return sqrt((x * x) + (y * y) + (z * z));
}

/*
void Vector3d::setNormal(float nx, float ny, float nz)
{
	nx_ = nx; ny_ = ny; nz_ = nz;
}*/

// mathematical vector operations

// vector addition
Vector3d operator+(Vector3d x, Vector3d y) {
  Vector3d v(x.x + y.x, x.y + y.y, x.z + y.z);
  return v;
}

// vector subtraction
Vector3d operator-(Vector3d x, Vector3d y) {
  Vector3d v(x.x - y.x, x.y - y.y, x.z - y.z);
  return v;
}

// scalar product
float operator*(Vector3d x, Vector3d y) {
  return (x.x*y.x + x.y*y.y +  x.z*y.z);
}

// scalar multiplication
Vector3d operator*(Vector3d x, float c) {
  Vector3d v(c*x.x, c*x.y, c*x.z);
  return v;
}
Vector3d operator*(float c, Vector3d x) {
  Vector3d v(c*x.x, c*x.y, c*x.z);
  return v;
}
Vector3d operator/(Vector3d x, float c) {
  Vector3d v(x.x/c, x.y/c, x.z/c);
  return v;
}
