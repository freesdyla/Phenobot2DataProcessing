//
// vector3d.h
//     Class providing common math operations for 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <iostream>

class Vector3d {
public:
  float x;
  float y;
  float z;

//  float nx_;
 // float ny_;
  //float nz_;

  Vector3d();
  Vector3d(float a, float b, float c);
  bool operator==(const Vector3d &rhs) const;
  Vector3d& operator=(const Vector3d& other);
  // nicely formatted output
  friend std::ostream& operator<<(std::ostream& os, const Vector3d& vec);
  // Euclidean norm
  float norm() const;

 // void setNormal(float nx, float ny, float nz);

};

// mathematical vector operations

// vector addition
Vector3d operator+(Vector3d x, Vector3d y);
// vector subtraction
Vector3d operator-(Vector3d x, Vector3d y);
// scalar product
float operator*(Vector3d x, Vector3d y);
// scalar multiplication
Vector3d operator*(Vector3d x, float c);
Vector3d operator*(float c, Vector3d x);
Vector3d operator/(Vector3d x, float c);


#endif /* VECTOR3D_H_ */
