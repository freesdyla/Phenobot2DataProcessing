//
// pointcloud3.cpp
//     Class for holding a set of 3D points
//
// Author:  Tilman Schramke, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#include "pointcloud3.h"
#include <stdio.h>
#include <math.h>
#include <string>


// translate point cloud so that center = origin
// total shift applied to this point cloud is stored in this->shift
void PointCloud3::shiftToOrigin(){
  Vector3d p1, p2, newshift;
  this->getMinMax3D(&p1, &p2);
  newshift = (p1 + p2) / 2.0;
  for(size_t i=0; i < points.size(); i++){
    points[i] = points[i] - newshift;
  }
  shift = shift + newshift;
}

// mean value of all points (center of gravity)
Vector3d PointCloud3::meanValue() const {
  Vector3d ret;
  for(size_t i = 0; i < points.size(); i++){
    ret = ret + points[i];
  }
  if (points.size() > 0)
    return (ret / (float)points.size());
  else
    return ret;
}

// bounding box corners
void PointCloud3::getMinMax3D(Vector3d* min_pt, Vector3d* max_pt){
  if(points.size() > 0){
    *min_pt = points[0];
    *max_pt = points[0];

    for(std::vector<Vector3d>::iterator it = points.begin(); it != points.end(); it++){
      if(min_pt->x > it->x) min_pt->x = it->x;
      if(min_pt->y > it->y) min_pt->y = it->y;
      if(min_pt->z > it->z) min_pt->z = it->z;

      if(max_pt->x < (*it).x) max_pt->x = (*it).x;
      if(max_pt->y < (*it).y) max_pt->y = (*it).y;
      if(max_pt->z < (*it).z) max_pt->z = (*it).z;
    }
  } else {
    *min_pt = Vector3d(0,0,0);
    *max_pt = Vector3d(0,0,0);
  }
}


void PointCloud3::readPCL(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	points.clear();

	for(auto & p : cloud->points)
	{
		Vector3d point(p.x, p.y, p.z);
		points.push_back(point);
	}
}

/*
void PointCloud3::setNormals(pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	if(normals->size() != points.size()) 
	{
		std::cout<<"normal cloud size not equal point cloud size!\n";
		return;
	}

	int i = 0;
	for(auto & n : normals->points)
	{
		points[i++].setNormal(n.normal_x, n.normal_y, n.normal_z);
	}

}
*/

// store points closer than dx to line (a, b) in Y
void PointCloud3::pointsCloseToLine(const Vector3d &a, const Vector3d &b, float dx, PointCloud3* Y) {

  Y->points.clear();
  for (size_t i=0; i < points.size(); i++) {
    // distance computation after IPOL paper Eq. (7)
    float t = (b * (points[i] - a));
    Vector3d d = (points[i] - (a + (t*b)));
    if (d.norm() <= dx) {
      Y->points.push_back(points[i]);
    }
  }
}

// removes the points in Y from PointCloud3
// WARNING: only works when points in same order as in PointCloud3!
void PointCloud3::removePoints(const PointCloud3 &Y){

  if (Y.points.empty()) return;
  std::vector<Vector3d> newpoints;
  size_t i,j;

  // important assumption: points in Y appear in same order in points
  for (i = 0, j = 0; i < points.size() && j < Y.points.size(); i++){
    if (points[i] == Y.points[j]) {
      j++;
    } else {
      newpoints.push_back(points[i]);
    }
  }
  // copy over rest after end of Y
  for (; i < points.size(); i++)
    newpoints.push_back(points[i]);

  points = newpoints;
}
