//
// hough.h
//     Implementation of Algorithm 2 (Hough transform) from IPOL paper
//
// Author:  Tilman Schramke, Manuel Jeltsch, Christoph Dalitz
// Date:    2017-03-16
// License: see LICENSE-BSD2
//

#ifndef HOUGH_H_
#define HOUGH_H_


#include "vector3d.h"
#include "sphere.h"
#include "pointcloud3.h"
#include <vector>
#include <deque>
#include <iostream>

class Hough {
public:
	// accumulator array A
	std::vector<unsigned int> VotingSpace;

	// Directions B
	Sphere *sphere;

	size_t num_b;

	// x' and y'
	float dx, max_x;

	size_t num_x;

	float verticality_;

	Vector3d minP_, maxP_;

	// parameter space discretization and allocation of voting space
	Hough(const Vector3d& minP, const Vector3d& maxP, float dx, unsigned int sphereGranularity, float verticality);

	~Hough();

	// returns the line with most votes (rc = number of votes)
	unsigned int getLine(Vector3d* point, Vector3d* direction);

	// add all points from point cloud to voting space
	void add(const PointCloud3 &pc);

	// subtract all points from point cloud to voting space
	void subtract(const PointCloud3 &pc);

private:
	// add or subtract (add==false) one point from voting space
	void pointVote(const Vector3d& point, bool add);
};

#endif /* HOUGH_H_ */
