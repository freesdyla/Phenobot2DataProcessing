#include <iostream>
#include <cstdlib>
#include <map>
#include <algorithm>
#include <fstream>
#include <cassert>

#include <omp.h>

#include <GeographicLib/UTMUPS.hpp>
using namespace std;
using namespace GeographicLib;


#include "vector3d.h"
#include "pointcloud3.h"
#include "hough.h"
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

using Eigen::MatrixXf;


#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_picking_event.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_stick.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/cpc_segmentation.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/common/transforms.h>
#include <pcl/common/angles.h>
#include <pcl/common/pca.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/incremental_registration.h>
#include <pcl/registration/transformation_estimation_2D.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/console/time.h>
#include <pcl/ml/kmeans.h>


#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/copy.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/config.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/two_bit_color_map.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>

//#include <voronoi_diagram.h>

using namespace boost::filesystem;
using namespace boost;


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

typedef pcl::PointXYZRGBA PointAT;
typedef pcl::PointCloud<PointAT> PointACloudT;

typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

typedef pcl::Normal NormalT;
typedef pcl::PointCloud<NormalT> NormalCloudT;
typedef pcl::PointXYZRGBNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;

//boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, uint32_t, float>
typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList Graph;
typedef boost::graph_traits<Graph>::vertex_iterator supervoxel_iterator;
typedef boost::graph_traits<Graph>::edge_iterator supervoxel_edge_iter;
typedef pcl::SupervoxelClustering<PointT>::VoxelID Voxel;
typedef pcl::SupervoxelClustering<PointT>::EdgeID Edge;
typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList::adjacency_iterator supervoxel_adjacency_iterator;


int rr_num_neighbor;
float rr_residual;
float rr_curvature;
float rr_angle;
float vox_size;
float normal_radius;
float cylinder_ransac_dist;
float neighboring_gps_threshold;
int pcl_view_time = 0;
float voxel_resolution = 0.005;
float seed_resolution = 0.05;
float color_importance = 0.0f;
float spatial_importance = 1.0f;
float normal_importance = 0.0f;
int max_d_jump = 10;
int sor_meank;
float sor_std;
int h_minvotes = 20;
double h_dx = 0.04;
int h_nlines = 4;
int h_granularity = 4;
int cnt = 0;
int skeleton_iteration = 20;
float step_size;
float min_linearity;
int min_branch_size;
int data_set = 0;	//0 2015; 1 2017
float min_z = 0.5f;
float max_z = 1.2f;
float stem_verticality = 0.9f;
int start_plant_id = 0;
float max_stem_radius = 0.3f;
float slice_thickness = 0.01f;
float max_spatial_res = 0.004f;

const double utm_o_x = 441546.;
const double utm_o_y = 4654933.;

float shortRainbowColorMap(const float value, const float min, const float max) {
    uint8_t r, g, b;

    // Normalize value to [0, 1]
    float value_normalized = (value - min) / (max - min);

    float a = (1.0f - value_normalized) / 0.25f;
    int X = static_cast<int>(floorf(a));
    int Y = static_cast<int>(floorf(255.0f * (a - X)));

    switch (X) {
        case 0: 
            r = 255;
            g = Y;
            b = 0;
            break;
        case 1: 
            r = 255 - Y;
            g = 255;
            b = 0;
            break;
        case 2: 
            r = 0;
            g = 255;
            b = Y;
            break;
        case 3: 
            r = 0;
            g = 255-Y;
            b = 255;
            break;
        case 4: 
            r = 0;
            g = 0;
            b = 255;
            break;
    }

    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    return *reinterpret_cast<float*>(&rgb);
}

float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event)
{
	float x, y, z;
	event.getPoint(x, y ,z);
	Dist = sqrt( pow(x-pre_x, 2) + pow(y-pre_y, 2) + pow(z-pre_z, 2) );
//	Eigen::Vector3f dir(pre_x-x, pre_y-y, pre_z-z);
//	dir.normalize();	
	pre_x = x;
	pre_y = y;
	pre_z = z;
	std::cout<<"x:"<<x<<" y:"<<y<<" z:"<<z<<" distance:"<<Dist/*<<" nx:"<<dir(0)<<" ny:"<<dir(1)<<" nz:"<<dir(2)*/<<std::endl;
	
}



float dist3D_Segment_to_Segment(std::pair<Eigen::Vector3f, Eigen::Vector3f> & S1, std::pair<Eigen::Vector3f, Eigen::Vector3f> & S2)
{	
	Eigen::Vector3f u = S1.second - S1.first;

	Eigen::Vector3f v = S2.second - S2.first;

	Eigen::Vector3f w = S1.first - S2.first;

	float a = u.dot(u);         // always >= 0
	float b = u.dot(v);
	float c = v.dot(v);         // always >= 0
	float d = u.dot(w);
	float e = v.dot(w);
	float D = a*c - b*b;        // always >= 0
	float sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
	float tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

	// compute the line parameters of the two closest points
	if (D < 1e-7) 
	{ // the lines are almost parallel
		sN = 0.0;         // force using point P0 on segment S1
		sD = 1.0;         // to prevent possible division by 0.0 later
		tN = e;
		tD = c;
	}
	else
	{                 // get the closest points on the infinite lines
		sN = (b*e - c*d);
		tN = (a*e - b*d);

		if (sN < 0.0) 
		{        // sc < 0 => the s=0 edge is visible
		    sN = 0.0;
		    tN = e;
		    tD = c;
		}
		else if (sN > sD) 
		{  // sc > 1  => the s=1 edge is visible
		    sN = sD;
		    tN = e + b;
		    tD = c;
		}
	}

	if (tN < 0.0) 
	{            // tc < 0 => the t=0 edge is visible
		tN = 0.0;
		// recompute sc for this edge
		if (-d < 0.0)
			sN = 0.0;
		else if (-d > a)
			sN = sD;
		else 
		{
			sN = -d;
			sD = a;
		}
	}
	else if (tN > tD) 
	{      // tc > 1  => the t=1 edge is visible
		tN = tD;
		// recompute sc for this edge
		if ((-d + b) < 0.0)
		    sN = 0;
		else if ((-d + b) > a)
		    sN = sD;
		else {
		    sN = (-d +  b);
		    sD = a;
		}
	}

	// finally do the division to get sc and tc
	sc = (abs(sN) < 1e-7 ? 0.0 : sN / sD);
	tc = (abs(tN) < 1e-7 ? 0.0 : tN / tD);

	// get the difference of the two closest points
	Eigen::Vector3f dP = w + (sc * u) - (tc * v);  // =  S1(sc) - S2(tc)

	return dP.norm();   // return the closest distance
}


// orthogonal least squares fit with libeigen
// rc = largest eigenvalue
//
float orthogonal_LSQ(const PointCloud3 &pc, Vector3d* a, Vector3d* b)
{
	float rc = 0.f;

	// anchor point is mean value
	*a = pc.meanValue();

	// copy points to libeigen matrix
	Eigen::MatrixXf points = Eigen::MatrixXf::Constant(pc.points.size(), 3, 0);

	for (int i = 0; i < points.rows(); i++)	{

		points(i,0) = pc.points.at(i).x;
		points(i,1) = pc.points.at(i).y;
		points(i,2) = pc.points.at(i).z;
	}

	// compute scatter matrix ...
	MatrixXf centered = points.rowwise() - points.colwise().mean();
	MatrixXf scatter = (centered.adjoint() * centered);

	// ... and its eigenvalues and eigenvectors
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(scatter);
	Eigen::MatrixXf eigvecs = eig.eigenvectors();

	// we need eigenvector to largest eigenvalue
	// libeigen yields it as LAST column
	b->x = eigvecs(0,2); b->y = eigvecs(1,2); b->z = eigvecs(2,2);

	rc = eig.eigenvalues()(2);

	return (rc);
}

int Hough3DLine(PointCloudT::Ptr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer, 
		std::vector<Eigen::Vector3f> & a_vec, std::vector<Eigen::Vector3f> & b_vec,
		double opt_dx = 0.04, int opt_nlines = 4, int opt_minvotes = 2, int granularity = 4, float verticality = 0.9f)
{

	// number of icosahedron subdivisions for direction discretization
	int num_directions[7] = {12, 21, 81, 321, 1281, 5121, 20481};

	// bounding box of point cloud
	Vector3d minP, maxP, minPshifted, maxPshifted;
	// diagonal length of point cloud
	double d;

	PointCloud3 X;

	X.readPCL(cloud);
	
#if 0
	pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT>> (new pcl::search::KdTree<PointT>);
	NormalCloudT::Ptr normals(new NormalCloudT);
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(cloud);
	//normal_estimator.setRadiusSearch(normal_radius);
	normal_estimator.setKSearch(15);
	normal_estimator.compute(*normals);

	NormalCloudT::Ptr normals1(new NormalCloudT);

	std::vector<int> indices;
	pcl::removeNaNNormalsFromPointCloud(*normals, *normals1, indices);

	std::cout<<"num nan "<<normals->size()-indices.size()<<"\n";

	X.setNormals(normals);
#endif

	Hough* hough;

	// center cloud and compute new bounding box
	X.getMinMax3D(&minP, &maxP);

	d = (maxP-minP).norm();

	if (d == 0.0) {

		fprintf(stderr, "Error: all points in point cloud identical\n");
		return 1;
	}

	X.shiftToOrigin();

	X.getMinMax3D(&minPshifted, &maxPshifted);

	// estimate size of Hough space
	if (opt_dx == 0.0) {

		opt_dx = d / 64.0;
	}
	else if (opt_dx >= d) {

		fprintf(stderr, "Error: dx too large\n");
		return 1;
	}

	double num_x = floor(d / opt_dx + 0.5);

	double num_cells = num_x * num_x * num_directions[granularity];

	// first Hough transform

	try {

		hough = new Hough(minPshifted, maxPshifted, opt_dx, granularity, verticality);
	} 
	catch (const std::exception &e) {

		fprintf(stderr, "Error: cannot allocate memory for %.0f Hough cells"
		    " (%.2f MB)\n", num_cells, 
		    (double(num_cells) / 1000000.0) * sizeof(unsigned int));
		return 2;
	}

	hough->add(X);

	a_vec.clear();
	b_vec.clear();

	// iterative Hough transform
	// (Algorithm 1 in IPOL paper)
	PointCloud3 Y;	// points close to line
	float rc;
	unsigned int nvotes;
	int nlines = 0;
	do {
		Vector3d a; // anchor point of line
		Vector3d b; // direction of line

		hough->subtract(Y); // do it here to save one call

		nvotes = hough->getLine(&a, &b);
		
		X.pointsCloseToLine(a, b, opt_dx, &Y);

/*		rc = orthogonal_LSQ(Y, &a, &b);

		if ( rc < 1e-4f ) {
			//cout<<"1. rc = 0\n";
			break;
		}

		X.pointsCloseToLine(a, b, opt_dx, &Y);
*/
		nvotes = Y.points.size();

		if ( nvotes < (unsigned int)opt_minvotes ) {
			//cout<<"nvotes < minvotes\n";
			break;
		}

/*		rc = orthogonal_LSQ(Y, &a, &b);

		if ( rc == 1e-4f ) {
			//cout<<"2. rc = 0\n";
			break;
		}
*/
		a = a + X.shift;

		nlines++;

		if(b.x > 0.) b = b*-1.;

		double t = (0.6 - a.x)/b.x;

		a.x = 0.6;
		a.y += t*b.y;
		a.z += t*b.z;	

		Eigen::Vector3f a_eigen(a.x, a.y, a.z);
		Eigen::Vector3f b_eigen(b.x, b.y, b.z);

		a_vec.push_back(a_eigen);
		b_vec.push_back(b_eigen);

		printf("npoints=%lu, a=(%f,%f,%f), b=(%f,%f,%f)\n", Y.points.size(), a.x, a.y, a.z, b.x, b.y, b.z);
			
		X.removePoints(Y);
	} 
	while ((X.points.size() > 1) && ((opt_nlines == 0) || (opt_nlines > nlines)));

//	viewer->addPointCloud(cloud, to_string(cv::getTickCount()));
//	viewer->spin();
//	viewer->removeAllPointClouds();
//	viewer->removeAllShapes();

	// clean up
	delete hough;
	return 0;
}


#if 1
// Graph edge properties (bundled properties)
struct SVEdgeProperty
{
	float weight;
};

struct SVVertexProperty
{
	uint32_t supervoxel_label;
	pcl::Supervoxel<PointT>::Ptr supervoxel;
	//float convexity;
	//float x, y, z;
	//PointCloudT cloud;
	uint32_t index;
	float max_width;
	int vertex;
	bool near_junction = false;
	std::vector<uint32_t> children;
	std::vector<uint32_t> parents;
	std::vector<int> cluster_indices;
};

struct found_goal{};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SVVertexProperty, SVEdgeProperty> sv_graph_t;

typedef boost::graph_traits<sv_graph_t>::vertex_descriptor sv_vertex_t;

typedef boost::graph_traits<sv_graph_t>::edge_descriptor sv_edge_t;

// visitor that terminates when goal found
template<class Vertex>
class bfs_goal_visitor : public boost::default_bfs_visitor
{
public:
	bfs_goal_visitor(Vertex root) : m_root(root) {}
	template <class Graph>
	void examine_vertex(Vertex u, Graph& g) 
	{
		float dist = std::sqrt(
				std::pow(g[u].x - g[m_root].x, 2.f)
				+std::pow(g[u].y - g[m_root].y, 2.f)
				+std::pow(g[u].z - g[m_root].z, 2.f)
				);

		if (dist > 0.06f)
			throw found_goal();
	}
private:
	Vertex m_root;
};

struct TreeNode
{
	int parent_node;
	int vertex_id;
	std::vector<int> children_nodes;
};

#endif

#if 0
Eigen::Vector3d project(const pcl::PointXYZRGBA & origin, const pcl::Normal & normal, const pcl::PointXYZRGBA & point) 
{
    // Bring the point to the origin
    Eigen::Vector3f p = point.getVector3fMap() - origin.getVector3fMap();
    Eigen::Vector3f n = normal.getNormalVector3fMap();

    n.normalize();
    const double projection = static_cast<double>(p.dot(n));

    return p.cast<double>() - projection * n.cast<double>();
}

// point-based manifold harmonic bases
void PB_MHB(sv_graph_t & sv_graph, const double seed_resolution, Eigen::MatrixXd & eigenvectors_out, Eigen::VectorXd & eigenvalues_out, int & first_non_zero_idx) {

	const auto num_sv = boost::num_vertices(sv_graph);

	int points_with_mass = 0;

	double avg_mass = 0.0;

        // Mass matrix and stiffness matrix
        std::vector<double> B, S;
        std::vector<int> I, J;

        Eigen::MatrixXd eigenfunctions;
        Eigen::VectorXd eigenvalues;

	B.resize(num_sv);

	std::cout << "Computing the Mass matrix..." << std::flush;
	for(int i=0; i<num_sv; i++) {

		const auto & point = sv_graph[i].supervoxel->centroid_;

		const auto & normal = sv_graph[i].supervoxel->normal_;

		const auto & normal_vector = normal.getNormalVector3fMap().template cast<double>();

		const auto & degree = boost::out_degree(i, sv_graph);

		if(degree < 3) {
			B[i] = 0.;
			continue;
		}

		// Project the neighbor points in the tangent plane at p_i with normal n_i
		std::vector<Eigen::Vector3d> projected_points;

		BGL_FORALL_ADJ(i, adj, sv_graph, sv_graph_t)
		{
			const auto & neighbor_point = sv_graph[adj].supervoxel->centroid_;

			projected_points.push_back( project(point, normal, neighbor_point) );

		}
		
		assert(projected_points.size() >= 3);

		// Use the first vector to create a 2D basis
		Eigen::Vector3d u = projected_points[0];
		u.normalize();
		Eigen::Vector3d v = (u.cross(normal_vector));
		v.normalize();

		// Add the points to a 2D plane
		std::vector<Eigen::Vector2d> plane;

		// Add the point at the center
		plane.push_back(Eigen::Vector2d::Zero());

		// Add the rest of the points
		for (const auto& projected : projected_points) {

		    double x = projected.dot(u);
		    double y = projected.dot(v);

		    // Add the 2D point to the vector
		    plane.push_back(Eigen::Vector2d(x, y));
		}

		assert(plane.size() >= 4);

		// Compute the voronoi cell area of the point
		double area = VoronoiDiagram::area(plane);
		B[i] = area;
		avg_mass += area;
		points_with_mass++;
	}

	cout<<"\npoints with mass "<< points_with_mass<<"\n";

	// Average mass
	if (points_with_mass > 0) {
		avg_mass /= static_cast<double>(points_with_mass);
	}

	// Set border points to have average mass
	for (auto & b : B) {
		if (b == 0.0) {
		    b = avg_mass; 
		} 
	}

	std::cout << "done" << std::endl;
	std::cout << "Computing the stiffness matrix..." << std::flush;

	std::vector<double> diag(num_sv, 0.0);

	// Compute the stiffness matrix Q
	for (int i = 0; i < num_sv; i++) {
		const auto& point = sv_graph[i].supervoxel->centroid_;

		BGL_FORALL_ADJ(i, adj, sv_graph, sv_graph_t)
		{
			const auto & neighbor = sv_graph[adj].supervoxel->centroid_;

			double d = (neighbor.getVector3fMap() - point.getVector3fMap()).norm();

			double w = B[i] * B[adj] * (1.0 / (4.0 * M_PI * seed_resolution * seed_resolution)) * exp(-(d * d) / (4.0 * seed_resolution));

			I.push_back(i);
			J.push_back(adj);
			S.push_back(w);

			diag[i] += w;
		}
	}

	// Fill the diagonal as the negative sum of the rows
	for (int i = 0; i < diag.size(); i++) {
		I.push_back(i);
		J.push_back(i);
		S.push_back(-diag[i]);
	}

	// Compute the B^{-1}Q matrix
	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(num_sv, num_sv);

	
	for (int i = 0; i < I.size(); i++) {
		const int row = I[i];
		const int col = J[i];
		Q(row, col) = S[i];
	}

	std::cout << "done" << std::endl;
	std::cout << "Computing eigenvectors" << std::endl;

	Eigen::Map<Eigen::VectorXd> B_vec(B.data(), B.size());

	pcl::console::TicToc tt;
	tt.tic();

	Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> ges;
	ges.compute(Q, B_vec.asDiagonal());

	tt.toc_print();

	eigenvalues = ges.eigenvalues();
	eigenfunctions = ges.eigenvectors();

	// Sort the eigenvalues by magnitude
	std::vector<std::pair<double, int> > map_vector(eigenvalues.size());

	for (auto i = 0; i < eigenvalues.size(); i++) {
		map_vector[i].first = std::abs(eigenvalues(i));
		map_vector[i].second = i;
	}

	std::sort(map_vector.begin(), map_vector.end());

	// truncate the first 100 eigenfunctions
	Eigen::MatrixXd eigenvectors(eigenfunctions.rows(), eigenfunctions.cols());
	Eigen::VectorXd eigenvals(eigenfunctions.cols());

	eigenvalues.resize(map_vector.size());

	first_non_zero_idx = 0;
	
	for (auto i = 0; i < map_vector.size(); i++) {
		const auto& pair = map_vector[i];
		eigenvectors.col(i) = eigenfunctions.col(pair.second); 
		eigenvals(i) = pair.first;
    	}

	for(int i=0; i<eigenvals.size(); i++) {

		if(eigenvals(i) > 1e-8) {
			first_non_zero_idx = i;
			break;
		}
	}

	cout<<eigenvals(first_non_zero_idx-1)<<" "<<eigenvals(first_non_zero_idx)<<" "<<eigenvals(first_non_zero_idx+1)<<"\n";

	eigenvectors_out = eigenvectors;
	eigenvalues_out = eigenvals;
}
#endif

struct DepthFrame {

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	double utm_x;
	double utm_y;
	double odo_x;
	double odo_y;
	double odo_z;
	double utc;
	std::string file_path;
	Eigen::Matrix4f* world2cam;
	Eigen::Matrix4f* world2robot;
	bool invalid;

	DepthFrame() {

		world2cam = new Eigen::Matrix4f();
		world2robot = new Eigen::Matrix4f();
	}
};

struct BGRD {

	unsigned char b;
	unsigned char g;
	unsigned char r;
	unsigned short d;
};

bool compareByUTC(const DepthFrame & a, const DepthFrame & b) {

	return a.utc < b.utc;
}

const int depth_width = 512;
const int depth_height = 424;
const int depth_size = depth_width*depth_height;

void readDataBinaryFile(std::string file_name, BGRD* buffer) {

	std::ifstream in(file_name, std::ios::in | std::ios::binary);

	in.read((char*)&buffer[0], depth_size*sizeof(BGRD));

	in.close();
}


cv::Mat cur_depth_cv, pre_depth_cv, pre_pre_depth_cv, depth_canvas_cv;

void skeleton(PointCloudT::Ptr cloud, Eigen::Vector4f & plane_coeffs, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer) {

	if(cloud->size() < 100) return;
	
	PointCloudT::Ptr filtered_cloud(new PointCloudT);	

	PointCloudT::Ptr tmp_cloud(new PointCloudT);	

	pcl::PCA<PointT> pca;

	pca.setInputCloud(cloud);

	pcl::KdTreeFLANN<PointT> kdtree;

	kdtree.setInputCloud(cloud);

	std::vector<int> indices;

	int index = 0;
	
	// filter line-like flying pixels 
	for(auto & p : cloud->points) {

		std::vector<int> k_indices; std::vector<float> k_sqr_distances;

		if( kdtree.nearestKSearch(p, 5, k_indices, k_sqr_distances) == 5) {		

			pca.setIndices(boost::make_shared<std::vector<int>>(k_indices));

			Eigen::Vector3f eigen_values = pca.getEigenValues();

			const float linearity = (eigen_values(0)-eigen_values(1))/eigen_values(0);

			if(linearity < 0.8f) 
				indices.push_back(index);
		}
		else
			cout<<"kdtree search fail\n";

		++index;
	}

#if 1
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud(cloud);
	sor.setIndices(boost::make_shared<std::vector<int>>(indices));
	sor.setMeanK(sor_meank);
	sor.setStddevMulThresh(sor_std);
	sor.filter(*filtered_cloud);
#endif

	Eigen::Vector4f min_pt, max_pt;

	pcl::getMinMax3D(*filtered_cloud, min_pt, max_pt);

	sv_graph_t skeleton_graph;

	pcl::PassThrough<PointT> pass;
	pass.setInputCloud (filtered_cloud);
	pass.setFilterFieldName ("x");
	pass.setFilterLimitsNegative (false);

	pcl::PassThrough<PointT> pass_in_cluster;
	pass_in_cluster.setFilterFieldName ("x");
	pass_in_cluster.setInputCloud(filtered_cloud);

	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
	tree->setInputCloud (filtered_cloud);

	
	pca.setInputCloud(filtered_cloud);
	pcl::PointIndices::Ptr point_indices(new pcl::PointIndices);

	pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
	feature_extractor.setInputCloud(filtered_cloud);
	feature_extractor.setAngleStep(400.f);
	
	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance (slice_thickness); // 2cm
	ec.setMinClusterSize (10);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (filtered_cloud);

	uint32_t rgb;

	int cnt = 0;

	PointCloudT::Ptr centroid_cloud(new PointCloudT);

	pcl::PointIndices::Ptr pre_indices(new pcl::PointIndices);

	bool first_slice = true;	
	


	kdtree.setInputCloud(filtered_cloud);

	for(float x=min_pt(0); x<=max_pt(0); x += slice_thickness) {

		pass.setFilterLimits (x, x+slice_thickness);

		pcl::PointIndices::Ptr indices(new pcl::PointIndices);	
		
		pass.filter(indices->indices);

		std::vector<int> tmp_indices;

		tmp_indices = indices->indices;

		// append previous slice
		indices->indices.insert(indices->indices.end(), pre_indices->indices.begin(), pre_indices->indices.end());

		pre_indices->indices = tmp_indices;

		ec.setIndices(indices);

		std::vector<pcl::PointIndices> cluster_indices;

		ec.extract(cluster_indices);

		if(first_slice) {

			first_slice = false;

			for(int i=0; i<cluster_indices.size(); i++) {

				Eigen::Vector4f centroid;				

				pcl::compute3DCentroid(*filtered_cloud, cluster_indices[i], centroid);

				PointT p; p.getVector3fMap() = centroid.head(3);

				std::vector<int> k_indices; std::vector<float> k_sqr_distances;

				if( kdtree.nearestKSearch(p, 1, k_indices, k_sqr_distances) == 1) {

					SVVertexProperty vp;

					vp.index = k_indices[0];	

					sv_vertex_t v = boost::add_vertex(vp, skeleton_graph);
				}
			}

			continue;
		}

		pass_in_cluster.setFilterLimits (x, x + slice_thickness);

		for(auto & two_layer_cluster : cluster_indices) {

			pcl::PointIndices::Ptr lower_slice_inliers_in_cluster(new pcl::PointIndices);

			pcl::PointIndices::Ptr upper_slice_inliers_in_cluster(new pcl::PointIndices);		

			boost::shared_ptr<std::vector<int>> two_layer_cluster_ptr(new std::vector<int>(two_layer_cluster.indices));

			pass_in_cluster.setIndices( two_layer_cluster_ptr );			

			pass_in_cluster.setFilterLimitsNegative (false);

			pass_in_cluster.filter(lower_slice_inliers_in_cluster->indices);

			pass_in_cluster.setFilterLimitsNegative (true);

			pass_in_cluster.filter(upper_slice_inliers_in_cluster->indices);
	
			std::vector<pcl::PointIndices> lower_clusters_indices_in_cluster;

			std::vector<pcl::PointIndices> upper_clusters_indices_in_cluster;
		
			ec.setIndices(lower_slice_inliers_in_cluster);

			ec.extract(lower_clusters_indices_in_cluster);

			ec.setIndices(upper_slice_inliers_in_cluster);

			ec.extract(upper_clusters_indices_in_cluster);

			for(auto & cl : lower_clusters_indices_in_cluster) 	
			{
				Eigen::Vector4f lower_centroid;

				pcl::compute3DCentroid(*filtered_cloud, cl, lower_centroid);

				PointT p; p.getVector3fMap() = lower_centroid.head(3);

				std::vector<int> k_indices; std::vector<float> k_sqr_distances;

				if( kdtree.nearestKSearch(p, 1, k_indices, k_sqr_distances) == 1) {

					SVVertexProperty vp;

					vp.index = k_indices[0];	

					vp.cluster_indices = cl.indices;

					sv_vertex_t new_vertex = boost::add_vertex(vp, skeleton_graph);

					for(auto & cu : upper_clusters_indices_in_cluster) {

						Eigen::Vector4f upper_centroid;

						pcl::compute3DCentroid(*filtered_cloud, cu, upper_centroid);
				
						PointT p_u; p_u.getVector3fMap() = upper_centroid.head(3); 

						k_indices.clear(); k_sqr_distances.clear();

						if( kdtree.nearestKSearch(p_u, 1, k_indices, k_sqr_distances) == 1 )
							for(int old_vertex=boost::num_vertices(skeleton_graph)-1; old_vertex>=0; old_vertex--) 
								if( skeleton_graph[old_vertex].index == k_indices[0] ) {

									skeleton_graph[old_vertex].parents.push_back(new_vertex);

									skeleton_graph[new_vertex].children.push_back(old_vertex);
									
									sv_edge_t edge; bool edge_added;

									boost::tie(edge, edge_added) = boost::add_edge(old_vertex, new_vertex, skeleton_graph);

									skeleton_graph[edge].weight = (lower_centroid - upper_centroid).norm();

									break;
								}
					}
				}
			}
		}

		uint32_t r = cnt % 2 == 0 ? 255 : 0;

		rgb = r << 16 | 0 << 8 | 255;

		for(auto & i : pre_indices->indices)	
			filtered_cloud->points[i].rgb =  *reinterpret_cast<float*>(&rgb);

		cnt++;
	}

	cout<<endl;

	viewer->addPointCloud(filtered_cloud, "slice cloud");


/*	BGL_FORALL_VERTICES(v, skeleton_graph, sv_graph_t) {

		rgb = 0 << 16 | 255 << 8 | 0;

		filtered_cloud->points[skeleton_graph[v].index].rgb = *reinterpret_cast<float*>(&rgb);

		for(auto & vertex : skeleton_graph[v].children) {

			viewer->addLine(filtered_cloud->points[skeleton_graph[vertex].index], filtered_cloud->points[skeleton_graph[v].index], 1, 0, 0, to_string(cv::getTickCount()));
		}

		for(auto & vertex : skeleton_graph[v].parents) {

			viewer->addLine(filtered_cloud->points[skeleton_graph[vertex].index], filtered_cloud->points[skeleton_graph[v].index], 1, 1, 1, to_string(cv::getTickCount()));
		}
		
		viewer->spin();
		
	}
*/
	BGL_FORALL_EDGES(e, skeleton_graph, sv_graph_t) {

		sv_vertex_t s = boost::source(e, skeleton_graph);

		sv_vertex_t t = boost::target(e, skeleton_graph);

		int s_idx = skeleton_graph[s].index;

		int t_idx = skeleton_graph[t].index;

		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}

	cout<<"initial skeleton\n";
	
//	viewer->spin();

	// remove loops in graph with MST
	std::vector<sv_edge_t> spanning_tree;

	boost::kruskal_minimum_spanning_tree(skeleton_graph, std::back_inserter(spanning_tree), boost::weight_map(boost::get(&SVEdgeProperty::weight, skeleton_graph)) );

	sv_graph_t mst(boost::num_vertices(skeleton_graph));

	BGL_FORALL_VERTICES(v, skeleton_graph, sv_graph_t)
		mst[v] = skeleton_graph[v];

	for(auto & e : spanning_tree)
	{
		sv_vertex_t s = boost::source(e, skeleton_graph);

		sv_vertex_t t = boost::target(e, skeleton_graph);

		sv_edge_t new_e;

		bool edge_added;		

		boost::tie(new_e, edge_added) = boost::add_edge(s, t, mst);

		mst[new_e].weight = skeleton_graph[e].weight;
	}

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) {

		int s_idx = mst[boost::source(e, mst)].index;

		int t_idx = mst[boost::target(e, mst)].index;
		
		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}
	
	cout<<"mst done\n";

//	viewer->spin();


	// prune short branches	
#if 1
	std::set<sv_vertex_t> vertices_to_remove;

	BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

		// process leaf node
		if (boost::out_degree(v, mst) != 1) continue;

		sv_vertex_t cur_v = *(adjacent_vertices(v, mst).first);

		std::vector<sv_vertex_t> visited_vertices;

		visited_vertices.push_back(v);
			
		while (true) {

			const int num_neighbors = boost::out_degree(cur_v, mst);

			if( num_neighbors == 1) {	// leaf node
				
				visited_vertices.push_back(cur_v);
				break;
			}
			else if( num_neighbors == 2 ) { // can continue
				
				BGL_FORALL_ADJ(cur_v, adj, mst, sv_graph_t) {

					if( adj != visited_vertices.back() ) {

						visited_vertices.push_back(cur_v);
						cur_v = adj;
						break;
					}
				}								
 
				continue;
			}
			else 	//intersection 
				break;
		}

		if ( visited_vertices.size() < min_branch_size )
			for(auto & visited_vertex : visited_vertices)
				vertices_to_remove.insert(visited_vertex);
	}


	for(auto iter = vertices_to_remove.begin(); iter != vertices_to_remove.end(); ++iter) 
		boost::clear_vertex(*iter, mst);

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) 
		viewer->addLine(filtered_cloud->points[mst[boost::source(e, mst)].index], filtered_cloud->points[mst[boost::target(e, mst)].index], 1, 1, 1, to_string(cv::getTickCount()));		


	cout<<"prune done\n";
	
//	viewer->spin();
#endif


	// refine branching points
#if 1
	const float stop_branching_dist = max_stem_radius;
	while(true)
	{
		bool modify = false;

		for(int v=boost::num_vertices(mst)-1; v>0; v--) {

			// find branch between leaf and stem
			if(boost::out_degree(v, mst) != 3 || mst[v].children.size() != 2 || mst[v].parents.size() != 1) continue;

			if( mst[mst[v].parents[0]].children.size() > 1 ) continue;

			std::vector<int> indices = mst[v].cluster_indices;

			std::vector<cv::Point3f> cv_points(indices.size());

			cv::Mat labels;

			for (int i = 0; i < cv_points.size(); i++) {

				cv::Point3f point;
				point.x = filtered_cloud->points[indices[i]].x;
				point.y = filtered_cloud->points[indices[i]].y;
				point.z = filtered_cloud->points[indices[i]].z;
				cv_points[i] = point;
			}

			cv::Mat object_centers;
		
			cv::kmeans(cv_points, 2, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 40, 0.001), 10, cv::KMEANS_PP_CENTERS, object_centers);

			cv::Vec3f & center1 = object_centers.at<cv::Vec3f>(0, 0);

			cv::Vec3f & center2 = object_centers.at<cv::Vec3f>(1, 0);

			float dist = std::sqrt( pow(center1[0]-center2[0], 2.f)+pow(center1[1]-center2[1], 2.f)+pow(center1[2]-center2[2], 2.f) );

			if( dist < stop_branching_dist) continue;

			PointT p1, p2;

			p1.x = center1[0]; p1.y = center1[1]; p1.z = center1[2];
			p2.x = center2[0]; p2.y = center2[1]; p2.z = center2[2];

			std::vector<PointT> point_pair(2);
	
			point_pair[0].x = center1[0]; point_pair[0].y = center1[1]; point_pair[0].z = center1[2];
			point_pair[1].x = center2[0]; point_pair[1].y = center2[1]; point_pair[1].z = center2[2];

			uint32_t rgb = 255 << 16 | 255 << 8 | 255;
    
			int p1_child_idx = -1;			

			if( ( p1.getVector3fMap() - p2.getVector3fMap() ).dot( 
			      filtered_cloud->points[mst[mst[v].children[0]].index].getVector3fMap() 
			      - filtered_cloud->points[mst[mst[v].children[1]].index].getVector3fMap() ) 
			     > 0.f )
				p1_child_idx = 0;
			else
				p1_child_idx = 1;
			
			mst[mst[v].parents[0]].children.clear();

			for(int p=0; p<2; p++) {
		
				std::vector<int> k_indices; std::vector<float> k_sqr_distances;

				if( kdtree.nearestKSearch(point_pair[p], 1, k_indices, k_sqr_distances) == 1) {
			
					SVVertexProperty vp;

					vp.index = k_indices[0];	

					for (int i = 0; i < cv_points.size(); i++)
						if( labels.at<int>(i, 0) == p ) 
							vp.cluster_indices.push_back(indices[i]);

					uint32_t rgb_p = ( 255*(1-p) << 16 | 255*p << 8 | 0 );

					for(auto & i : vp.cluster_indices) 
						filtered_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb_p);

					vp.children = mst[v].children;

					vp.parents = mst[v].parents;

					vp.near_junction = true;

					sv_vertex_t new_vertex = boost::add_vertex(vp, mst);

					const int p_child_idx = p == 0 ? p1_child_idx : 1 - p1_child_idx;

					boost::add_edge(new_vertex, mst[v].children[p_child_idx], mst);

					boost::add_edge(new_vertex, mst[v].parents[0], mst);

				//	filtered_cloud->points[k_indices[0]].rgb = *reinterpret_cast<float*>(&rgb);

					mst[mst[v].children[p_child_idx]].parents[0] = new_vertex;

					mst[mst[v].parents[0]].children.push_back(new_vertex);
				}
				else cout<<"kdtree for p fail\n";
			}
						
			boost::clear_vertex(v, mst); 

			mst[v].near_junction = true;

			modify = true;

			break;
		}

#if 0
		viewer->removeAllShapes();

		BGL_FORALL_EDGES(e, mst, sv_graph_t) {

			sv_vertex_t s = boost::source(e, mst);

			sv_vertex_t t = boost::target(e, mst);

			int s_idx = mst[s].index;

			int t_idx = mst[t].index;

			viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
		}

		viewer->updatePointCloud(filtered_cloud, "slice cloud");
		
		viewer->spin();
#endif

		if( !modify ) break;
	}

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t)
		viewer->addLine(filtered_cloud->points[mst[boost::source(e, mst)].index], filtered_cloud->points[mst[boost::target(e, mst)].index], 1, 1, 1, to_string(cv::getTickCount()));		


	viewer->updatePointCloud(filtered_cloud, "slice cloud");

	cout<<"refine branching points done\n";
	
//	viewer->spin();
#endif

	// prune short branches	
#if 0
	vertices_to_remove.clear();

	BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

		// process leaf node
		if (boost::out_degree(v, mst) != 1) continue;

		sv_vertex_t cur_v = *(adjacent_vertices(v, mst).first);

		std::vector<sv_vertex_t> visited_vertices;

		visited_vertices.push_back(v);
			
		while (true) {

			const int num_neighbors = boost::out_degree(cur_v, mst);

			if( num_neighbors == 1) {	// leaf node
				
				visited_vertices.push_back(cur_v);
				break;
			}
			else if( num_neighbors == 2 ) { // can continue
				
				BGL_FORALL_ADJ(cur_v, adj, mst, sv_graph_t) {

					if( adj != visited_vertices.back() ) {

						visited_vertices.push_back(cur_v);
						cur_v = adj;
						break;
					}
				}								
 
				continue;
			}
			else 	//intersection 
				break;
		}

		if ( visited_vertices.size() < min_branch_size )
			for(auto & visited_vertex : visited_vertices)
				vertices_to_remove.insert(visited_vertex);
	}


	for(auto iter = vertices_to_remove.begin(); iter != vertices_to_remove.end(); ++iter) 
		boost::clear_vertex(*iter, mst);

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) 
		viewer->addLine(filtered_cloud->points[mst[boost::source(e, mst)].index], filtered_cloud->points[mst[boost::target(e, mst)].index], 1, 1, 1, to_string(cv::getTickCount()));		


	cout<<"prune done\n";
	
//	viewer->spin();
#endif

	// ***************************partition segments*************************************
	rgb = 255 << 16 | 255 << 8 | 0;

	std::vector<std::vector<sv_vertex_t>> segment_vec;

	std::vector<bool> visited_map(boost::num_vertices(mst), false);

	BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

		const int degree = boost::out_degree(v, mst);

		if( degree == 2 || degree == 0) continue;
		
		visited_map[v] = true;

		BGL_FORALL_ADJ(v, adj, mst, sv_graph_t) {
	
			if( visited_map[adj] || boost::out_degree(adj, mst) > 2 ) 
				continue;

			sv_vertex_t cur_v = adj;

			std::vector<sv_vertex_t> segment;

			if(degree == 1) 
				segment.push_back(v);
			else
				segment.push_back(cur_v);

			visited_map[cur_v] = true;

			bool first_time = true;

			while(true) {

				const int num_neighbors = boost::out_degree(cur_v, mst);

				if( num_neighbors == 1) { //another leaf node
			
					segment.push_back(cur_v);

					visited_map[cur_v] = 1;

					break;
				}
				else if( num_neighbors == 2 ) {

					BGL_FORALL_ADJ(cur_v, adj_cur, mst, sv_graph_t) {
						
						if(degree != 1) {

							if( first_time ) { // first time

								if( adj_cur != v ) { 

									cur_v = adj_cur;

									first_time = false;

									break;
								}
							}
							else {
							
								if( adj_cur != segment.back() ) {

									segment.push_back(cur_v);

									visited_map[cur_v] = true;

									cur_v = adj_cur;

									break;
								}
							}
						}
						else {	// degree == 1
	
							if( adj_cur != segment.back() ) {

								segment.push_back(cur_v);

								visited_map[cur_v] = true;

								cur_v = adj_cur;

								break;
							}							
						}
					}									
	 
					continue;
				}
				else // find a branching point
					break;
			}

			segment_vec.push_back(segment);


//			for(auto & v : segment)	filtered_cloud->points[mst[v].index].rgb = *reinterpret_cast<float*> (&rgb);

		//	viewer->updatePointCloud(filtered_cloud, "slice cloud");
	
		//	viewer->spin();
		}
	}

	rgb = 255 << 16 | 255 << 8 | 0;

//	for(auto & segment : segment_vec)
//		for(auto & v : segment) 
//			filtered_cloud->points[mst[v].index].rgb = *reinterpret_cast<float*> (&rgb);

	viewer->updatePointCloud(filtered_cloud, "slice cloud");

	cout<<"partition segments done\n";
	
	//viewer->spin();



	// break V type segments
	point_indices->indices.clear();
	std::vector<std::pair<int, int>> segment_to_break_indices;	//with index of the breaking node

	int seg_idx = 0;

	for(auto & segment : segment_vec) {

		if( segment.size() > 2*min_branch_size) {

			if( filtered_cloud->points[mst[segment[0]].index].x - filtered_cloud->points[mst[segment[1]].index].x < 0.f 
			  &&filtered_cloud->points[mst[segment.back()].index].x - filtered_cloud->points[mst[segment[segment.size()-2]].index].x < 0.f ) {

				int break_idx = -1;
			
				for(int i=min_branch_size; i<segment.size()-min_branch_size; i++) {
					
					float x = filtered_cloud->points[mst[segment[i]].index].x;

					float xm = filtered_cloud->points[mst[segment[i-1]].index].x;

					float xp = filtered_cloud->points[mst[segment[i+1]].index].x;

					if( x > xm && x> xp ) {

						break_idx = i;

						//boost::clear_vertex(segment[break_idx], mst); //dont break connection in graph, break connection in segments
						
						break;
					}
				}

				if( break_idx != -1 )
					segment_to_break_indices.push_back(std::make_pair(seg_idx, break_idx));

//				for(auto & v : segment) filtered_cloud->points[mst[v].index].rgb = *reinterpret_cast<float*> (&rgb); 
			}
		}

		++seg_idx;
	}

	for(auto & pair : segment_to_break_indices) {

		const int seg_idx = pair.first;

		const int break_idx = pair.second;

		std::vector<sv_vertex_t> new_segment;

		new_segment.insert(new_segment.end(), segment_vec[seg_idx].begin()+break_idx, segment_vec[seg_idx].end());

		segment_vec.push_back(new_segment);

		segment_vec[seg_idx].erase(segment_vec[seg_idx].begin()+break_idx, segment_vec[seg_idx].end());
	}

	viewer->updatePointCloud(filtered_cloud, "slice cloud");

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) {

		int s_idx = mst[boost::source(e, mst)].index;

		int t_idx = mst[boost::target(e, mst)].index;

		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}

	

	//************************ detect stems using 3d hough line*******************************
	centroid_cloud->clear();

	BGL_FORALL_VERTICES(v, mst, sv_graph_t)
		if( boost::out_degree(v, mst) > 0 
		  // && mst[v].max_width < 0.04f
		  ) 
			centroid_cloud->push_back(filtered_cloud->points[mst[v].index]);

	cout<<"centroids size "<<centroid_cloud->size()<<"\n";

	std::vector<Eigen::Vector3f> a_vec, b_vec;

	Hough3DLine(centroid_cloud, viewer, a_vec, b_vec, max_stem_radius, h_nlines, h_minvotes, h_granularity, stem_verticality);

	//viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) {

		const int s_idx = mst[boost::source(e, mst)].index;

		const int t_idx = mst[boost::target(e, mst)].index;

		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}

	//std::cout<<"num line detected "<<a_vec.size()<<"\n";

	float line_len = max_pt(0) - min_pt(0);//1.1f;

	std::vector<int> valid_stem_line_indices;

	for(int i=0; i<a_vec.size(); i++) {

		bool valid_line = true;

		//check root position
//		if( a_vec[i](2) > 1.3f || a_vec[i](2) < 0.7f ) valid_line = false;
		
		for (int j=0; valid_line && j<i; j++) {

			std::pair<Eigen::Vector3f, Eigen::Vector3f> segment1(a_vec[i], a_vec[i] + line_len*b_vec[i]);

			std::pair<Eigen::Vector3f, Eigen::Vector3f> segment2(a_vec[j], a_vec[j] + line_len*b_vec[j]);

			float l2l_dist = dist3D_Segment_to_Segment(segment1, segment2);

			if(l2l_dist < 0.08f) 
				valid_line = false;
		}
		
		pcl::PointXYZ p1(a_vec[i](0), a_vec[i](1), a_vec[i](2));

		Eigen::Vector3f vector = a_vec[i] + line_len*b_vec[i];

		pcl::PointXYZ p2(vector(0), vector(1), vector(2));

		if(!valid_line)	{

			viewer->addLine(p1, p2, 0.5, 0.5, 0.5, "line"+to_string(cv::getTickCount()));	

			continue;
		}		
		else 
			viewer->addLine(p1, p2, 1, 0, 0, "line"+to_string(cv::getTickCount()));	

		valid_stem_line_indices.push_back(i);
	}

	//viewer->spin();


	// label stem inlier slices
	std::vector<int> stem_inlier_map(boost::num_vertices(mst), -1);

	rgb = 255<<16 | 255<<8 | 255;

	BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

		if(boost::out_degree(v, mst) == 0)
			continue;

		float min_dist = std::numeric_limits<float>::max();

		int best_idx = -1;

		for(auto & stem_line_index : valid_stem_line_indices) {
		
			Eigen::Vector3f vector = filtered_cloud->points[mst[v].index].getVector3fMap() - a_vec[stem_line_index];

			float point_to_line_dist = (vector - vector.dot(b_vec[stem_line_index])*b_vec[stem_line_index]).norm();

			if( point_to_line_dist < min_dist) {
		
				min_dist = point_to_line_dist;

				best_idx = stem_line_index;
			}
		}

		if(best_idx != -1 && min_dist < max_stem_radius) {
		
			stem_inlier_map[v] = best_idx;

			
			for(auto & i : mst[v].cluster_indices) filtered_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);
		}			
	}

	viewer->updatePointCloud(filtered_cloud, "slice cloud");

	
#if 0
	// find vertices on stems
	point_indices->indices.resize(3);

	rgb = 180<<16 | 255<<8 | 255;

	for(auto & segment : segment_vec) {

		if(segment.size() < min_branch_size)
			continue;

		for(auto niter = segment.begin()+1; niter != segment.end()-1; ++niter) {

			if( mst[*niter].near_junction )
				continue;
			
			Eigen::Vector3f pre_p = filtered_cloud->points[mst[*(niter-1)].index].getVector3fMap();

			Eigen::Vector3f cur_p = filtered_cloud->points[mst[*(niter)].index].getVector3fMap();

			Eigen::Vector3f nex_p = filtered_cloud->points[mst[*(niter+1)].index].getVector3fMap();

			for(auto & stem_line_index: valid_stem_line_indices) {

				Eigen::Vector3f base_to_centroid = cur_p - a_vec[stem_line_index];

				const float centroid_to_stem_line_dist = (base_to_centroid - base_to_centroid.dot(b_vec[stem_line_index])*b_vec[stem_line_index]).norm();
		
				if( centroid_to_stem_line_dist < 0.03f 
				    && std::abs((pre_p-cur_p).normalized().dot(b_vec[stem_line_index])) > 0.8
				    && std::abs((nex_p-cur_p).normalized().dot(b_vec[stem_line_index])) > 0.8
				  ) {
					
					point_indices->indices = mst[*niter].cluster_indices;

					pca.setIndices(point_indices);

					Eigen::Vector3f eigenvalues = pca.getEigenValues();

					PointCloudT::Ptr origional(new PointCloudT);

					pcl::copyPointCloud(*filtered_cloud, point_indices->indices, *origional);

					PointCloudT::Ptr projected(new PointCloudT);

					pca.project(*origional, *projected);

					Eigen::Vector4f min_pt, max_pt;

					pcl::getMinMax3D(*projected, min_pt, max_pt);

					//cout<< (max_pt-min_pt).transpose()<<"\n";

					float surface_variation = eigenvalues(2)/eigenvalues.sum();

					//cout<<surface_variation<<"\n";

					if( surface_variation > 0.1) 
						break;


					for(auto & index : mst[*niter].cluster_indices) 
						filtered_cloud->points[index].rgb = *reinterpret_cast<float*>(&rgb);
					
		
					//stem_segment_indices.push_back(idx);

				//	viewer->updatePointCloud(filtered_cloud, "slice cloud");
				//	viewer->spin();
			
					break;
				}
			}

		}

	}

	viewer->updatePointCloud(filtered_cloud, "slice cloud");
#endif


	// plant segmentation
#if 1
	std::vector<int> stem_id_for_segment(segment_vec.size(), -1);

	std::vector<bool> first_node_end_map(segment_vec.size(), true);	// is segment.front() the node closest to the stem line?

	rgb = 255<<16 | 255<<8 | 0;

	float max_gap_dist = 0.1f;

	seg_idx = 0;

	for(auto & segment : segment_vec) {

		point_indices->indices.clear();

		for(auto & v : segment)
			point_indices->indices.insert(point_indices->indices.end(), mst[v].cluster_indices.begin(), mst[v].cluster_indices.end());

		//for( auto & i : point_indices->indices ) filtered_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);

		pca.setIndices(point_indices);

		Eigen::Vector3f major_vector = pca.getEigenVectors().col(0);
		Eigen::Vector3f mean = pca.getMean().head(3);

		float min_dist_to_stem = std::numeric_limits<float>::max();

		int best_stem_idx = -1;

		int first_node = true;

		for(auto & stem_line_index: valid_stem_line_indices) {

			Eigen::Vector3f vector = filtered_cloud->points[mst[segment.front()].index].getVector3fMap() - a_vec[stem_line_index];

			float point_to_stem_line_dist = (vector - vector.dot(b_vec[stem_line_index])*b_vec[stem_line_index]).norm();
	
			if( point_to_stem_line_dist < min_dist_to_stem) {
		
				min_dist_to_stem = point_to_stem_line_dist;

				best_stem_idx = stem_line_index;

				first_node = true;
			}

			vector = filtered_cloud->points[mst[segment.back()].index].getVector3fMap() - a_vec[stem_line_index];

			point_to_stem_line_dist = (vector - vector.dot(b_vec[stem_line_index])*b_vec[stem_line_index]).norm();
	
			if( point_to_stem_line_dist < min_dist_to_stem) {
		
				min_dist_to_stem = point_to_stem_line_dist;

				best_stem_idx = stem_line_index;

				first_node = false;
			}
		}

		if(best_stem_idx != -1 
		&& min_dist_to_stem < max_gap_dist) {

			stem_id_for_segment[seg_idx] = best_stem_idx;
			
			first_node_end_map[seg_idx] = first_node;

		//	if(first_node) filtered_cloud->points[mst[segment.front()].index].rgb = rgb;
		//	else filtered_cloud->points[mst[segment.back()].index].rgb = rgb;	
		}
			
		++seg_idx;
	}


	std::vector<float> random_color_for_stems(valid_stem_line_indices.size());

	for(int i=0; i<valid_stem_line_indices.size(); i++) {

		rgb = std::max(rand()%255, 50) << 16 | std::max(rand()%255, 50) << 8 | std::max(rand()%255, 50);

		random_color_for_stems[i] = *reinterpret_cast<float*>(&rgb);
	}
#endif


#if 0
	for(int i=0; i<segment_vec.size(); i++) {

		if(stem_id_for_segment[i] == -1) 
			continue;

		for(auto & v : segment_vec[i])
			for(auto & index : mst[v].cluster_indices) 
				filtered_cloud->points[index].rgb = random_color_for_stems[stem_id_for_segment[i]];
	}	

	viewer->updatePointCloud(filtered_cloud, "slice cloud");
#endif

	// break connection of broken leaves
	seg_idx = 0;

	rgb = 255;

	std::vector<sv_vertex_t> vertices_to_clear;

	for(auto & segment : segment_vec) {

		bool contain_stem = false;

		for(auto & v : segment) {
	
			if(stem_inlier_map[v] != -1) {

				contain_stem = true;
				break;
			}
		}

		if(contain_stem) {
		
			++seg_idx;
			continue;
		}

		const int front_degree = boost::out_degree(segment.front(), mst);

		const int back_degree = boost::out_degree(segment.back(), mst);

		if( front_degree == 1 && back_degree == 2 ) {

			Eigen::Vector3f end_point = filtered_cloud->points[mst[segment.front()].index].getVector3fMap();

			Eigen::Vector3f connect_point = filtered_cloud->points[mst[segment.back()].index].getVector3fMap();

			if( end_point(0) > connect_point(0) || first_node_end_map[seg_idx])
				vertices_to_clear.push_back(segment.back());
			
		//	for(auto & v : segment) 
		//		for(auto & i : mst[v].cluster_indices)
		//				filtered_cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);
		}
		else if( front_degree == 2 && back_degree == 1 ) {

			Eigen::Vector3f end_point = filtered_cloud->points[mst[segment.back()].index].getVector3fMap();

			Eigen::Vector3f connect_point = filtered_cloud->points[mst[segment.front()].index].getVector3fMap();

			if( end_point(0) > connect_point(0) || !first_node_end_map[seg_idx])
				vertices_to_clear.push_back(segment.front());
		}	

		++seg_idx;
	}

	for(auto iter = vertices_to_clear.begin(); iter!= vertices_to_clear.end(); ++iter)
		boost::clear_vertex(*iter, mst);


	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) {

		int s_idx = mst[boost::source(e, mst)].index;

		int t_idx = mst[boost::target(e, mst)].index;

		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}


	std::vector<int> cc(boost::num_vertices(mst));
	
	int num = boost::connected_components(mst, &cc[0]);


	//**********************************find plant orientation**************************************************
	Eigen::Vector3f point_on_plane = Eigen::Vector3f::Zero();

	point_on_plane(0) = -plane_coeffs(3)/plane_coeffs(0);
	
	std::map<int, Eigen::Vector3f> plant_orientation_map;

	for(auto & stem_line_index : valid_stem_line_indices) {

		point_indices->indices.clear();

		Eigen::Vector3f & base_point = a_vec[stem_line_index];

		Eigen::Vector3f & line_dir = b_vec[stem_line_index];

		BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

			Eigen::Vector3f vector = filtered_cloud->points[mst[v].index].getVector3fMap() - base_point;

			float point_to_line_dist = (vector - vector.dot(line_dir)*line_dir).norm();

			if(point_to_line_dist < 0.1f)
				point_indices->indices.push_back(mst[v].index);
		}

		pca.setIndices(point_indices);

		Eigen::Vector3f middle_vector = pca.getEigenVectors().col(1);

		if(middle_vector(2) < 0.f) 
			middle_vector *= -1.0f;

		float d = (point_on_plane - base_point).dot(plane_coeffs.head<3>())/(line_dir.dot(plane_coeffs.head<3>()));

		Eigen::Vector3f line_plane_intersection = base_point + d*line_dir;

		PointT p0; p0.getVector3fMap() = line_plane_intersection;

		middle_vector = middle_vector - middle_vector.dot(line_dir)*line_dir;

		middle_vector.normalize();

		plant_orientation_map.insert( std::pair<int, Eigen::Vector3f>(stem_line_index, middle_vector) );		

		PointT p1; p1.getVector3fMap() = line_plane_intersection - 0.1f*middle_vector;

		viewer->addLine(p0, p1, 1,1,0, "line"+to_string(cv::getTickCount()));
	}
	
	//*************************************************plant height***************************************************
	std::map<int, Eigen::Vector3f> plant_base_map;

	for(auto & stem_line_id : valid_stem_line_indices) {

		//compute intersection of stem line and ground plane
		Eigen::Vector3f & base_point = a_vec[stem_line_id];

		Eigen::Vector3f & line_dir = b_vec[stem_line_id];
	
		float d = (point_on_plane - base_point).dot(plane_coeffs.head<3>())/(line_dir.dot(plane_coeffs.head<3>()));

		Eigen::Vector3f line_plane_intersection = base_point + d*line_dir;

		plant_base_map.insert( std::pair<int, Eigen::Vector3f>(stem_line_id, line_plane_intersection) );

		PointT p0, p1;
	
		p0.getVector3fMap() = line_plane_intersection;

		p1.getVector3fMap() = line_plane_intersection + line_len*b_vec[stem_line_id];

		viewer->addLine(p0, p1, 1, 0, 0, "line"+to_string(cv::getTickCount()));

		// search plant top vertex
		float max_dist = 0.0f;

		Eigen::Vector3f best_plant_top;

		BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

			if(stem_line_id != stem_inlier_map[v]) 
				continue;
			
			Eigen::Vector3f p = filtered_cloud->points[mst[v].index].getVector3fMap();

			// projection on line
			p = (p-base_point).dot(line_dir)*line_dir + base_point;

			float tmp_dist = (p-line_plane_intersection).norm();

			if( tmp_dist > max_dist ) {

				max_dist = tmp_dist;

				best_plant_top = p;
			}
		}

		if(max_dist != 0.0f) {
			
			PointT pt; pt.getVector3fMap() = best_plant_top;
			
			viewer->addText3D(to_string(max_dist), pt, 0.01);
		}
	}


	//**********************************stem diameter********************************
	for(auto & stem_line_id : valid_stem_line_indices) {
	
		Eigen::Vector3f & plant_base = plant_base_map.find(stem_line_id)->second;
	
		BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

			if( stem_inlier_map[v] != stem_line_id )
				continue;
				
			if( mst[v].near_junction )
				continue;
				
			//Eigen::Vector3f slice_centroid;// = filtered_cloud->points[mst[v].index].getVector3fMap();
			
			Eigen::Vector4f centroid4;
			
			Eigen::Vector3f slice_centroid;
			
			pcl::compute3DCentroid(*filtered_cloud, mst[v].cluster_indices, centroid4);
			
			slice_centroid = centroid4.head(3);
			
			float slice_height = (slice_centroid - plant_base).norm();
			
			if(slice_height > 0.2f)
				continue;

			Eigen::Vector3f line_dir = -b_vec[stem_line_id];	// align with world frame
			
			cout<<"line dir "<<line_dir.transpose()<<"\n";

			PointCloudT::Ptr slice_cloud(new PointCloudT);

			pcl::copyPointCloud(*filtered_cloud, mst[v].cluster_indices, *slice_cloud);
			
			Eigen::Vector3f & plant_ori = plant_orientation_map.find(stem_line_id)->second;
			
			Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

			transform.col(0).head<3>() = line_dir;
			//transform.col(2).head<3>() = plant_orientation_map.find(stem_line_id)->second;	// plant orientation
			transform.col(2).head<3>() = (slice_centroid - slice_centroid.dot(line_dir)*line_dir).normalized();	//camera view
			transform.col(1).head<3>() = transform.col(2).head<3>().cross( transform.col(0).head<3>() );
			transform.col(3).head<3>() = slice_centroid;

			PointCloudT::Ptr slice_cloud_view(new PointCloudT);
			
			pcl::transformPointCloud(*slice_cloud, *slice_cloud_view, transform.inverse());

			Eigen::Vector3f ellipse_center;
			
			ellipse_center(0) = 0.0f;
			
			Eigen::Vector4f min, max, min1, max1;
			
			std::vector<int> one_side_indices;
			
		
			pass.setInputCloud(slice_cloud_view);
			pass.setFilterFieldName ("y");
			pass.setFilterLimits (-1.0, .0);
			pass.filter (one_side_indices);
			
			pcl::getMinMax3D(*slice_cloud_view, one_side_indices, min, max);
			
			pass.setFilterLimits (0., 1.0);
			pass.setFilterLimitsNegative (true);
			pass.filter (one_side_indices);
			
			pcl::getMinMax3D(*slice_cloud_view, one_side_indices, min1, max1);
			
			ellipse_center(2) = (max1(2) + max(2))*0.5f;
		
			pcl::getMinMax3D(*slice_cloud_view, min, max);
			
			//if( (max-min)(1) > 2.0f*max_stem_radius )
			//	continue;
				
			ellipse_center(1) = (max(1)+min(1))*0.5f;
			
			ellipse_center = transform.topLeftCorner<3,3>()*ellipse_center + transform.col(3).head<3>();
			
			transform.col(2).head<3>() = plant_ori;	// plant orientation
			transform.col(1).head<3>() = transform.col(2).head<3>().cross( transform.col(0).head<3>() );
			transform.col(3).head<3>() = ellipse_center;
			
			pcl::transformPointCloud(*slice_cloud, *slice_cloud_view, transform.inverse());
			

			
			float short_diameter = 0.f;
			
			for(auto & p : slice_cloud_view->points) 
				short_diameter += std::sqrt(4.f*p.y*p.y+p.z*p.z);
				
			short_diameter /= slice_cloud_view->size();
			
			cout<<"short diameter "<<short_diameter<<"\n";
			
			viewer->removeShape("line");
			viewer->removePointCloud("stem");
			viewer->addPointCloud(slice_cloud_view, "stem");
			viewer->addLine(filtered_cloud->points[mst[v].index], pcl::PointXYZ(0,0,0), 1,1,1, "line");
			viewer->spin();
		}
	}	

	

#if 1
	// leaf segmentation
	std::vector<bool> leaf_map(segment_vec.size(), false);

	seg_idx = 0;

	const float stem_radius = max_stem_radius;//0.03f;

	rgb = 255<<16 | 255<<8 | 0;	

	for(auto & segment : segment_vec) {

		if(stem_id_for_segment[seg_idx] == -1) {

			++seg_idx;
			continue;
		}

		bool segment_connected_to_stem = false;

		BGL_FORALL_VERTICES(v, mst, sv_graph_t) {

			if( stem_inlier_map[v] != -1 && cc[v] == cc[segment.front()] ) {

				segment_connected_to_stem = true;
				break;
			}
		}

		if(!segment_connected_to_stem) {

			++seg_idx;
			continue;
		}

		if(stem_inlier_map[segment.front()] != -1 && stem_inlier_map[segment.back()] != -1) {

			++seg_idx;
			continue;
		}
		

		const int stem_line_index = stem_id_for_segment[seg_idx];
	
		point_indices->indices.clear();

		for(auto & v : segment) 
			if( stem_inlier_map[v] == -1 ) 
				point_indices->indices.insert(point_indices->indices.end(), mst[v].cluster_indices.begin(), mst[v].cluster_indices.end());
				//point_indices->indices.push_back(mst[v].index);
		
		if( point_indices->indices.size() < 3 ) {

			++seg_idx;
			continue;
		}

		for(auto & i : point_indices->indices) filtered_cloud->points[i].rgb = rgb;

		Eigen::Vector3f & base_point = a_vec[stem_line_index];

		Eigen::Vector3f & line_dir = b_vec[stem_line_index];
			
		pca.setIndices(point_indices);

		Eigen::Vector3f major_vector = pca.getEigenVectors().col(0);

		Eigen::Vector3f mean = pca.getMean().head(3);

		Eigen::Vector3f vector = mean - base_point;

		float point_to_line_dist = (vector - vector.dot(line_dir)*line_dir).norm();

		float abs_cosine = std::abs(major_vector.dot(line_dir));

		if ( point_to_line_dist > stem_radius
		     && abs_cosine < 0.99f 	
		) {

			for(auto & i : point_indices->indices) filtered_cloud->points[i].rgb = rgb*0.5f;

			leaf_map[seg_idx] = true;

			//measure leaf angle

			point_indices->indices.clear();
			
			for(auto & v : segment) {

				vector = filtered_cloud->points[mst[v].index].getVector3fMap() - base_point;

				point_to_line_dist = (vector - vector.dot(line_dir)*line_dir).norm();

				if( point_to_line_dist < 0.1f && stem_inlier_map[v] == -1 )
					point_indices->indices.push_back(mst[v].index);
			}

			if(point_indices->indices.size() > 2) {

				pca.setIndices(point_indices);

				major_vector = pca.getEigenVectors().col(0);

				mean = pca.getMean().head(3);

				if(major_vector(0) > 0.f) major_vector *= -1.0f;

				abs_cosine = std::abs(major_vector.dot(line_dir));

				float line_to_line_dist = std::abs( (mean-base_point).dot(line_dir.cross(major_vector)) ) / (line_dir.cross(major_vector)).norm();

				if( line_to_line_dist < stem_radius ) 
				{
					int start_v = -1;

					if( first_node_end_map[seg_idx] ) {

						for(auto iter=segment.begin(); iter != segment.end(); ++iter) {

							if( stem_inlier_map[*iter] != -1 )
								continue;

							start_v = *iter;
					
							break;
						}
					}
					else {
						for(auto iter=segment.rbegin(); iter != segment.rend(); ++iter) {

							if( stem_inlier_map[*iter] != -1 )
								continue;

							start_v = *iter;
					
							break;
						}
					}

					if(start_v != -1) {

						PointT start_point = filtered_cloud->points[mst[start_v].index];

						PointT end_point; end_point.getVector3fMap() = start_point.getVector3fMap() + line_dir*0.05f;

						float rgb_f = shortRainbowColorMap(1.f - abs_cosine, 1.0f-0.99f, 1.0f-0.6f);

						uint32_t rgb_int = *reinterpret_cast<uint32_t*>(&rgb_f);

						float r = ((unsigned char)(rgb_int >> 16))/255.f;
						float g = ((unsigned char)(rgb_int >> 8))/255.f;
						float b = ((unsigned char)(rgb_int))/255.f;

						viewer->addLine( start_point, end_point, r, g, b, "line"+to_string(cv::getTickCount()));

						end_point.getVector3fMap() = start_point.getVector3fMap() + major_vector*0.1f;

						viewer->addLine( start_point, end_point, r, g, b, "line"+to_string(cv::getTickCount()));

						float angle = acos(abs_cosine)/M_PI*180.f;	//0~90 degrees

						viewer->addText3D(to_string(angle), end_point, 0.01, r, g, b);
					}
				}
			}

//			for(auto & v : segment)	for(auto & index : mst[v].cluster_indices) filtered_cloud->points[index].rgb = random_color_for_stems[stem_id_for_segment[seg_idx]];
		}

		++seg_idx;
	}


	viewer->updatePointCloud(filtered_cloud, "slice cloud");
#endif



#if 0
	// break nonlinear segments

	viewer->updatePointCloud(filtered_cloud, "slice cloud");
	//viewer->spin();


	std::vector<std::pair<int, int>> segment_to_break_indices;	//with index of the breaking node

	pcl::PCA<PointT> pca;

	pca.setInputCloud(filtered_cloud);

	pcl::PointIndices::Ptr point_indices(new pcl::PointIndices);

	pcl::PointIndices::Ptr point_indices_complement(new pcl::PointIndices);

	int seg_idx = 0;

	for(auto & segment : segment_vec) {

		if( segment.size() > 20 ) {

			int break_idx = -1;
			float min_total_curvature = 2.f;

			for(int ni=4; ni<segment.size()-4; ni++) {

				point_indices->indices.resize(ni);

				point_indices_complement->indices.resize(segment.size() - ni);

				for(int j=0; j<ni; j++) 
					point_indices->indices[j] = mst[segment[j]].index;

				for(int j=0; j<point_indices_complement->indices.size(); j++) 
					point_indices_complement->indices[j] = mst[segment[j+ni]].index;

				pca.setIndices(point_indices);

				Eigen::Vector3f first = pca.getEigenValues().head(3);
				
				pca.setIndices(point_indices_complement);

				Eigen::Vector3f second = pca.getEigenValues().head(3);

				float tmp_total_curvature = first(0)/first.sum() + second(0)/second.sum();

				if( tmp_total_curvature < min_total_curvature ) {

					min_total_curvature = tmp_total_curvature;

					break_idx = ni;
				}
			}

			cout<<"min_total_curvature "<<min_total_curvature<<"\n";

			if(break_idx != -1 && min_total_curvature < 1.8) {

				segment_to_break_indices.push_back(std::make_pair(seg_idx, break_idx));

				boost::clear_vertex(segment[break_idx], mst);
			}
		}

		++seg_idx;
	}

	viewer->removeAllShapes();

	BGL_FORALL_EDGES(e, mst, sv_graph_t) {

		int s_idx = mst[boost::source(e, mst)].index;

		int t_idx = mst[boost::target(e, mst)].index;

		viewer->addLine(filtered_cloud->points[s_idx], filtered_cloud->points[t_idx], 1, 1, 1, to_string(cv::getTickCount()));		
	}

	
	for(auto & pair : segment_to_break_indices) {

		const int seg_idx = pair.first;
		const int break_idx = pair.second;

		std::vector<sv_vertex_t> new_segment;

		new_segment.insert(new_segment.end(), segment_vec[seg_idx].begin()+break_idx, segment_vec[seg_idx].end());

		segment_vec.push_back(new_segment);

		segment_vec[seg_idx].erase(segment_vec[seg_idx].begin()+break_idx, segment_vec[seg_idx].end());
	}
#endif


#if 0
	// classify segments to good leaf, stem, or garbage
	std::vector<int> stem_segment_indices;

	int idx = 0;

	for(auto & segment : segment_vec) {

		if( segment.size() < 2*min_branch_size ) {

			idx++;
			continue;
		}

		point_indices->indices.resize(segment.size());

		for(int i=0; i<segment.size(); i++) point_indices->indices[i] = mst[segment[i]].index;

		pca.setIndices(point_indices);

#if 0
		Eigen::Vector3f eigen_values = pca.getEigenValues();

		float linearity = (eigen_values(0)-eigen_values(1))/eigen_values(0);

		cout<<"linearity "<< linearity<<"\n";

		if( linearity < min_linearity ) {

			uint32_t rgb = 255 << 16 | 255 << 8 | 255;

			for(auto & v : segment) {

				for(auto & index : mst[v].cluster_indices)
					filtered_cloud->points[index].rgb = *reinterpret_cast<float*> (&rgb);
			}

#if 0
			PointCloudT::Ptr segment_cloud(new PointCloudT);

			pcl::copyPointCloud(*filtered_cloud, point_indices, *segment_cloud);

			std::vector<Eigen::Vector3f> point_vec, vector_vec;

			cout<<"segment cloud size " << segment_cloud->size()<<"\n";
			
			Hough3DLine(segment_cloud, viewer, point_vec, vector_vec, 0.02, 2, 5, 5);

			cout<<point_vec.size()<<"\n";

			if( point_vec.size() == 2 ) {
			
				for(int i=0; i<2; i++) {

					pcl::PointXYZ p1(point_vec[i](0), point_vec[i](1), point_vec[i](2));	

					Eigen::Vector3f vector = point_vec[i] + line_len*vector_vec[i];

					pcl::PointXYZ p2(vector(0), vector(1), vector(2));

					viewer->addLine(p1, p2, 0, 0, 1, "line"+to_string(cv::getTickCount()));	
				}
			}
#endif
			
		}
#endif
	

		Eigen::Vector3f major_eigen_vector = pca.getEigenVectors().col(0);

		Eigen::Vector3f mean = pca.getMean().head(3);

		for(auto & stem_line_index: valid_stem_line_indices) {

			Eigen::Vector3f base_to_centroid = mean - a_vec[stem_line_index];

			const float centroid_to_stem_line_dist = (base_to_centroid - base_to_centroid.dot(b_vec[stem_line_index])*b_vec[stem_line_index]).norm();
		
			if( centroid_to_stem_line_dist < 0.1f /* && std::abs(major_eigen_vector.dot(b_vec[stem_line_index])) > 0.9*/) {


				
		
				stem_segment_indices.push_back(idx);
			
				break;
			}
		}

		idx++;
	}
	

	rgb = 255 << 16 | 255 << 8 | 0;

	for(auto & segment_id : stem_segment_indices)
		for(auto & v : segment_vec[segment_id])
			for(auto & i : mst[v].cluster_indices) 
				filtered_cloud->points[i].rgb = *reinterpret_cast<float*> (&rgb);

	viewer->updatePointCloud(filtered_cloud, "slice cloud");
	
#endif	

	viewer->spin();
	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

}


void spectralCluster(PointCloudT::Ptr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer) {

	if(cloud->size() < 100) return;

	pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
	super.setInputCloud(cloud);
	super.setColorImportance (color_importance);
	super.setSpatialImportance (spatial_importance);
	super.setNormalImportance (normal_importance);

	std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

	super.extract(supervoxel_clusters);

	pcl::console::print_info("Found %d supervoxels\n", supervoxel_clusters.size ());

	if(supervoxel_clusters.size() < 10) return;

	Graph supervoxel_adjacency_list;

	super.getSupervoxelAdjacencyList(supervoxel_adjacency_list);

	PointLCloudT::Ptr color_cloud = super.getLabeledCloud();

	sv_graph_t sv_graph;

	std::map<uint32_t, sv_vertex_t> label_ID_map;

	BGL_FORALL_VERTICES(vertex, supervoxel_adjacency_list, Graph) {

		sv_vertex_t v = boost::add_vertex(sv_graph);

		sv_graph[v].supervoxel_label = supervoxel_adjacency_list[vertex];

		sv_graph[v].supervoxel = supervoxel_clusters.at(sv_graph[v].supervoxel_label);

		label_ID_map.insert(std::make_pair(supervoxel_adjacency_list[vertex], v));
	}

	BGL_FORALL_EDGES(edge, supervoxel_adjacency_list, Graph) {

		Voxel s = boost::source(edge, supervoxel_adjacency_list);
		Voxel t = boost::target(edge, supervoxel_adjacency_list);

		uint32_t s_l = supervoxel_adjacency_list[s];
		uint32_t t_l = supervoxel_adjacency_list[t];

		sv_vertex_t sv_s = (label_ID_map.find(s_l))->second;
		sv_vertex_t sv_t = (label_ID_map.find(t_l))->second;

		sv_edge_t sv_edge;
		bool edge_added;

		boost::tie(sv_edge, edge_added) = boost::add_edge(sv_s, sv_t, sv_graph);

		if(edge_added) {

			pcl::Supervoxel<PointT>::Ptr svs = supervoxel_clusters.at(s_l);
			pcl::Supervoxel<PointT>::Ptr svt = supervoxel_clusters.at(t_l);

			sv_graph[sv_edge].weight = supervoxel_adjacency_list[edge];//*( 1.0f- std::abs(dir(0)) );
		}
	}


	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t) {

		sv_vertex_t s = boost::source(edge, sv_graph);
		sv_vertex_t t = boost::target(edge, sv_graph);
		
		viewer->addLine(sv_graph[s].supervoxel->centroid_, sv_graph[t].supervoxel->centroid_, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
	}

#if 0
	double min_h = 1e5;
	double max_h = 0.;


	BGL_FORALL_VERTICES(vertex, sv_graph, sv_graph_t)
	{
		PointCloudT::Ptr one_ring_cloud(new PointCloudT);

		pcl::Supervoxel<PointT>::Ptr sv = supervoxel_clusters.at(sv_graph[vertex].supervoxel_label);

		*one_ring_cloud += *sv->voxels_;

		BGL_FORALL_ADJ(vertex, adj_v, sv_graph, sv_graph_t)
		{
			pcl::Supervoxel<PointT>::Ptr adj_sv = supervoxel_clusters.at(sv_graph[adj_v].supervoxel_label);
			
			*one_ring_cloud += *adj_sv->voxels_;			
		}

		if(one_ring_cloud->size() < 10 ) 
		{
			continue;
		}

		pcl::ConvexHull<PointT> chull;
		chull.setDimension(3);
		chull.setComputeAreaVolume(true);

		chull.setInputCloud (one_ring_cloud);	

		std::vector<pcl::Vertices> vertices_chull;
		PointCloudT::Ptr cloud_hull(new PointCloudT);

		chull.reconstruct (*cloud_hull, vertices_chull);

		double H = std::pow(voxel_resolution, 3.0)*one_ring_cloud->size()/chull.getTotalVolume();

		sv_graph[vertex].convexity = H;

		if(H<min_h) min_h = H;
		if(H>max_h) max_h = H;

		cout<<H<<" ";

	}

	cout<<"\n";

	cout<<min_h <<" "<<max_h<<"\n";


	return;


	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
	{
		sv_vertex_t s = boost::source(edge, sv_graph);
		sv_vertex_t t = boost::target(edge, sv_graph);

		sv_graph[edge].weight = std::abs(sv_graph[s].convexity - sv_graph[t].convexity);
	}
#endif

	int num_v = boost::num_vertices(sv_graph);

	Eigen::MatrixXf L = Eigen::MatrixXf::Constant(num_v, num_v, 0.f);

	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t) {

		int s = boost::source(edge, sv_graph);
		int t = boost::target(edge, sv_graph);

		//Eigen::Vector3f vector = (sv_graph[t].supervoxel->centroid_.getVector3fMap() - sv_graph[s].supervoxel->centroid_.getVector3fMap());

		const float weight = exp(-sv_graph[edge].weight);

		L(s, t) = L(t, s) = -1.0f;//-weight;
	}

	cout<<"\n";

	for(int i=0; i<num_v; i++) L(i, i) = -1.0f*L.col(i).sum();

	viewer->addPointCloud(color_cloud, "color_cloud");
	viewer->spin();

	// AX=B initialization
	const float sl = 3.0f;
	const float S0 = seed_resolution;
	float St = S0;

	Eigen::MatrixXf A = Eigen::MatrixXf::Constant( 2*num_v, num_v, 0.f );	

	Eigen::MatrixXf B = Eigen::MatrixXf::Constant( 2*num_v, 3, 0.f );

	Eigen::VectorXf WL_diag_vec = Eigen::VectorXf::Constant( num_v, 0.5f);///(40*S0) );

	for(int i=0; i<num_v; i++) WL_diag_vec(i) *= /*boost::out_degree(i, sv_graph) < 2 ? 0.0f :*/ (float)boost::out_degree(i, sv_graph);
	

	Eigen::VectorXf WH0_diag_vec = Eigen::VectorXf::Constant( num_v, 1.f );

	Eigen::VectorXf WHt_diag_vec = WH0_diag_vec;

	Eigen::VectorXf St_vec = Eigen::VectorXf::Constant( num_v, S0);

	Eigen::MatrixXf P = Eigen::MatrixXf::Constant( num_v, 3, 0.f);

	Eigen::MatrixXf P_new = Eigen::MatrixXf::Constant( num_v, 3, 0.f);

	// supervoxel centroid coordinates to eigen matrix form
	for(int i=0; i<num_v; i++) P.row(i) = sv_graph[i].supervoxel->centroid_.getVector3fMap().transpose();

	for(int iter=0; iter<1; iter++) {

		A.block(num_v, 0, num_v, num_v) = WHt_diag_vec.asDiagonal();

		A.block(0, 0, num_v, num_v) = WL_diag_vec.asDiagonal()*L;

		B.block(num_v, 0, num_v, 3) = WHt_diag_vec.asDiagonal()*P;

		//P_new = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(B);

		P_new = A.fullPivHouseholderQr().solve(B);

		cout<<"compute done\n";

		viewer->removeAllShapes();

/*		for(int i=0; i<num_v; i++) {

			PointAT p ;
		
			p.getVector3fMap() = P_new.row(i).transpose();

			
			viewer->addLine(sv_graph[i].supervoxel->centroid_, p, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
		}
*/
		for(int i=0; i<num_v; i++) {

			sv_graph[i].supervoxel->centroid_.getVector3fMap() = P_new.row(i).transpose();
		}
		

		BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t) {

			sv_vertex_t s = boost::source(edge, sv_graph);
			sv_vertex_t t = boost::target(edge, sv_graph);
		
			viewer->addLine(sv_graph[s].supervoxel->centroid_, sv_graph[t].supervoxel->centroid_, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
		}

		viewer->spin();
		//viewer->removeAllPointClouds();

		// update weight matrices
		WL_diag_vec *= sl;
		
		for(int v=0; v<num_v; v++) {
		
			float min_dist = 1e10f;
	
			BGL_FORALL_ADJ(v, adj, sv_graph, sv_graph_t) {
				
				float dist = (P_new.row(v) - P_new.row(adj)).norm();
	
				if(dist < min_dist) min_dist = dist;
			}

			if(min_dist < 1e10f && min_dist > 1e-5f) St_vec(v) = S0 / min_dist;
		}

		WHt_diag_vec = WH0_diag_vec.cwiseProduct(St_vec);

		P = P_new;
	}

	return;
	


	// ... and its eigenvalues and eigenvectors
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(L);
	Eigen::MatrixXf eigvecs = eig.eigenvectors();
	Eigen::VectorXf eigenvalues = eig.eigenvalues();

	PointCloudT::Ptr spectral_cloud(new PointCloudT);


	int nonzero_eigen_idx = 0;

	for(int i=0; i<num_v; i++) {
	
		if(eigenvalues(i) > 1e-5) {
		
			nonzero_eigen_idx = i;
			break;
		}
	} 

	if ( nonzero_eigen_idx > 0 )
		cout<<eigenvalues(nonzero_eigen_idx-1)<<" ";

	cout<<eigenvalues(nonzero_eigen_idx)<<" "<<eigenvalues(nonzero_eigen_idx+1)<<"\n";

#if 1	
	for(int f=nonzero_eigen_idx; f<num_v; f++)
	{
		Eigen::VectorXf v = eigvecs.col(f);

		std::vector<std::pair<float, int>> map_vector(num_v);

		for(int i=0; i<num_v; i++)
		{
			map_vector[i].first = v(i);
			map_vector[i].second = i;
		}

		std::sort(map_vector.begin(), map_vector.end());

		float min = v.minCoeff();
		float max = v.maxCoeff();

	//	std::cout<<eigvecs<<"\n";

		std::cout<<"eigenvalue "<<eigenvalues(f)<<"\n";

	//	cout<<eigvecs.block(0, 1, num_v, 3)<<"\n";

		spectral_cloud->clear();

		for(int i=0; i<num_v; i++) {
			PointT p;

			pcl::copyPoint(sv_graph[map_vector[i].second].supervoxel->centroid_, p);

			p.rgb = shortRainbowColorMap(map_vector[i].first, min, max);

			spectral_cloud->push_back(p);

			viewer->removeAllPointClouds();
			viewer->addPointCloud(spectral_cloud, "sc");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "sc");
			viewer->spin();
			
		}

		

		break;
	}

	return;
#endif

	for(int i=0; i<num_v; i++) {

		PointT p;
		p.getVector3fMap() = eigvecs.block(i, nonzero_eigen_idx, 1, 3).transpose();

//		p.x /= std::sqrt(eigenvalues(nonzero_eigen_idx));
//		p.y /= std::sqrt(eigenvalues(nonzero_eigen_idx+1));
//		p.z /= std::sqrt(eigenvalues(nonzero_eigen_idx+2));

		p.r = p.g = p.b = 255;
//		p.rgb = 

		spectral_cloud->push_back(p);

		PointT p1;
		pcl::copyPoint(sv_graph[i].supervoxel->centroid_, p1);

		viewer->addLine(p1, p, "c"+to_string(i));

		viewer->spin();		
	}

	viewer->addPointCloud(spectral_cloud, "spectral");

	viewer->spin();

	viewer->removePointCloud("spectral");
	viewer->removeAllShapes();
}


void readBGRD2PointCloud(BGRD* buffer, PointCloudT::Ptr cloud, int color, float min_z = 0.5f, float max_z = 3.0f, bool color_hist_equalization = true)
{
	cloud->clear();

	// BGRD is flipped horizontally
#if 1
	BGRD tmp_bgrd;
	for(int y=0; y<depth_height; y++)
	{
		BGRD* ptr1 = &buffer[y*depth_width]; 
		BGRD* ptr2 = &buffer[(y+1)*depth_width-1];
		for(int x=0; x < (depth_width>>1); ++x, ++ptr1, --ptr2)
		{
			tmp_bgrd = *ptr1;
			*ptr1 = *ptr2;
			*ptr2 = tmp_bgrd;
		}
	}
#endif

	cv::Mat img_hist_equalized; 

	if(color_hist_equalization)
	{
		cv::Mat color;
		color.create(depth_height, depth_width, CV_8UC3);
		int index = 0;
		for(int y=0; y<depth_height; y++)
		{
			for(int x=0; x<depth_width; x++, index++)
			{
				color.at<cv::Vec3b>(y,x)[0] = buffer[index].b;
				color.at<cv::Vec3b>(y,x)[1] = buffer[index].g;
				color.at<cv::Vec3b>(y,x)[2] = buffer[index].r;
			}
		}

		cv::imshow("color"/*+std::to_string(cv::getTickCount())*/, color); cv::waitKey(50);

		std::vector<cv::Mat> channels; 

		cv::split(color,channels); 

		cv::equalizeHist(channels[0], channels[0]); 
		cv::equalizeHist(channels[1], channels[1]); 
		cv::equalizeHist(channels[2], channels[2]);

		cv::merge(channels,img_hist_equalized);
	}

	// filter flying pixels
#if 1
	unsigned short depth_buffer[depth_size];

	for(int i=0; i<depth_size; i++)	depth_buffer[i] = buffer[i].d;

	for (int y = 1; y < depth_height-1; y++) {

		for (int x = 1; x < depth_width-1; x++)	{

			int centerIdx = y*depth_width + x;
		
			int maxJump = 0;

			int centerD = depth_buffer[centerIdx];

			for (int h = -1; h <= 1; h+=1) {

				for (int w = -1; w <= 1; w+=1) {

					if(h == 0 && w == 0) continue;

					int neighborD = std::abs(centerD - depth_buffer[centerIdx + h*depth_width + w]);

					if(neighborD > maxJump) maxJump = neighborD;
				}
			}

			if(maxJump > max_d_jump) buffer[centerIdx].d = 10000;
		}
	}
#endif		

#if 1
	cv::Mat depth_cv;
	cv::Mat depth_u16; 
	depth_cv.create(depth_height, depth_width, CV_8U);
	depth_u16.create(depth_height, depth_width, CV_16U);
	for(int i=0; i<depth_size; i++)	depth_cv.ptr<unsigned char>()[i] = buffer[i].d < 1524 ? buffer[i].d : 0;
	for(int i=0; i<depth_size; i++)	depth_u16.ptr<unsigned short>()[i] = buffer[i].d == 0 ? 10000 : buffer[i].d;
	cv::transpose(depth_cv, depth_cv);
	cv::flip(depth_cv, depth_cv, 1);

	cv::transpose(depth_u16, depth_u16);
	cv::flip(depth_u16, depth_u16, 1);

//	cv::Mat binary;
//	binary = depth_u16 < 1524;
	
//	cv::imshow("binary", binary);

//	cv::imshow("cur_depth_u16", depth_u16); 
	cv::imshow("cur_depth", depth_cv); 
//	cv::imshow("pre_depth", pre_depth_cv);
//	cv::imshow("pre_pre_depth", pre_pre_depth_cv);
	cv::waitKey(100);
//	pre_depth_cv.copyTo(pre_pre_depth_cv);
//	depth_cv.copyTo(pre_depth_cv);
#endif
	
	uint32_t rgb;

	switch(color) {

		case 0:
			rgb = 255<<16 | 0<<8 | 0;
		case 1:
			rgb = 0<<16 | 0<<8 | 255;
		case 2: 
			rgb = 0<<16 | 255<<8 | 0;
		case 3: 
			rgb = 255<<16 | 0<<8 | 255;
	}

	BGRD* ptr = buffer;
	for(int y=0; y<depth_height; y++) {

		for(int x=0; x<depth_width; x++, ptr++) {

			PointT p;

			p.z = ptr->d*0.001f;

			if(p.z > max_z || p.z < min_z) continue;

			if(color_hist_equalization) {

				p.b = img_hist_equalized.at<cv::Vec3b>(y,x)[0];
				p.g = img_hist_equalized.at<cv::Vec3b>(y,x)[1];
				p.r = img_hist_equalized.at<cv::Vec3b>(y,x)[2];
			}
			else {

				p.rgb = *reinterpret_cast<float*>(&rgb);
			}

			p.x = (x - 258.6f) / 366.f*p.z;
			p.y = (y - 206.5f) / 366.f*p.z;
			
			cloud->push_back(p);
		}
	}
}

float projectionCoverScore(PointACloudT::Ptr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
	pcl::MomentOfInertiaEstimation<PointAT> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.setAngleStep(400);
	feature_extractor.compute();

	std::vector <float> moment_of_inertia;
	std::vector <float> eccentricity;
	PointAT min_point_AABB;
	PointAT max_point_AABB;
	PointAT min_point_OBB;
	PointAT max_point_OBB;
	PointAT position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

//	feature_extractor.getMomentOfInertia (moment_of_inertia);
//	feature_extractor.getEccentricity (eccentricity);
//	feature_extractor.getAABB (min_point_AABB, max_point_AABB);
	feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues (major_value, middle_value, minor_value);
	feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter (mass_center);

	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
 	coefficients->values.resize(4);

	float leaf_size = 0.005f;
	pcl::VoxelGrid<PointAT> vox;
	vox.setLeafSize(leaf_size,leaf_size,leaf_size);

	float p;
#if 1
	p = major_vector.dot(mass_center);
	
	coefficients->values[0] = major_vector(0);
	coefficients->values[1] = major_vector(1);
	coefficients->values[2] = major_vector(2);

	coefficients->values[3] = -p;

	
	
	
	PointACloudT::Ptr cloud_down(new PointACloudT);	

	// Create the filtering object
	PointACloudT::Ptr cloud_projected(new PointACloudT);
	pcl::ProjectInliers<PointAT> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	proj.setInputCloud (cloud);
	proj.setModelCoefficients (coefficients);
	proj.filter (*cloud_projected);



	//viewer->addPointCloud(cloud, "o");
//	viewer->addPointCloud(cloud_projected, "p"+to_string(cv::getTickCount()));

	p = middle_vector.dot(mass_center);
	
	coefficients->values[0] = middle_vector(0);
	coefficients->values[1] = middle_vector(1);
	coefficients->values[2] = middle_vector(2);

	coefficients->values[3] = -p;



//	viewer->addPointCloud(cloud_projected, "p"+to_string(cv::getTickCount()));

#endif
	p = minor_vector.dot(mass_center);
	
	coefficients->values[0] = minor_vector(0);
	coefficients->values[1] = minor_vector(1);
	coefficients->values[2] = minor_vector(2);

	coefficients->values[3] = -p;

	proj.setModelCoefficients (coefficients);
	proj.filter (*cloud_projected);

	vox.setInputCloud(cloud_projected);

	vox.filter(*cloud_down);

	viewer->addPointCloud(cloud_projected, "p"+to_string(cv::getTickCount()));


	viewer->spin();

	//viewer->removeAllPointClouds();

	return 0;
}


bool separatePlantsFromSoil(PointCloudT::Ptr cloud, Eigen::Matrix4f & transform, PointCloudT::Ptr plant_cloud, NormalCloudT::Ptr plant_normals, 
			    PointCloudT::Ptr soil_cloud, pcl::ModelCoefficients::Ptr coefficients_plane,
			    std::vector<pcl::ModelCoefficients::Ptr> & stem_lines,
        		    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
	PointCloudT::Ptr cloud_cam(new PointCloudT);
	PointCloudT::Ptr tmp_cloud(new PointCloudT);

	pcl::transformPointCloud(*cloud, *cloud_cam, transform);

	pcl::PassThrough<PointT> pass;
	pass.setFilterFieldName ("x");
	pass.setFilterLimits(-2.0, 0.5);
	//pass.setFilterLimitsNegative (true);
	pass.setInputCloud(cloud_cam);
	pass.filter(*plant_cloud);
	pass.setFilterLimitsNegative(true);
	pass.filter(*soil_cloud);

//	for(auto & p : soil_cloud->points) p.r = p.g = p.b = 255;

	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

	pcl::SACSegmentation<PointT> segp;
	segp.setOptimizeCoefficients(true);
	segp.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	segp.setMethodType(pcl::SAC_RANSAC);
	segp.setMaxIterations(200);
	segp.setDistanceThreshold(0.03);
	segp.setInputCloud(soil_cloud);
	segp.setAxis(Eigen::Vector3f::UnitX());
	segp.setEpsAngle(pcl::deg2rad(15.));

	segp.segment (*inliers_plane, *coefficients_plane);

	if(std::abs(coefficients_plane->values[0]) < 0.85)	//check if vertical
		std::cerr << "Wrong plane coefficients: " << *coefficients_plane << std::endl;

	Eigen::Vector4f plane_eigen;
	plane_eigen << coefficients_plane->values[0], coefficients_plane->values[1], coefficients_plane->values[2], coefficients_plane->values[3];
	pcl::SampleConsensusModelPlane<PointT> scmp(soil_cloud);
	std::vector<int> refined_inliers_plane;
	scmp.selectWithinDistance(plane_eigen, 0.03, refined_inliers_plane);

	inliers_plane->indices = refined_inliers_plane;

	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<NormalT> extract_normals;

	extract.setInputCloud(soil_cloud);
	extract.setIndices(inliers_plane);
	extract.setNegative(false);
	extract.filter(*tmp_cloud);
	extract.setNegative(true);
	extract.filter(*cloud_cam);

	*soil_cloud = *tmp_cloud;

	*plant_cloud += *cloud_cam;

	uint32_t rgb = 100<<16 | 100<<8 | 100;
	for(auto & p : soil_cloud->points)
		p.rgb = *reinterpret_cast<float*>(&rgb);

	if(plant_cloud->size() < 2000) return true;




#if 1
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud (plant_cloud);
	sor.setMeanK (sor_meank);
	sor.setStddevMulThresh (sor_std);
	sor.filter (*tmp_cloud);


	viewer->removeAllShapes(); skeleton(tmp_cloud, plane_eigen, viewer); return true;

//	*tmp_cloud = *plant_cloud;

	// Create a KD-Tree
	pcl::search::KdTree<PointT>::Ptr tree_mls (new pcl::search::KdTree<PointT>);

	// Output has the PointNormal type in order to store the normals calculated by MLS
	pcl::PointCloud<PointT> mls_points;

	pcl::console::TicToc tt;
	tt.tic();
	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<PointT, PointT> mls;

	mls.setComputeNormals (false);

	// Set parameters
	mls.setInputCloud (tmp_cloud);
	mls.setPolynomialFit (false);
	mls.setSearchMethod (tree_mls);
	mls.setSearchRadius (normal_radius);

	// Reconstruct
	mls.process(mls_points);

	tt.toc_print();

	cout<<"mls input size "<<tmp_cloud->size()<<"\n";

	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(mls_points, indices);

	cout<<"nan size: "<<mls_points.size()-indices.size()<<"\n";

	pcl::copyPointCloud(mls_points, *tmp_cloud);

	pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT>> (new pcl::search::KdTree<PointT>);
	NormalCloudT::Ptr normals(new NormalCloudT);
	pcl::NormalEstimationOMP<PointT, NormalT> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(tmp_cloud);
	normal_estimator.setRadiusSearch(normal_radius);
	//normal_estimator.setKSearch(15);
	normal_estimator.compute(*normals);

	
	pcl::RegionGrowing<PointT, NormalT> reg;
	//reg.setMinClusterSize(3);
	//reg.setMaxClusterSize(1);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(rr_num_neighbor);
	reg.setInputCloud(tmp_cloud);
	//reg.setIndices(indices);
	reg.setInputNormals(normals);
	reg.setSmoothnessThreshold(pcl::deg2rad(rr_angle));
	reg.setCurvatureThreshold(rr_curvature); 
	reg.setCurvatureTestFlag(true);
	reg.setSmoothModeFlag(true);
	reg.setResidualTestFlag(true);
	reg.setResidualThreshold(rr_residual);

	std::vector<pcl::PointIndices> clusters;

	reg.extract(clusters);

	std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;

	PointCloudT::Ptr colored_cloud = reg.getColoredCloud ();

	viewer->addPointCloud(colored_cloud,"color", 0);
	//viewer->spin();
	//viewer->removeAllPointClouds();

	PointCloudT::Ptr init_stem_cloud(new PointCloudT);

	std::vector<int> canopy_indices;

	for(auto & indices : clusters)
	{
		PointCloudT::Ptr rr_segment_cloud(new PointCloudT);

		pcl::copyPointCloud(*tmp_cloud, indices, *rr_segment_cloud);

		//*init_stem_cloud += *rr_segment_cloud;
		

#if 1
/*		pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
		super.setInputCloud(rr_segment_cloud);
		super.setColorImportance (color_importance);
		super.setSpatialImportance (spatial_importance);
		super.setNormalImportance (normal_importance);

		std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

		super.extract(supervoxel_clusters);

		//pcl::console::print_info("Found %d supervoxels\n", supervoxel_clusters.size ());

		if(supervoxel_clusters.size() < 10) continue;

		spectralCluster(rr_segment_cloud, viewer);

		Graph supervoxel_adjacency_list;

		super.getSupervoxelAdjacencyList(supervoxel_adjacency_list);
*/
		//PointLCloudT::Ptr color_cloud = super.getLabeledCloud(); 

		pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
		feature_extractor.setInputCloud(rr_segment_cloud);
		feature_extractor.setAngleStep(400);	// >360, only compute around major axis once
		feature_extractor.compute();

		std::vector <float> moment_of_inertia;
		std::vector <float> eccentricity;
		PointT min_point_AABB;
		PointT max_point_AABB;
		PointT min_point_OBB;
		PointT max_point_OBB;
		PointT position_OBB;
		Eigen::Matrix3f rotational_matrix_OBB;
		float major_value, middle_value, minor_value;
		Eigen::Vector3f major_vector, middle_vector, minor_vector;
		Eigen::Vector3f mass_center;

		feature_extractor.getMomentOfInertia (moment_of_inertia);
		feature_extractor.getEccentricity (eccentricity);
		feature_extractor.getAABB (min_point_AABB, max_point_AABB);
		feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
		feature_extractor.getEigenValues (major_value, middle_value, minor_value);
		feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
		feature_extractor.getMassCenter (mass_center);

	//	if(major_value/middle_value < 4.f)  continue;

		if( rr_segment_cloud->size() > 300 )
		{
			double surface_variation = minor_value/(middle_value+minor_value+major_value);
			
			if( std::abs(major_vector(0)) < 0.95f || eccentricity[0] < 0.95 || surface_variation > 0.007)
			{
				
				for(auto & idx : indices.indices)
				{
					canopy_indices.push_back(idx);
				}
				//continue;


				Eigen::Vector3f position (position_OBB.x, position_OBB.y, position_OBB.z);
				Eigen::Quaternionf quat (rotational_matrix_OBB);
				viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, 
						"OBB"+to_string(cv::getTickCount()));
				viewer->setRepresentationToWireframeForAllActors();
	
			}

			//std::cout<<<<"\n";
			
			//viewer->spin();
		}


#if 0
		BGL_FORALL_EDGES(edge, supervoxel_adjacency_list, Graph)
		{
			Voxel s = boost::source(edge, supervoxel_adjacency_list);
			Voxel t = boost::target(edge, supervoxel_adjacency_list);

			uint32_t s_l = supervoxel_adjacency_list[s];
			uint32_t t_l = supervoxel_adjacency_list[t];

			pcl::Supervoxel<PointT>::Ptr svs = supervoxel_clusters.at(s_l);
			pcl::Supervoxel<PointT>::Ptr svt = supervoxel_clusters.at(t_l);


	//		PointACloudT::Ptr s_cloud = svs->voxels_;
	//		PointACloudT::Ptr t_cloud = svt->voxels_;


			pcl::PointXYZ p0, p1;
			p0.x = svs->centroid_.x;
			p0.y = svs->centroid_.y;
			p0.z = svs->centroid_.z;

			p1.x = svt->centroid_.x;
			p1.y = svt->centroid_.y;
			p1.z = svt->centroid_.z;

			viewer->addLine(p0, p1, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
		}
#endif

		

	//	viewer->addPointCloud(color_cloud, "color"+to_string(cv::getTickCount()), 0);

		
		//major_vector *= 0.1;

		//pcl::PointXYZ center (mass_center (0), mass_center (1), mass_center (2));
		//pcl::PointXYZ x_axis (major_vector (0) + mass_center (0), major_vector (1) + mass_center (1), major_vector (2) + mass_center (2));
		//pcl::PointXYZ y_axis (middle_vector (0) + mass_center (0), middle_vector (1) + mass_center (1), middle_vector (2) + mass_center (2));
		//pcl::PointXYZ z_axis (minor_vector (0) + mass_center (0), minor_vector (1) + mass_center (1), minor_vector (2) + mass_center (2));

		//viewer->spin();

		
	/*	if(x_axis.x >= center.x)	
		 viewer->addLine (center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector"+to_string(cv::getTickCount()));
		else
		 viewer->addLine (x_axis, center, 1.0f, 0.0f, 0.0f, "major eigen vector"+to_string(cv::getTickCount()));
	*/	//viewer->addLine (center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
		//viewer->addLine (center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");	

#endif	
	}

	//viewer->spin();
	//viewer->removeAllPointClouds();
	//viewer->removeAllShapes();

	pcl::PointIndices::Ptr inliers_canopy(new pcl::PointIndices);

	inliers_canopy->indices = canopy_indices;

	extract.setInputCloud(tmp_cloud);
	extract.setIndices(inliers_canopy);
	extract.setNegative(true);
	extract.filter(*init_stem_cloud);

	std::vector<Eigen::Vector3f> a_vec, b_vec;

	Hough3DLine(init_stem_cloud, viewer, a_vec, b_vec, h_dx, h_nlines, h_minvotes, h_granularity);

	std::cout<<"num line detected "<<a_vec.size()<<"\n";

	float line_len = 1.2f;

	for(int i=0; i<a_vec.size(); i++)
	{
		bool valid_line = true;

		//check root position and line direction
		if( a_vec[i](2) > 1.3f || a_vec[i](2) < 0.7f || b_vec[i](0) > -0.95) valid_line = false;
		
		for (int j=0; valid_line && j<i; j++)		
		{
			std::pair<Eigen::Vector3f, Eigen::Vector3f> segment1(a_vec[i], a_vec[i] + line_len*b_vec[i]);
			std::pair<Eigen::Vector3f, Eigen::Vector3f> segment2(a_vec[j], a_vec[j] + line_len*b_vec[j]);

			float l2l_dist = dist3D_Segment_to_Segment(segment1, segment2);

			if(l2l_dist < 0.1f)
			{
				valid_line = false;
			}
		}

		pcl::PointXYZ p1(a_vec[i](0), a_vec[i](1), a_vec[i](2));

		Eigen::Vector3f vector = a_vec[i] + line_len*b_vec[i];

		pcl::PointXYZ p2(vector(0), vector(1), vector(2));

		if(!valid_line)
		{
			viewer->addLine(p1, p2, 0, 1, 0, "line"+to_string(cv::getTickCount()));	

			continue;
		}		
		else 
			viewer->addLine(p1, p2, 1, 0, 0, "line"+to_string(cv::getTickCount()));	

		Eigen::VectorXf line_eigen(6);
		line_eigen(0) = a_vec[i](0); line_eigen(1) = a_vec[i](1); line_eigen(2) = a_vec[i](2);
		line_eigen(3) = b_vec[i](0); line_eigen(4) = b_vec[i](1); line_eigen(5) = b_vec[i](2);

		pcl::SampleConsensusModelLine<PointT> scml(colored_cloud);
		std::vector<int> inliers_line;
		scml.selectWithinDistance(line_eigen, 0.15, inliers_line);

		for(auto & idx : inliers_line)
		{
			colored_cloud->points[idx].r = 255;
			colored_cloud->points[idx].g = 255; 
			colored_cloud->points[idx].b = 255;  
		}	

		viewer->spin();

		continue;
		


		PointCloudT::Ptr super_cloud_in(new PointCloudT);

		pcl::copyPointCloud(*colored_cloud, inliers_line, *super_cloud_in);

		cout<<"super_cloud_in size "<<super_cloud_in->size()<<"\n";


		viewer->removeAllShapes(); skeleton(super_cloud_in, plane_eigen, viewer); continue;

	/*	ofstream myfile;
		myfile.open ("cloud.off");
		myfile << "OFF\r\n"<<to_string(super_cloud_in->size())<<" 0 0"<<"\r\n";
		for(auto & p : super_cloud_in->points)
			myfile << to_string(p.x)<<" "<<to_string(p.y)<<" "<<to_string(p.y)<<"\r\n";
		myfile.close();

		pcl::io::savePLYFile("cloud.ply", *super_cloud_in);
		*/

		Eigen::Vector3f stem_line_dir = b_vec[i];

		Graph supervoxel_adjacency_list;

		// my supervoxel graph
		sv_graph_t sv_graph, sv_graph_origional;

		std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

		//viewer->removeAllShapes();

		for(int iter=0; iter<skeleton_iteration; iter++)
		{
			pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
			super.setInputCloud(super_cloud_in);
			super.setColorImportance (color_importance);
			super.setSpatialImportance (spatial_importance);
			super.setNormalImportance (normal_importance);

			super.extract(supervoxel_clusters);

			cout<<"supervoxel_clusters size "<<supervoxel_clusters.size()<<"\n";

			if(supervoxel_clusters.size() == 0) return true;

			PointLCloudT::Ptr color_cloud = super.getLabeledCloud();

			super.getSupervoxelAdjacencyList(supervoxel_adjacency_list);

			sv_graph.clear();

			std::map<uint32_t, sv_vertex_t> label_ID_map;

			BGL_FORALL_VERTICES(vertex, supervoxel_adjacency_list, Graph)
			{
				sv_vertex_t v = boost::add_vertex(sv_graph);

				sv_graph[v].supervoxel_label = supervoxel_adjacency_list[vertex];

				sv_graph[v].supervoxel = supervoxel_clusters.at(sv_graph[v].supervoxel_label);

				label_ID_map.insert(std::make_pair(supervoxel_adjacency_list[vertex], v));
			}

			BGL_FORALL_EDGES(edge, supervoxel_adjacency_list, Graph)
			{
				Voxel s = boost::source(edge, supervoxel_adjacency_list);
				Voxel t = boost::target(edge, supervoxel_adjacency_list);

				sv_vertex_t sv_s = (label_ID_map.find(supervoxel_adjacency_list[s]))->second;
				sv_vertex_t sv_t = (label_ID_map.find(supervoxel_adjacency_list[t]))->second;

				boost::add_edge(sv_s, sv_t, sv_graph);
			}

			if(iter == 0) sv_graph_origional = sv_graph;

#if 0
			std::vector<int> component(boost::num_vertices(sv_graph));
			int num_cc = boost::connected_components(sv_graph, &component[0]);

			int count = 0;

			std::vector<int> v_2_remove;
			for(int i=0; i<component.size(); i++) {
				if(component[i] != 0) {

					v_2_remove.push_back(i);

				}
			}

			for(auto& v:v_2_remove) {
				boost::clear_vertex(v, sv_graph);
				boost::remove_vertex(v, sv_graph);
			}
#endif

#if 0

			Eigen::MatrixXd eigenfunctions;
			Eigen::VectorXd eigenvalues;
			int first_non_zero_idx;

			PB_MHB(sv_graph, seed_resolution, eigenfunctions, eigenvalues, first_non_zero_idx);

			cout<<"PB MHB done\n";
#endif

#if 0
			Eigen::VectorXd w = eigenfunctions.col(first_non_zero_idx);

			float min_w = w.minCoeff();
			float max_w = w.maxCoeff();

			viewer->removeAllPointClouds();

			PointCloudT::Ptr gps(new PointCloudT);

			for(int i=0; i<eigenvalues.size(); i++)	{
				PointT p;

				Eigen::Vector3d point = eigenfunctions.block(i, first_non_zero_idx, 1, 3);

				point(0) /= std::sqrt(eigenvalues(first_non_zero_idx)); 
				point(1) /= std::sqrt(eigenvalues(first_non_zero_idx+1)); 
				point(2) /= std::sqrt(eigenvalues(first_non_zero_idx+2)); 

				point *= 0.01;

				point(2) = 0.0;

				p.getVector3fMap() = point.cast<float>();

				PointT p1;
				pcl::copyPoint(sv_graph[i].supervoxel->centroid_, p1);

				viewer->addLine(p1, p, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);

				p.rgb = shortRainbowColorMap(w(i), min_w, max_w);

				gps->push_back(p);
			}

			viewer->addPointCloud(gps, "gps");

			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "gps");
#endif
		
#if 0
			for (int f = first_non_zero_idx; f <= first_non_zero_idx /*eigenvalues.size()*/; f++) {

				Eigen::VectorXd v = eigenfunctions.col(f);

				//cout<<v<<"\n\n";
				cout<<f<<" eigenvalue "<<eigenvalues(f)<<"\n";

				// Search for the minimum and maximum curvature
				float min = v.minCoeff();
				float max = v.maxCoeff();

				cout<<"min "<<min<<" max "<<max<<"\n";

				std::vector<std::pair<double, int> > map_vector(v.size());

				for (auto i = 0; i < v.size(); i++) {
					map_vector[i].first = v(i);
					map_vector[i].second = i;
				}

				std::sort(map_vector.begin(), map_vector.end());

				BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
				{
					sv_vertex_t s = boost::source(edge, sv_graph);
					sv_vertex_t t = boost::target(edge, sv_graph);

					viewer->addLine(sv_graph[s].supervoxel->centroid_, sv_graph[t].supervoxel->centroid_, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
				}


				tmp_cloud->clear();

#if 1
				for(auto & pair:map_vector) {
					
					PointT p;

					pcl::copyPoint(sv_graph[pair.second].supervoxel->centroid_, p);

					p.rgb = shortRainbowColorMap(pair.first, min, max);

					tmp_cloud->push_back(p);

					viewer->removeAllPointClouds();

					viewer->addPointCloud(tmp_cloud, "sv");
				
					viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "sv");

					viewer->spin();

				}
#endif

/*				for (int i = 0; i < boost::num_vertices(sv_graph); i++) {
					
					PointT p;

					pcl::copyPoint(sv_graph[i].supervoxel->centroid_, p);
		
					p.rgb = shortRainbowColorMap(v(i), min, max);

					tmp_cloud->push_back(p);
				}
*/


				viewer->removeAllPointClouds();

				viewer->addPointCloud(tmp_cloud, "Scene");

				viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8.0, "Scene");

				viewer->spin();
	
			}
#endif

			super_cloud_in->clear();

			float total_motion = 0.0f;

			//viewer->addPointCloud(color_cloud, "color_cloud");
			//viewer->spin();
			
#if 0
			//for(int i=0; i<num_v; i++) sv_graph[i].supervoxel->centroid_.getVector3fMap() = P_new.row(i).transpose();
						
			
	
			BGL_FORALL_VERTICES(vertex, sv_graph, sv_graph_t)
			{
				pcl::Supervoxel<PointT>::Ptr supervoxel = sv_graph[vertex].supervoxel;

				Eigen::Vector3f centroid = supervoxel->centroid_.getVector3fMap();

				int num_neighbors = boost::out_degree(vertex, sv_graph);

				bool triangle = false;

				if(num_neighbors == 0)
					continue;				
				else if(num_neighbors == 1) {

					*super_cloud_in += *supervoxel->voxels_;
					continue;
				}
				else if(num_neighbors == 2) {

					std::vector<sv_vertex_t> two_adjs;

					BGL_FORALL_ADJ(vertex, adj_v, sv_graph, sv_graph_t) two_adjs.push_back(adj_v);

					BGL_FORALL_ADJ(two_adjs[0], adj_adj_v, sv_graph, sv_graph_t)
					{
						if(two_adjs[1] == adj_adj_v)
						{
							triangle = true;
							break;
						}
					}
																
					pcl::Supervoxel<PointT>::Ptr neighbor_1_supervoxel = sv_graph[two_adjs[0]].supervoxel;

					pcl::Supervoxel<PointT>::Ptr neighbor_2_supervoxel = sv_graph[two_adjs[1]].supervoxel;

					Eigen::Vector3f dir_1 = (neighbor_1_supervoxel->centroid_.getVector3fMap() - centroid).normalized();

					Eigen::Vector3f dir_2 = (neighbor_2_supervoxel->centroid_.getVector3fMap() - centroid).normalized();						
					

					if( !triangle 
					    && dir_1.dot(dir_2) < -0.95f
					  ) {
						*super_cloud_in += *supervoxel->voxels_;
						continue;
					}
				}


				PointACloudT::Ptr neighbor_centroids(new PointACloudT);

				neighbor_centroids->push_back(supervoxel->centroid_);

				Eigen::Vector3f neighbor_mean(0,0,0);

				float weight_sum = 0.0f;

				BGL_FORALL_ADJ(vertex, adj_v, sv_graph, sv_graph_t) {

					pcl::Supervoxel<PointT>::Ptr neighbor_supervoxel = sv_graph[adj_v].supervoxel;

					neighbor_centroids->push_back(neighbor_supervoxel->centroid_);

					Eigen::Vector3f neighbor_centroid = neighbor_supervoxel->centroid_.getVector3fMap();

					// more weight for neighbors above or below the point
					float weight = std::abs( stem_line_dir.dot( (neighbor_centroid - centroid).normalized() ) );

					neighbor_mean += neighbor_centroid*weight;
					//neighbor_mean += neighbor_centroid;

					weight_sum += weight;

				}

				neighbor_mean += centroid; weight_sum += 1.f;

				neighbor_mean /= weight_sum;

				pcl::PCA<PointAT> pca;

				pca.setInputCloud(neighbor_centroids);

				Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

				Eigen::Vector3f eigen_values = pca.getEigenValues();

				//float sigma = eigen_values(0)/(eigen_values.sum());

				//float sigma = (eigen_values(0)-eigen_values(1))/(eigen_values(0));

				Eigen::Vector3f moving_dir = (neighbor_mean - centroid);

				if( moving_dir.dot( eigen_vectors.col(1) ) < 0.f) eigen_vectors.col(1) *= -1.f;

				// prevent moving towards plant growing direction
				moving_dir *= 1.0f - std::max(0.f, stem_line_dir.dot(eigen_vectors.col(1)));

				Eigen::Vector3f delta = eigen_vectors.col(1)*(moving_dir.dot(eigen_vectors.col(1)));

				//total_motion += delta.norm()*supervoxel->voxels_->size();

				for(auto & p : supervoxel->voxels_->points) p.getVector3fMap() += delta;
					
				*super_cloud_in += *supervoxel->voxels_;
			}

#endif

			BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
			{
				sv_vertex_t s = boost::source(edge, sv_graph);
				sv_vertex_t t = boost::target(edge, sv_graph);

				viewer->addLine(sv_graph[s].supervoxel->centroid_, sv_graph[t].supervoxel->centroid_, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
			}

			viewer->addLine(p1, p2, 1, 0, 0, "line"+to_string(cv::getTickCount()));	
			viewer->addPointCloud(color_cloud, "super", 0);
			viewer->spin();

			if(iter < skeleton_iteration-1)
			{
				viewer->removeAllPointClouds();
				viewer->removeAllShapes();
			}

			//if(total_motion < 0.01f) break;
		}

#if 1

		std::vector<int> component(boost::num_vertices(sv_graph));

		int num_cc = boost::connected_components(sv_graph, &component[0]);

		pcl::PointCloud<pcl::PointXYZ>::Ptr centroid_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		std::vector<sv_vertex_t> one_neighbor_vertices;

		BGL_FORALL_VERTICES(vertex, sv_graph, sv_graph_t)
		{
			pcl::Supervoxel<PointT>::Ptr sv = supervoxel_clusters.at(sv_graph[vertex].supervoxel_label);
	
			pcl::PointXYZ p;
	
			pcl::copyPoint(sv->centroid_, p);

			centroid_cloud->push_back(p);

			if(boost::out_degree(vertex, sv_graph) == 1) one_neighbor_vertices.push_back(vertex);
		}

		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

		kdtree.setInputCloud (centroid_cloud);

		for(auto vertex : one_neighbor_vertices)
		{
			pcl::Supervoxel<PointT>::Ptr sv = supervoxel_clusters.at(sv_graph[vertex].supervoxel_label);

			pcl::PointXYZ search_point; pcl::copyPoint(sv->centroid_, search_point);

			Eigen::Vector3f point_to_connect = search_point.getVector3fMap();

			std::vector<int> pointIdxRadiusSearch;

			std::vector<float> pointRadiusSquaredDistance;


			if(kdtree.radiusSearch(search_point, 3.f*seed_resolution, pointIdxRadiusSearch, pointRadiusSquaredDistance) == 0) continue;


			int idx = -1;
			float min_dist = 1e7f;				
		

			for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			{
				if(component[vertex] == component[pointIdxRadiusSearch[i]]) continue;

				sv_vertex_t adj_vertex = *(boost::adjacent_vertices(vertex, sv_graph).first);

				pcl::Supervoxel<PointT>::Ptr adj_sv = supervoxel_clusters.at(sv_graph[adj_vertex].supervoxel_label);

				Eigen::Vector3f adjacent_point = adj_sv->centroid_.getVector3fMap();

				Eigen::Vector3f test_point = centroid_cloud->points[pointIdxRadiusSearch[i]].getVector3fMap();

				Eigen::Vector3f ray = (point_to_connect - adjacent_point).normalized();

				Eigen::Vector3f connect_edge = (test_point - point_to_connect);

				if( ray.dot(connect_edge.normalized()) < 0.8f
				    || connect_edge(0) < 0.f
				   ) continue;

				float tmp_dist = (connect_edge - connect_edge.dot(ray)*ray).norm();
					
				if(tmp_dist < min_dist)
				{
					min_dist = tmp_dist;

					idx = i;
				}					
			}

			if(idx != -1)
			{
				PointCloudT::Ptr connection_cloud(new PointCloudT);

				*connection_cloud += *sv->voxels_;
		
				Eigen::Vector3f delta = 0.5f*(centroid_cloud->points[pointIdxRadiusSearch[idx]].getVector3fMap() - point_to_connect);

				for(auto & p : connection_cloud->points) 
				{
					p.getVector3fMap() += delta;
				}

				*super_cloud_in += *connection_cloud;

				boost::add_edge(vertex, pointIdxRadiusSearch[idx], sv_graph);
		
				viewer->addLine(search_point, centroid_cloud->points[pointIdxRadiusSearch[idx]], 1, 1, 0, "line"+std::to_string(cv::getTickCount()), 0);
			}
		}
#endif		


	/*	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
		{
			sv_vertex_t s = boost::source(edge, sv_graph);
			sv_vertex_t t = boost::target(edge, sv_graph);

			pcl::Supervoxel<PointT>::Ptr svs = supervoxel_clusters.at(sv_graph[s].supervoxel_label);
			pcl::Supervoxel<PointT>::Ptr svt = supervoxel_clusters.at(sv_graph[t].supervoxel_label);

			viewer->addLine(svs->centroid_, svt->centroid_, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
		}
	*/

	//	viewer->addLine(p1, p2, 1, 0, 0, "line"+to_string(cv::getTickCount()));	

		BGL_FORALL_EDGES(edge, sv_graph_origional, sv_graph_t)
		{
			sv_vertex_t s = boost::source(edge, sv_graph_origional);
			sv_vertex_t t = boost::target(edge, sv_graph_origional);

			viewer->addLine(sv_graph_origional[s].supervoxel->centroid_, sv_graph_origional[t].supervoxel->centroid_, 0, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
		}

		viewer->addPointCloud(colored_cloud, "superv", 0);
		viewer->spin();
		viewer->removeAllPointClouds();
		viewer->removeAllShapes();

		int min_branch_segments = 4;
	}

	

//	viewer->removeAllPointClouds();
//	viewer->addPointCloud(colored_cloud, "stem");
//	viewer->addPointCloud(soil_cloud, "soil");
//	viewer->addPointCloud(plant_cloud, "plant");
//	viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "plant");
	viewer->spin();
	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

	return true;

	segp.setModelType(pcl::SACMODEL_PARALLEL_LINE);
	segp.setMethodType(pcl::SAC_RANSAC);
	segp.setMaxIterations(1000);
	segp.setDistanceThreshold(0.03);
	segp.setInputCloud(soil_cloud);
	segp.setAxis(Eigen::Vector3f::UnitX());
	segp.setEpsAngle(pcl::deg2rad(15.));

	PointCloudT::Ptr plant_cloud_copy(new PointCloudT);

	*plant_cloud_copy = *plant_cloud;

	plant_cloud->clear();


	for(int i=0; i<4; i++)
	{
		pcl::ModelCoefficients::Ptr coefficients_line (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_line (new pcl::PointIndices);

		segp.setInputCloud(plant_cloud_copy);
		segp.segment(*inliers_line, *coefficients_line);

		std::cout<<*coefficients_line<<"\n";

		if(coefficients_line->values.size() != 6	//no solution
		  || inliers_line->indices.size() < 200	//inliers too few
		  || std::abs(coefficients_line->values[3]) < 0.99	// not vertical 
		  ) continue;


		if(coefficients_line->values[3] > 0.)
		{
			coefficients_line->values[3] *= -1.;
			coefficients_line->values[4] *= -1.;
			coefficients_line->values[5] *= -1.;
		}

		double t = (0.65 - coefficients_line->values[0])/coefficients_line->values[3];

		coefficients_line->values[0] = 0.65;
		coefficients_line->values[1] += t*coefficients_line->values[4];
		coefficients_line->values[2] += t*coefficients_line->values[5];

		// cylinder near image edge, ignore
		//if(std::abs(coefficients_line->values[1]) > 0.4 ) continue;

		stem_lines.push_back(coefficients_line);

		PointCloudT::Ptr inlier_cloud(new PointCloudT);

#if 0
		Eigen::VectorXf line_eigen(6);
		for(int j=0; j<6; j++) line_eigen(j) = coefficients_line->values[j];
		pcl::SampleConsensusModelLine<PointT> scml(plant_cloud_copy);
		std::vector<int> refined_inliers_line;
		scml.selectWithinDistance(line_eigen, 0.03, refined_inliers_line);

		inliers_line->indices = refined_inliers_line;

		extract.setInputCloud(plant_cloud_copy);
		extract.setIndices(inliers_line);
		extract.setNegative(false);
		extract.filter(*inlier_cloud);
		extract.setNegative(true);
		extract.filter(*tmp_cloud);

		for(auto & p: inlier_cloud->points)
		{
			p.r = p.g = p.b = 180;
		}
#else
		Eigen::Vector3f unit_line_dir;

		unit_line_dir << coefficients_line->values[3], coefficients_line->values[4], coefficients_line->values[5];

		tmp_cloud->clear();

		for(int j=0; j<plant_cloud_copy->points.size(); j++)
		{
			PointT & p = plant_cloud_copy->points[j];

			Eigen::Vector3f vec;
			vec << coefficients_line->values[0] - p.x, 
			       coefficients_line->values[1] - p.y, 
			       coefficients_line->values[2] - p.z;

			float point_to_line_dist = (vec - vec.dot(unit_line_dir)*unit_line_dir).norm();

			if(point_to_line_dist > 0.03)
			{
				tmp_cloud->push_back(p);	
			}
			else
			{
				p.r = p.g = p.b = 180;
				inlier_cloud->push_back(p);
			}
		}
#endif

		*plant_cloud += *inlier_cloud;
		*plant_cloud_copy = *tmp_cloud;
		std::cout<<i<<"\n";

#if 1
		viewer->addPointCloud(inlier_cloud,"inlier_cloud"+std::to_string(i), 0);
		viewer->addPointCloud(tmp_cloud,"tmp_cloud"+std::to_string(i), 0);
		viewer->addLine(*coefficients_line, "line"+std::to_string(i), 0);
		//viewer->spinOnce(pcl_view_time);
#endif
	}

	*plant_cloud += *plant_cloud_copy;


	viewer->addPointCloud(plant_cloud,"plant_cloud",0);
	viewer->spinOnce(pcl_view_time);
	viewer->removeAllPointClouds();	viewer->removeAllShapes();

	return true;

#endif




#if 0
	NormalCloudT::Ptr normals(new NormalCloudT);
	pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);
	pcl::NormalEstimation<PointT, NormalT> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(tmp_cloud);
	normal_estimator.setRadiusSearch (normal_radius);
	//normal_estimator.setKSearch(50);
  	normal_estimator.compute (*normals);
#endif

	PointACloudT::Ptr super_cloud_in(new PointACloudT);
	
/*	for(auto & p : plant_cloud->points)
	{
		PointAT p_;
		p_.x = p.x; p_.y = p.y; p_.z = p.z;
		p_.r = p.r; p_.g = p.g; p_.b = p.b;
		p_.a = 255;
		super_cloud_in->push_back(p_);
	}

*/
	pcl::copyPointCloud(*plant_cloud, *super_cloud_in);	

	viewer->addPointCloud(super_cloud_in, std::to_string(cv::getTickCount()), 0);
	viewer->spin();
	viewer->removeAllPointClouds();

#if 0

	pcl::SupervoxelClustering<PointAT> super(voxel_resolution, seed_resolution);
	super.setInputCloud(super_cloud_in);
	super.setColorImportance (color_importance);
	super.setSpatialImportance (spatial_importance);
	super.setNormalImportance (normal_importance);

	std::map<uint32_t, pcl::Supervoxel<PointAT>::Ptr> supervoxel_clusters;

	super.extract(supervoxel_clusters);

	pcl::console::print_info("Found %d supervoxels\n", supervoxel_clusters.size ());

	if(supervoxel_clusters.size() == 0) return true;


	Graph supervoxel_adjacency_list;

	super.getSupervoxelAdjacencyList(supervoxel_adjacency_list);

	PointLCloudT::Ptr color_cloud = super.getLabeledCloud();

	// my supervoxel graph
	sv_graph_t sv_graph;

	std::map<uint32_t, sv_vertex_t> label_ID_map;

	BGL_FORALL_VERTICES(vertex, supervoxel_adjacency_list, Graph)
	{
		sv_vertex_t v = boost::add_vertex(sv_graph);

		sv_graph[v].supervoxel_label = supervoxel_adjacency_list[vertex];

		label_ID_map.insert(std::make_pair(supervoxel_adjacency_list[vertex], v));
	}

	BGL_FORALL_EDGES(edge, supervoxel_adjacency_list, Graph)
	{
		Voxel s = boost::source(edge, supervoxel_adjacency_list);
		Voxel t = boost::target(edge, supervoxel_adjacency_list);

		uint32_t s_l = supervoxel_adjacency_list[s];
		uint32_t t_l = supervoxel_adjacency_list[t];

		sv_vertex_t sv_s = (label_ID_map.find(s_l))->second;
		sv_vertex_t sv_t = (label_ID_map.find(t_l))->second;

		sv_edge_t sv_edge;
		bool edge_added;

		boost::tie(sv_edge, edge_added) = boost::add_edge(sv_s, sv_t, sv_graph);

		if(edge_added)
		{
			pcl::Supervoxel<PointAT>::Ptr svs = supervoxel_clusters.at(s_l);
			pcl::Supervoxel<PointAT>::Ptr svt = supervoxel_clusters.at(t_l);

			Eigen::Vector3f dir;

			dir << svs->centroid_.x - svt->centroid_.x,
			       svs->centroid_.y - svt->centroid_.y,
			       svs->centroid_.z - svt->centroid_.z;

			dir.normalize();

			sv_graph[sv_edge].weight = supervoxel_adjacency_list[edge];//*( 1.0f- std::abs(dir(0)) );
		}

	}


	std::vector<sv_edge_t> spanning_tree;

	boost::kruskal_minimum_spanning_tree(sv_graph, std::back_inserter(spanning_tree), boost::weight_map(boost::get(&SVEdgeProperty::weight, sv_graph)) );

	sv_graph_t mst;

	for(auto & e : spanning_tree)
	{
		sv_vertex_t s = boost::source(e, sv_graph);

		sv_vertex_t t = boost::target(e, sv_graph);

		sv_edge_t new_e;

		bool edge_added;		

		boost::tie(new_e, edge_added) = boost::add_edge(s, t, mst);

		mst[s].supervoxel_label = sv_graph[s].supervoxel_label;

		mst[t].supervoxel_label = sv_graph[t].supervoxel_label;

		mst[new_e].weight = sv_graph[e].weight;
	}


	std::cout<<"mst num vertices: "<<boost::num_vertices(mst)<<"\n";

	std::cout<<"mst num edges: "<<boost::num_edges(mst)<<" "<<boost::num_edges(sv_graph)<<"\n";

	BGL_FORALL_EDGES(edge, mst, sv_graph_t)
	{
		sv_vertex_t s = boost::source(edge, mst);

		sv_vertex_t t = boost::target(edge, mst);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_s = supervoxel_clusters.at(mst[s].supervoxel_label);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_t = supervoxel_clusters.at(mst[t].supervoxel_label);

		pcl::PointXYZ p0, p1;
		p0.x = supervoxel_s->centroid_.x;
		p0.y = supervoxel_s->centroid_.y;
		p0.z = supervoxel_s->centroid_.z;

		p1.x = supervoxel_t->centroid_.x;
		p1.y = supervoxel_t->centroid_.y;
		p1.z = supervoxel_t->centroid_.z;

		viewer->addLine(p0, p1, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
	}

	viewer->addPointCloud(color_cloud, "color_cloud", 0);

	viewer->spin();
	viewer->removeAllShapes();


	for(int i=0; i<10; i++)
	{
		std::vector<std::pair<sv_vertex_t, sv_vertex_t>> edges_to_remove;
	
		BGL_FORALL_VERTICES(vertex, mst, sv_graph_t)
		{
			int num_neighbors = boost::out_degree(vertex, mst);
		
			if(num_neighbors == 4)
			{
				pcl::Supervoxel<PointAT>::Ptr supervoxel = supervoxel_clusters.at(mst[vertex].supervoxel_label);

				sv_vertex_t vertex_to_disconnect;

				float min_abs_x = 1e5f;

				BGL_FORALL_ADJ(vertex, adj_v, mst, sv_graph_t)
				{
					pcl::Supervoxel<PointAT>::Ptr neighbor_supervoxel = supervoxel_clusters.at(mst[adj_v].supervoxel_label);

					Eigen::Vector3f vector(neighbor_supervoxel->centroid_.x - supervoxel->centroid_.x,
								neighbor_supervoxel->centroid_.y - supervoxel->centroid_.y,
								neighbor_supervoxel->centroid_.z - supervoxel->centroid_.z
								);

					if(std::abs(vector(0)) < min_abs_x)
					{
						min_abs_x = std::abs(vector(0));
						vertex_to_disconnect = adj_v;
					}
				}
	
				edges_to_remove.push_back(std::pair<sv_vertex_t, sv_vertex_t>(vertex, vertex_to_disconnect));

			}

			if(num_neighbors == 3)
			{
				pcl::Supervoxel<PointAT>::Ptr supervoxel = supervoxel_clusters.at(mst[vertex].supervoxel_label);

				std::vector<Eigen::Vector3f> vectors;

				std::vector<sv_vertex_t> adj_vertices;

				BGL_FORALL_ADJ(vertex, adj_v, mst, sv_graph_t)
				{
					pcl::Supervoxel<PointAT>::Ptr neighbor_supervoxel = supervoxel_clusters.at(mst[adj_v].supervoxel_label);

					Eigen::Vector3f vector(neighbor_supervoxel->centroid_.x - supervoxel->centroid_.x,
								neighbor_supervoxel->centroid_.y - supervoxel->centroid_.y,
								neighbor_supervoxel->centroid_.z - supervoxel->centroid_.z
								);

					vectors.push_back(vector);

					adj_vertices.push_back(adj_v);
				}

				int idx_to_cut = 2;

				float max_vertical = std::abs((vectors[0] - vectors[1]).normalized()(0));

				float tmp_vertical = std::abs((vectors[0] - vectors[2]).normalized()(0));

				if(tmp_vertical > max_vertical)
				{
					max_vertical = tmp_vertical;
					idx_to_cut = 1;
				}

				tmp_vertical = std::abs((vectors[1] - vectors[2]).normalized()(0));

				if(tmp_vertical > max_vertical)
				{
					max_vertical = tmp_vertical;
					idx_to_cut = 0;
				}

				edges_to_remove.push_back(std::pair<sv_vertex_t, sv_vertex_t>(vertex, adj_vertices[idx_to_cut]));
			}

			if(num_neighbors == 2)
			{
				pcl::Supervoxel<PointAT>::Ptr supervoxel = supervoxel_clusters.at(mst[vertex].supervoxel_label);

				std::vector<Eigen::Vector3f> vectors;

				std::vector<sv_vertex_t> adj_vertices;

				BGL_FORALL_ADJ(vertex, adj_v, mst, sv_graph_t)
				{
					pcl::Supervoxel<PointAT>::Ptr neighbor_supervoxel = supervoxel_clusters.at(mst[adj_v].supervoxel_label);

					Eigen::Vector3f vector(neighbor_supervoxel->centroid_.x - supervoxel->centroid_.x,
								neighbor_supervoxel->centroid_.y - supervoxel->centroid_.y,
								neighbor_supervoxel->centroid_.z - supervoxel->centroid_.z
								);

					vector.normalize();

					vectors.push_back(vector);

					adj_vertices.push_back(adj_v);
				}

				if( vectors[0].dot(vectors[1] ) >= -0.9f)
				{
					edges_to_remove.push_back(std::pair<sv_vertex_t, sv_vertex_t>(vertex, adj_vertices[0]));
					edges_to_remove.push_back(std::pair<sv_vertex_t, sv_vertex_t>(vertex, adj_vertices[1]));
				}

			}
		
		}

		if(edges_to_remove.size() == 0) break;

		for(auto edge : edges_to_remove) boost::remove_edge(edge.first, edge.second, mst);
	}

	PointCloudT::Ptr new_cloud(new PointCloudT);

	BGL_FORALL_VERTICES(vertex, mst, sv_graph_t)
	{
		auto search = supervoxel_clusters.find(mst[vertex].supervoxel_label);

		if(search != supervoxel_clusters.end())
		{
		
			pcl::Supervoxel<PointAT>::Ptr supervoxel = search->second;

			PointT p0;
			p0.x = supervoxel->centroid_.x;
			p0.y = supervoxel->centroid_.y;
			p0.z = supervoxel->centroid_.z;

			p0.r = p0.g = p0.b = 255;

			new_cloud->push_back(p0);

		}
	}

	pcl::PointIndices::Ptr inliers_line (new pcl::PointIndices);

	PointACloudT::Ptr centroid_cloud = super.getVoxelCentroidCloud();

	pcl::SACSegmentation<PointAT> seg;
	//segn.setOptimizeCoefficients (true);
	seg.setModelType(pcl::SACMODEL_LINE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(500);
	seg.setDistanceThreshold(0.03);
	seg.setInputCloud(centroid_cloud);
	seg.setAxis(Eigen::Vector3f::UnitX());
	seg.setEpsAngle(pcl::deg2rad(10.));

	// Obtain the plane inliers and coefficients
	seg.segment (*inliers_line, *coefficients_plane);



#if 1
/*	std::vector<int> inliers;
 	pcl::SampleConsensusModelLine<PointT>::Ptr model_l(new pcl::SampleConsensusModelLine<PointT> (new_cloud));

	pcl::RandomSampleConsensus<PointT> ransac(model_l);
	ransac.setDistanceThreshold (.04);
	ransac.computeModel();
	ransac.getInliers(inliers);

*/

	for(auto & i : inliers_line->indices)
	{
		centroid_cloud->points[i].r = 255;
		centroid_cloud->points[i].g = 255;
		centroid_cloud->points[i].b = 255;
		centroid_cloud->points[i].a = 255;
	}


#endif

	BGL_FORALL_EDGES(edge, mst, sv_graph_t)
	{
		sv_vertex_t s = boost::source(edge, mst);

		sv_vertex_t t = boost::target(edge, mst);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_s = supervoxel_clusters.at(mst[s].supervoxel_label);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_t = supervoxel_clusters.at(mst[t].supervoxel_label);

		pcl::PointXYZ p0, p1;
		p0.x = supervoxel_s->centroid_.x;
		p0.y = supervoxel_s->centroid_.y;
		p0.z = supervoxel_s->centroid_.z;

		p1.x = supervoxel_t->centroid_.x;
		p1.y = supervoxel_t->centroid_.y;
		p1.z = supervoxel_t->centroid_.z;

		viewer->addLine(p0, p1, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
	}

	viewer->spin();

	viewer->removeAllPointClouds();

	if(centroid_cloud->size() > 0)
		viewer->addPointCloud(centroid_cloud, "inliers", 0);

	viewer->spin();

	viewer->removeAllShapes();
	viewer->removeAllPointClouds();


	return true;



	//create vector type mst for tree filtering


	int mst_size = boost::num_vertices(mst);

	std::vector<TreeNode> mst_sorted_nodes_vec(mst_size);

	std::vector<bool> vertex_visited_map(mst_size, false);
	
	int sorted_node_id = 0;

	//viewer->addPointCloud(color_cloud, "color_cloud", 0);

	for(int vertex_id = 0; vertex_id < mst_size; ++vertex_id)
	{
		// check if node is visited
		if(vertex_visited_map[vertex_id]) continue;

		//this is a root node
		mst_sorted_nodes_vec[sorted_node_id].parent_node = -1;

		boost::graph_traits<sv_graph_t>::adjacency_iterator ai, a_end;

		// traverse connected components
		std::queue<int> vertices_queue;

		vertex_visited_map[vertex_id] = true;
		
		vertices_queue.push(vertex_id);

		while( !vertices_queue.empty() )
		{			
			const int cur_vertex = vertices_queue.front();	

			vertices_queue.pop();	

			boost::tie(ai, a_end) = boost::adjacent_vertices(cur_vertex, mst); 

			mst_sorted_nodes_vec[sorted_node_id].vertex_id = cur_vertex;

			for (; ai != a_end; ++ai) 	
			{
				if(vertex_visited_map[*ai]) 
				{
					mst_sorted_nodes_vec[sorted_node_id].parent_node = *ai;	
				}
				else
				{
					vertex_visited_map[*ai] = true;

					vertices_queue.push(*ai);
				
					mst_sorted_nodes_vec[sorted_node_id].children_nodes.push_back(*ai);
				}
			}

			sorted_node_id++;

		}

	}
	
	std::cout<<"sorted node id: "<<sorted_node_id<<"\n";


	float search_dist = 0.1f;

	// leaf to root
	for(int i=mst_sorted_nodes_vec.size()-1; i>=0; i--)
	{
		TreeNode * tree_node = &mst_sorted_nodes_vec[i];

		if(tree_node->children_nodes.size() == 0)	// leaf node
		{
			

		}


		for(auto & child_node : tree_node->children_nodes)
		{

			pcl::Supervoxel<PointAT>::Ptr supervoxel_s = supervoxel_clusters.at(mst[tree_node->vertex_id].supervoxel_label);

			pcl::Supervoxel<PointAT>::Ptr supervoxel_t = supervoxel_clusters.at(mst[child_node].supervoxel_label);

		}
	}

	// root to leaf
	for(int i=0; i<mst_sorted_nodes_vec.size(); i++)
	{
				const int cur_node =  mst_sorted_nodes_vec[i].vertex_id;

		const int parent_node = mst_sorted_nodes_vec[i].parent_node;

		
	}

	BGL_FORALL_EDGES(edge, mst, sv_graph_t)
	{
		sv_vertex_t s = boost::source(edge, mst);

		sv_vertex_t t = boost::target(edge, mst);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_s = supervoxel_clusters.at(mst[s].supervoxel_label);

		pcl::Supervoxel<PointAT>::Ptr supervoxel_t = supervoxel_clusters.at(mst[t].supervoxel_label);

		pcl::PointXYZ p0, p1;
		p0.x = supervoxel_s->centroid_.x;
		p0.y = supervoxel_s->centroid_.y;
		p0.z = supervoxel_s->centroid_.z;

		p1.x = supervoxel_t->centroid_.x;
		p1.y = supervoxel_t->centroid_.y;
		p1.z = supervoxel_t->centroid_.z;

		viewer->addLine(p0, p1, 1, 1, 1, "line"+std::to_string(cv::getTickCount()), 0);
	}
	
	viewer->spin();

	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

#endif

	return true;
}


void detectStems(PointCloudT::Ptr plant_cloud, NormalCloudT::Ptr plant_normals, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
		 std::vector<pcl::ModelCoefficients::Ptr> & cylinders)
{
	pcl::ExtractIndices<PointT> extract;
	PointCloudT::Ptr cloud(new PointCloudT);
	PointCloudT::Ptr tmp_cloud(new PointCloudT);

	cylinders.clear();
	*cloud = *plant_cloud;

	pcl::SACSegmentation<PointT> seg;
	seg.setModelType (pcl::SACMODEL_LINE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (500);
	seg.setDistanceThreshold (0.04);
	seg.setAxis(Eigen::Vector3f::UnitX());
	seg.setEpsAngle(pcl::deg2rad(10.));


	for(int i=0; i<4; i++)
	{
		pcl::ModelCoefficients::Ptr coefficients_line (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_line (new pcl::PointIndices);

		seg.setInputCloud (cloud);
		seg.segment (*inliers_line, *coefficients_line);


		if(coefficients_line->values.size() != 6	//no solution
		  || inliers_line->indices.size() < 200	//inliers too few
		  || std::abs(coefficients_line->values[3]) < 0.9	// not vertical 
		  ) continue;


		if(coefficients_line->values[3] > 0.)
		{
			coefficients_line->values[3] *= -1.;
			coefficients_line->values[4] *= -1.;
			coefficients_line->values[5] *= -1.;
		}


		double t = (0.65 - coefficients_line->values[0])/coefficients_line->values[3];

		coefficients_line->values[0] = 0.65;
		coefficients_line->values[1] += t*coefficients_line->values[4];
		coefficients_line->values[2] += t*coefficients_line->values[5];

		// cylinder near image edge, ignore
		if(std::abs(coefficients_line->values[1]) > 0.4 ) continue;

		//std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

		cylinders.push_back(coefficients_line);


		Eigen::Vector3f unit_line_dir;

		unit_line_dir << coefficients_line->values[3], coefficients_line->values[4], coefficients_line->values[5];

		tmp_cloud->points.clear();

		PointCloudT::Ptr inlier_cloud(new PointCloudT);

		for(int j=0; j<cloud->points.size(); j++)
		{
			PointT & p = cloud->points[j];

			Eigen::Vector3f vec;
			vec << coefficients_line->values[0] - p.x, 
			       coefficients_line->values[1] - p.y, 
			       coefficients_line->values[2] - p.z;

			float point_to_line_dist = (vec - vec.dot(unit_line_dir)*unit_line_dir).norm();

			if(point_to_line_dist > 0.15)
			{
				tmp_cloud->push_back(p);	
			}
			else
			{
				p.r = p.g = p.b = 255;
				inlier_cloud->push_back(p);
			}
		}

		*cloud = *tmp_cloud;
		std::cout<<i<<"\n";

#if 1
		viewer->addPointCloud(inlier_cloud,"inlier_cloud"+std::to_string(i), 0);
		viewer->addPointCloud(tmp_cloud,"tmp_cloud"+std::to_string(i), 0);
		//viewer->addCylinder(*coefficients_cylinder, "cylinder"+std::to_string(i), 0);
		viewer->spinOnce(pcl_view_time);
#endif
	}

//	for(int i=0; i<4; i++)
//	{
//		viewer->removePointCloud("inlier_cloud"+std::to_string(i), 0);
//		viewer->removeShape("cylinder"+std::to_string(i), 0);
//	}

//	viewer->addPointCloud(plant_cloud,"plant_cloud",0);
//	viewer->spinOnce(pcl_view_time);
	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

	return;


	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> segn;

	pcl::ExtractIndices<pcl::Normal> extract_normals;

	//segn.setOptimizeCoefficients (true);
	segn.setModelType (pcl::SACMODEL_CYLINDER);
	segn.setMethodType (pcl::SAC_RANSAC);
	segn.setNormalDistanceWeight (0.1);
	segn.setMaxIterations (5000);
	segn.setDistanceThreshold (0.04);
	segn.setRadiusLimits (0.005, 0.03);
	segn.setAxis(Eigen::Vector3f::UnitX());
	segn.setEpsAngle(pcl::deg2rad(10.));
	
	NormalCloudT::Ptr normal(new NormalCloudT);
	NormalCloudT::Ptr tmp_normal(new NormalCloudT);

	*normal = *plant_normals;

	for(int i=0; i<4; i++)
	{
		pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);

		segn.setInputCloud (cloud);
		segn.setInputNormals (normal);
		segn.segment (*inliers_cylinder, *coefficients_cylinder);


		if(coefficients_cylinder->values.size() != 7	//no solution
		  || inliers_cylinder->indices.size() < 200	//inliers too few
		  || std::abs(coefficients_cylinder->values[3]) < 0.9	// not vertical 
		  ) continue;


		if(coefficients_cylinder->values[3] > 0.)
		{
			coefficients_cylinder->values[3] *= -1.;
			coefficients_cylinder->values[4] *= -1.;
			coefficients_cylinder->values[5] *= -1.;
		}


		double t = (0.65 - coefficients_cylinder->values[0])/coefficients_cylinder->values[3];

		coefficients_cylinder->values[0] = 0.65;
		coefficients_cylinder->values[1] += t*coefficients_cylinder->values[4];
		coefficients_cylinder->values[2] += t*coefficients_cylinder->values[5];

		// cylinder near image edge, ignore
		if(std::abs(coefficients_cylinder->values[1]) > 0.4 ) continue;

		//std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

		cylinders.push_back(coefficients_cylinder);


		Eigen::Vector3f unit_line_dir;

		unit_line_dir << coefficients_cylinder->values[3], coefficients_cylinder->values[4], coefficients_cylinder->values[5];

		tmp_cloud->points.clear();
		tmp_normal->points.clear();

		PointCloudT::Ptr inlier_cloud(new PointCloudT);

		for(int j=0; j<cloud->points.size(); j++)
		{
			PointT & p = cloud->points[j];

			Eigen::Vector3f vec;
			vec << coefficients_cylinder->values[0] - p.x, 
			       coefficients_cylinder->values[1] - p.y, 
			       coefficients_cylinder->values[2] - p.z;

			float point_to_line_dist = (vec - vec.dot(unit_line_dir)*unit_line_dir).norm();

			if(point_to_line_dist > 0.15)
			{
				tmp_cloud->push_back(p);	
				tmp_normal->push_back(normal->points[j]);
			}
			else
			{
				p.r = p.g = p.b = 255;
				inlier_cloud->push_back(p);
			}
		}

		*cloud = *tmp_cloud;
		*normal = *tmp_normal;
		std::cout<<i<<"\n";

#if 1
		viewer->addPointCloud(inlier_cloud,"inlier_cloud"+std::to_string(i), 0);
		viewer->addPointCloud(tmp_cloud,"tmp_cloud"+std::to_string(i), 0);
		//viewer->addCylinder(*coefficients_cylinder, "cylinder"+std::to_string(i), 0);
		viewer->spinOnce(pcl_view_time);
#endif
	}

//	for(int i=0; i<4; i++)
//	{
//		viewer->removePointCloud("inlier_cloud"+std::to_string(i), 0);
//		viewer->removeShape("cylinder"+std::to_string(i), 0);
//	}

//	viewer->addPointCloud(plant_cloud,"plant_cloud",0);
//	viewer->spinOnce(pcl_view_time);
	viewer->removeAllPointClouds();
	viewer->removeAllShapes();
}

void addNormal(PointCloudT::Ptr cloud, PointNCloudT::Ptr cloud_with_normals)
{
	NormalCloudT::Ptr normals (new NormalCloudT);

	pcl::search::KdTree<PointT>::Ptr searchTree (new pcl::search::KdTree<PointT>);
	searchTree->setInputCloud (cloud);

	pcl::NormalEstimation<PointT, pcl::Normal> normalEstimator;
	normalEstimator.setInputCloud( cloud );
	normalEstimator.setSearchMethod(searchTree);
	normalEstimator.setRadiusSearch(normal_radius);
	normalEstimator.compute(*normals);

	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
}



bool processDepthFrameFileName(std::string data_folder, std::vector<std::vector<DepthFrame>> & depth_frame_vec_vec,
			       std::vector<PointCloudT::Ptr> & gps_cloud_vec, std::vector<std::vector<int>> & row_frame_range_vec_vec)
{
	std::vector<path> sensor_folder_vec;

	for(int i=1; i<=4; i++)
	{
		path sensor_folder = data_folder + "/" + std::to_string(i);

		if(!exists(sensor_folder)) 
		{
			std::cout<< "sensor "<<i<<" folder does not exists\n";
			return false;
		}

		sensor_folder_vec.push_back(sensor_folder);
	}

	gps_cloud_vec.clear();
	depth_frame_vec_vec.clear();
	row_frame_range_vec_vec.clear();

	for(int i=0; i<4; i++)
	{
		recursive_directory_iterator it(sensor_folder_vec[i]);
		recursive_directory_iterator endit;

		std::vector<DepthFrame> depth_frame_vec;

		while (it != endit)
		{
			if (is_regular_file(*it) && it->path().extension() == ".bgrd")
			{

				std::vector<std::string> strs;
				boost::split(strs, it->path().filename().string(), boost::is_any_of("DQ"));

				std::vector<std::string>::iterator sit = strs.begin();

			//	std::cout<<*(sit+6)<<" "<<*(sit+8)<<"\n";

				std::string odo_x_str = *(sit+6);
				std::string odo_y_str = *(sit+8);
				std::string odo_z_str = *(sit+10);

				double odo_x = std::atof(odo_x_str.c_str());
				double odo_y = std::atof(odo_y_str.c_str());
				double odo_z = std::atof(odo_z_str.c_str());
			
				// GPS
			//	std::cout << *(sit+14);
				std::string gps_line = *(sit+14);

				std::vector<std::string> gps_strs;
				boost::split(gps_strs, gps_line, boost::is_any_of(","));

				sit = gps_strs.begin();

		//		std::cout << *(sit)<<"\n";

				std::string utc_str = *(sit);
				std::string lat_str = *(sit+1);
				std::string lon_str = *(sit+3);

				double utc = std::atof(utc_str.c_str());
				double lat = std::atof(lat_str.c_str())*0.01;
			
				lat = std::atof(lat_str.substr(0,2).c_str()) + std::atof(lat_str.substr(2,10).c_str())/60.;
				double lon = std::atof(lon_str.c_str())*(-0.01);
				lon = std::atof(lon_str.substr(0,3).c_str()) + std::atof(lon_str.substr(3,10).c_str())/60.;
				lon *= -1.0;

				int zone;
				bool northp;
				double x, y;
				UTMUPS::Forward(lat, lon, zone, northp, x, y, 15);
				//std::string zonestr = UTMUPS::EncodeZone(zone, northp);
				//std::cout << fixed << setprecision(2) << zonestr << " " << x << " " << y << "\n";

				DepthFrame df;
				df.utm_x = x; df.utm_y = y;
				df.odo_x = odo_x; df.odo_y = odo_y;	
				df.odo_z = odo_z;
				df.utc = utc;
				df.file_path = it->path().string();
				depth_frame_vec.push_back(df);

	//			std::cout.precision(10);
	//			std::cout<<std::fixed<<lat<<" "<<lon<<" "<<x<<" "<<y<<"\n\n";
	//			std::getchar();
	//			for(; sit != strs.end(); ++sit) std::cout << *sit <<" ";
			}

			++it;
		}

		std::sort(depth_frame_vec.begin(), depth_frame_vec.end(), compareByUTC);

		depth_frame_vec_vec.push_back(depth_frame_vec);

		PointCloudT::Ptr gps_cloud(new PointCloudT);

		std::cout<<i<<": "<<depth_frame_vec.size()<<"\n";

		for(int j=0; j<depth_frame_vec.size(); j++)
		{
			DepthFrame* cur_df = &depth_frame_vec[j];

			cur_df->invalid = false;

			PointT p;

			p.x = cur_df->utm_x - utm_o_x;
			p.y = cur_df->utm_y - utm_o_y;

	#if 0
			if(j == 0)
			{
				p.x = (cur_df->utm_x - utm_o_x)*1.;
				p.y = cur_df->utm_y - utm_o_y;
			}
			else
			{
				double dist = std::sqrt(pow(cur_df->odo_x - pre_df1->odo_x, 2.)+pow(cur_df->odo_y - pre_df1->odo_y, 2.));
				if(dist < neighboring_gps_threshold)
				{
					Eigen::Vector2d unit_dir;

					unit_dir << cur_df->utm_x - pre_df1->utm_x, cur_df->utm_y - pre_df1->utm_y;

					unit_dir.normalize();

					p.x = unit_dir(0)*dist + gps_cloud->points.back().x;
					p.y = unit_dir(1)*dist + gps_cloud->points.back().y;
				}
				else
				{
					p.x = (cur_df->utm_x - utm_o_x)*1.;
					p.y = cur_df->utm_y - utm_o_y;
				}
			}
			
			pre_df1 = cur_df;
	#endif

			//p.x = cur_df->odo_x; p.y = cur_df->odo_y;
			p.z = 1.257;

			if(i == 0)
			{
				p.r = p.g = p.b = 255;
			}
			else if(i == 1)
			{
				p.g = p.b = 255; 
				p.r = 0;
			}
			else if(i == 2)
			{
				p.r = p.b = 255;
				p.g = 0;
			}
			else if(i == 3)
			{
				p.r = p.g = 255;
				p.b = 0;
			}
			
			gps_cloud->push_back(p);
		}

		gps_cloud_vec.push_back(gps_cloud);


		Eigen::Matrix4f gps2cam = Eigen::Matrix4f::Zero();

		/*robot front (GPS): not the e-stop side
		kinect id
		3: front bottom left, looking right
		4: front top left, looking right
		1: rear bottom right, looking left
		2: rear top right, looking left

		GPS center to kinect_3 length 96cm, width 11cm, height 58cm
		GPS center to kinect_1 length 89cm, width 11cm, height 56cm*/


		if(i == 0)	//kinect 1
		{
			gps2cam(2,0) = -1.;	
			gps2cam(0,1) = -1.;
			gps2cam(1,2) = 1.;
			gps2cam(0,3) = -0.89;	// measured with tape
			gps2cam(1,3) = -0.11;
			gps2cam(2,3) = -0.56;
			gps2cam(3,3) = 1.;

			gps2cam.block<3,3>(0,0) = gps2cam.block<3,3>(0,0)*Eigen::AngleAxisf(10./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();
		}
		else if(i == 1)
		{
			gps2cam(2,0) = -1.;
			gps2cam(0,1) = -1.;
			gps2cam(1,2) = 1.;
			gps2cam(0,3) = -0.89;	// measured with tape
			gps2cam(1,3) = -0.11;
			gps2cam(2,3) = -0.56 + 0.6;
			gps2cam(3,3) = 1.;

			gps2cam.block<3,3>(0,0) = gps2cam.block<3,3>(0,0)*Eigen::AngleAxisf(10./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();

		}
		else if(i == 2)
		{
			gps2cam(2,0) = -1.;
			gps2cam(0,1) = 1.;
			gps2cam(1,2) = -1.;
			gps2cam(0,3) = 0.96;	// measured with tape
			gps2cam(1,3) = 0.11;
			gps2cam(2,3) = -0.58;
			gps2cam(3,3) = 1.;

			gps2cam.block<3,3>(0,0) = gps2cam.block<3,3>(0,0)*Eigen::AngleAxisf(10./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();

		}
		else if(i== 3)
		{
			gps2cam(2,0) = -1.;
			gps2cam(0,1) = 1.;
			gps2cam(1,2) = -1.;
			gps2cam(0,3) = 0.96;	// measured with tape
			gps2cam(1,3) = 0.11;
			gps2cam(2,3) = -0.58 + 0.6;
			gps2cam(3,3) = 1.;

			gps2cam.block<3,3>(0,0) = gps2cam.block<3,3>(0,0)*Eigen::AngleAxisf(10./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();
		}


		//std::cout<<"gps2cam\n"<<gps2cam<<"\n";

		DepthFrame* cur_df = &depth_frame_vec[0];
		DepthFrame* pre_df = cur_df;
		const int turn_window_size = 5;

		std::vector<int> row_frame_range_vec(1, 0);

		bool pre_status = cur_df->invalid;

		// compute robot pose
		for(int j=0; j<gps_cloud->points.size(); j++, cur_df++)
		{
			Eigen::Matrix4f robot_pose = Eigen::Matrix4f::Identity();

			Eigen::Vector3f pre_point, cur_point, next_point;

			cur_point << gps_cloud->points[j].x, gps_cloud->points[j].y, gps_cloud->points[j].z;

			robot_pose.col(3).head<3>() = cur_point;

			float cur_pre_dist = 0.f, cur_next_dist = 0.f;

			if(j == 0)
			{
				next_point << gps_cloud->points[j+1].x, gps_cloud->points[j+1].y, gps_cloud->points[j+1].z;
				pre_point = cur_point;
				robot_pose.col(0).head<3>() = (next_point - cur_point).normalized();
			}
			else if(j < gps_cloud->points.size() - 1)
			{
				pre_point << gps_cloud->points[j-1].x, gps_cloud->points[j-1].y, gps_cloud->points[j-1].z;
				next_point << gps_cloud->points[j+1].x, gps_cloud->points[j+1].y, gps_cloud->points[j+1].z;

				cur_pre_dist = (cur_point - pre_point).norm();
				cur_next_dist = (cur_point - next_point).norm();

				if(cur_pre_dist < neighboring_gps_threshold && cur_next_dist < neighboring_gps_threshold)
					robot_pose.col(0).head<3>() = (next_point - pre_point).normalized();
				else if(cur_pre_dist < neighboring_gps_threshold)
					robot_pose.col(0).head<3>() = (cur_point - pre_point).normalized();
				else
					robot_pose.col(0).head<3>() = (next_point - cur_point).normalized();
			}
			else
			{
				pre_point << gps_cloud->points[j-1].x, gps_cloud->points[j-1].y, gps_cloud->points[j-1].z;
				next_point = cur_point;
				robot_pose.col(0).head<3>() = (cur_point - pre_point).normalized();
			}

			//robot_pose.col(0).head<3>() *= -1.0f;

			robot_pose.col(1).head<3>() = robot_pose.col(2).head<3>().cross(robot_pose.col(0).head<3>());

			*cur_df->world2robot = robot_pose;

			*cur_df->world2cam = robot_pose*gps2cam;

			//boundary
			if(cur_pre_dist > neighboring_gps_threshold || cur_next_dist > neighboring_gps_threshold)
			{
				gps_cloud->points[j].r = 255;
				gps_cloud->points[j].g = 0;
				gps_cloud->points[j].b = 0;

				cur_df->invalid = true;
			}
			else if((*cur_df->world2cam)(0,1)*(*pre_df->world2cam)(0,1) < 0.) // changing direction
			{
				gps_cloud->points[j].r = 255;
				gps_cloud->points[j].g = 0;
				gps_cloud->points[j].b = 0;

				for(int k = -turn_window_size; k<=turn_window_size; k++)
				{
					if( (j+k>=0) && (j+k <depth_frame_vec.size())) 	
					{
						depth_frame_vec[j+k].invalid = true;
					}
				}
			}
			else if(j != 0 && std::abs(robot_pose(0, 0)) < 0.9)
			{
				gps_cloud->points[j].r = 255;
				gps_cloud->points[j].g = 0;
				gps_cloud->points[j].b = 0;

				cur_df->invalid = true;
			}

			if(pre_status != cur_df->invalid)
			{
				if(cur_df->invalid)
				{
					row_frame_range_vec.push_back(j-1);				
				}
				else
					row_frame_range_vec.push_back(j);
			}

			pre_status = cur_df->invalid;
	
			pre_df = cur_df;
		}

		std::cout<<"row_frame_range_vec size:"<<row_frame_range_vec.size()<<"\n";

		for(int j=1; j<row_frame_range_vec.size(); j+=2)
		{
			gps_cloud->points[row_frame_range_vec[j]].r = 0;
			gps_cloud->points[row_frame_range_vec[j]].g = 255;
			gps_cloud->points[row_frame_range_vec[j]].b = 0;

			if(row_frame_range_vec[j] - row_frame_range_vec[j-1] < 100)
			{
				row_frame_range_vec[j] = row_frame_range_vec[j-1] = -1;
			}
		}

		row_frame_range_vec.erase(std::remove(row_frame_range_vec.begin(), row_frame_range_vec.end(), -1), row_frame_range_vec.end());

		if(row_frame_range_vec.size() % 2 != 0)
			row_frame_range_vec.push_back(gps_cloud->points.size()-1);

		std::cout<<"after remove row_frame_range_vec size:"<<row_frame_range_vec.size()<<"\n";

		row_frame_range_vec_vec.push_back(row_frame_range_vec);
	}

	return true;
}

std::string dateID2dateStr(int date_id) {

	switch(date_id) {

		case 0:
			return "0702";
		case 1:
			return "0709";
		case 2:
			return "0715";
		case 3:
			return "0723";
		case 4:
			return "0730";
		case 5:
			return "0806";
		case 6:
			return "0812";
		case 7:
			return "0820";
		case 8:
			return "0827";
		case 9:
			return "0903";
		default:
			return "";
	}
}

int main(int argc , char** argv)
{
	if(argc != 3) {

		cout<<"year id (0, 1), date id\n";
		return -1;
	}

	int year_id = atoi(argv[1]);

	if( year_id < 0 || year_id > 1) {

		cout<<"wrong year id\n";
		return -1;
	}

	int date_id = atoi(argv[2]);

	if( year_id == 0 ) {
		
		if(date_id < 0 || date_id > 3) {

			cout<<"wrong date id\n";
			return -1;
		}
	}
	else if( year_id == 1 ) {

		if(date_id < 0 || date_id > 9) {

			cout<<"wrong date id\n";
			return -1;
		}
	}

	cv::FileStorage fs("parameters.yml", cv::FileStorage::READ);
	fs["rr_num_neighbor"] >> rr_num_neighbor;
	fs["rr_residual"] >> rr_residual;
	fs["rr_curvature"] >> rr_curvature;
	fs["rr_angle"] >> rr_angle;
	fs["max_d_jump"] >> max_d_jump;
	fs["vox_size"] >> vox_size;
	fs["normal_radius"] >> normal_radius;
	fs["cylinder_ransac_dist"] >> cylinder_ransac_dist;
	fs["neighboring_gps_threshold"] >> neighboring_gps_threshold;
	fs["pcl_view_time"] >> pcl_view_time;
	fs["voxel_resolution"] >> voxel_resolution;
	fs["seed_resolution"] >> seed_resolution;
	fs["color_importance"] >> color_importance;
	fs["spatial_importance"] >> spatial_importance;
	fs["normal_importance"] >> normal_importance;
	fs["sor_meank"] >> sor_meank;
	fs["sor_std"] >> sor_std;
	fs["h_minvotes"] >> h_minvotes;
	fs["h_dx"] >> h_dx;
	fs["h_nlines"] >> h_nlines;
	fs["h_granularity"] >> h_granularity;
	fs["skeleton_iteration"] >> skeleton_iteration;
	fs["step_size"] >> step_size;
	fs["min_linearity"] >> min_linearity;
	fs["min_branch_size"] >> min_branch_size;
	fs["data_set"] >> data_set;
	fs["min_z"] >> min_z;
	fs["max_z"] >> max_z;
	fs["stem_verticality"] >> stem_verticality;
	fs["start_plant_id"] >> start_plant_id;
	fs["max_stem_radius"] >> max_stem_radius;
	fs["slice_thickness"] >> slice_thickness;
	fs["max_spatial_res"] >> max_spatial_res;

	fs.release();

	std::cout<<"parameters loaded\n";

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->addCoordinateSystem(0.01);
	viewer->registerPointPickingCallback(&pp_callback);
	viewer->setSize(900,700);
//	viewer->setBackgroundColor(0.8, 0.8, 0.8, 0);

	depth_canvas_cv.create(depth_height, 3*depth_width, CV_8U);
	pre_depth_cv.create(depth_width, depth_height, CV_8U);
	cur_depth_cv.create(depth_width, depth_height, CV_8U);
	pre_pre_depth_cv.create(depth_width, depth_height, CV_8U);

	BGRD bgrd_buffer[depth_size];

	std::string data_folder = "";

	if( year_id == 0 ) {

		switch(date_id) {
			case 0: 
				data_folder = "Data/081115";
				break;
			case 1:
				data_folder = "Data/081215";
				break;
			case 2:
				data_folder = "Data/081315";
				break;
			case 3:
				data_folder = "Data/081415";
				break;
		}
	}
	else if( year_id == 1 ) {

		data_folder = "CornData2017";

		for(; date_id < 10; date_id++) {

			std::string date_folder_str = data_folder + "/" + dateID2dateStr(date_id);

			cout<<"date id "<<date_id<<"\n";

			for(int plant_id = start_plant_id; plant_id < 20; plant_id++) {

				path plant_folder_path = date_folder_str + "/" + to_string(plant_id+1);

				if( date_id < 2 ) {

					recursive_directory_iterator it(plant_folder_path);
					recursive_directory_iterator endit;

					while (it != endit)
					{
						if (is_regular_file(*it) && it->path().extension() == ".bgrd") {

							PointCloudT::Ptr cloud(new PointCloudT);

							cout<<it->path().string()<<"\n";

							readDataBinaryFile(it->path().string(), bgrd_buffer);

							//flip horizontally and vertically
							if(date_id == 0) {

								BGRD tmp;
								
								for(int y=0; y<depth_height/2; y++) {
									
									for(int x=0; x<depth_width; x++) {
										
										tmp = bgrd_buffer[y*depth_width+x];
										bgrd_buffer[y*depth_width+x] = bgrd_buffer[(depth_height-y-1)*depth_width+depth_width-x-1];
										bgrd_buffer[(depth_height-y-1)*depth_width+depth_width-x-1] = tmp;
									}
								}
							}

							cv::Mat depth; depth.create(depth_height, depth_width, CV_8U);

							cv::Mat mask;

							int size = it->path().string().size();

							mask = cv::imread(it->path().string().substr(0, size-5) + "_mask.pgm", CV_LOAD_IMAGE_GRAYSCALE);

							unsigned char* ptr = depth.ptr<unsigned char>();

							for(int i=0; i<depth_size; ++i, ++ptr) {

								unsigned short & d = bgrd_buffer[i].d;
			
								if( d < 500 || d > 2000 ) 
									*ptr = 255;
								else 
									*ptr = (unsigned char)((d - 500)/1500.f*254.f);
							}

							std::string file_name = it->path().string().substr(0, size-5) + ".pgm";

							cv::imwrite(file_name, depth);

							if(mask.data != NULL) {

								cv::flip(mask, mask, -1);
	
								cv::imshow("mask", mask);

								ptr = mask.ptr<unsigned char>();
								for(int i=0; i<depth_size; i++, ptr+=1)
									if( (uint32_t)*ptr == 255 )
										bgrd_buffer[i].d = 0;
							}

							cv::imshow("depth", depth); cv::waitKey(100);

					
							readBGRD2PointCloud(bgrd_buffer, cloud, 1, min_z, max_z, true);	
							
				
							pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);

							pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);

							pcl::SACSegmentation<PointT> segp;
							segp.setOptimizeCoefficients(true);
							segp.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
							segp.setMethodType(pcl::SAC_RANSAC);
							segp.setMaxIterations(200);
							segp.setDistanceThreshold(0.02);
							segp.setInputCloud(cloud);
							segp.setAxis(Eigen::Vector3f::UnitX());
							segp.setEpsAngle(pcl::deg2rad(30.));
							segp.segment (*inliers_plane, *coefficients_plane);	

							Eigen::Vector4f plane_coeffs;

							plane_coeffs << coefficients_plane->values[0],
										coefficients_plane->values[1],
										coefficients_plane->values[2],
										coefficients_plane->values[3];

							
							

							uint32_t rgb = 180<<16 | 180<<8 | 180;

							for(auto & i : inliers_plane->indices)
								cloud->points[i].rgb = *reinterpret_cast<float*>(&rgb);

							// align ground with yz plane							
							Eigen::Matrix4f rot = Eigen::Matrix4f::Identity();
	
							rot.col(0).head<3>() = plane_coeffs.head<3>();

							cout<<rot.col(0).head(3).transpose()<<"\n";

							if(rot(0, 0) < 0.f )
								rot.col(0).head(3) *= -1.0f;
							
							rot.col(1).head(3) = Eigen::Vector3f::UnitY() - (rot.col(0).head(3).dot(Eigen::Vector3f::UnitY()))*rot.col(0).head(3);
							
							rot.col(1).head(3).normalize();

							rot.col(2).head<3>() = rot.col(0).head<3>().cross(rot.col(1).head<3>());

							pcl::transformPointCloud(*cloud, *cloud, rot.inverse());
							
							Eigen::Vector4f new_plane_coeffs;

							new_plane_coeffs = rot.inverse()*plane_coeffs;


							PointCloudT::Ptr plant_cloud(new PointCloudT);
						
							pcl::ExtractIndices<PointT> extract;
	
							extract.setInputCloud(cloud);

							extract.setIndices(inliers_plane);

							extract.setNegative(true);

							extract.filter(*plant_cloud);
							
							viewer->removeAllPointClouds();

							viewer->addPointCloud(cloud, "cloud");

						//	viewer->addPlane(*coefficients_plane, "plane"+to_string(cv::getTickCount())); viewer->spin();

							for(int i=0; i<4; i++) coefficients_plane->values[i] = new_plane_coeffs[i];

							viewer->removeAllShapes();
							
							viewer->addPlane(*coefficients_plane, "plane"+to_string(cv::getTickCount()));

							viewer->spin(); 

							skeleton(plant_cloud, new_plane_coeffs, viewer);
						}

						++it;

						break;
					}

				}
				else if (0){

					path plant_folder_path_0 = date_folder_str + "/" + to_string(plant_id+1) + "/" + "1";

					recursive_directory_iterator it(plant_folder_path_0);
					recursive_directory_iterator endit;

					while (it != endit)
					{
						if (is_regular_file(*it) && it->path().extension() == ".bgrd") {

							PointCloudT::Ptr cloud(new PointCloudT);

							readDataBinaryFile(it->path().string(), bgrd_buffer);

							readBGRD2PointCloud(bgrd_buffer, cloud, 1, 0.5f, 2.0f, true);	
			
							viewer->removeAllPointClouds();

							viewer->addPointCloud(cloud, "cloud");

							viewer->spin(); 
						}

						++it;
					}

				}
				

			}

		}

		return 0;
	}

	std::vector<std::vector<DepthFrame>> depth_frame_vec_vec;

	std::vector<PointCloudT::Ptr> gps_cloud_vec;

	std::vector<std::vector<int>> row_frame_range_vec_vec;

	double t = (double)cv::getTickCount();
	processDepthFrameFileName(data_folder, depth_frame_vec_vec, gps_cloud_vec, row_frame_range_vec_vec);
	std::cout <<"process file name time:"<< ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;

	int sensor_id = 3;

	std::vector<DepthFrame> depth_frame_vec = depth_frame_vec_vec[sensor_id];

#if 0
	for(int i=0; i<depth_frame_vec.size(); i++)
	{
		DepthFrame* df = &depth_frame_vec[i];

		BGRD buffer[depth_size];
	
		readDataBinaryFile(df->file_path, buffer);
		PointCloudT::Ptr cloud(new PointCloudT);
		readBGRD2PointCloud(buffer, cloud, i%4, false);

		viewer->addPointCloud(cloud, "cloud", 0);
		viewer->spin();
		viewer->removeAllPointClouds();
	}
#endif

#if 0	
	for(int i=0; i<4; i++)
	{

		for(auto & p : gps_cloud_vec[i]->points)
			p.y += i*0.5f;

		viewer->addPointCloud(gps_cloud_vec[i], "gps cloud"+std::to_string(i), 0);
		viewer->spin();
	}
#endif

	PointCloudT::Ptr gps_cloud = gps_cloud_vec[sensor_id];

	std::vector<int> row_frame_range_vec = row_frame_range_vec_vec[sensor_id];

	pcl::IterativeClosestPointWithNormals<PointNT, PointNT>::Ptr icpn(new pcl::IterativeClosestPointWithNormals<PointNT, PointNT>);
	icpn->setMaxCorrespondenceDistance(0.1);
	icpn->setMaximumIterations(30);

	pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr icp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>);
	icp->setMaxCorrespondenceDistance(0.05);
	icp->setMaximumIterations(100);
	icp->setEuclideanFitnessEpsilon(0.0001);
	

	pcl::registration::IncrementalRegistration<PointT> iicp;
	iicp.setRegistration (icp);

	pcl::PassThrough<PointT> pass;

	pass.setFilterFieldName ("z");
	pass.setFilterLimits(0.05, 2.);
	//pass.setFilterLimitsNegative (true);

	PointCloudT::Ptr total_cloud(new PointCloudT);

	
	double last_utc = -1.0; 

	double first_utc = 1e10;

	for(int i=0; i<4; i++)
	{
		first_utc = depth_frame_vec_vec[i].front().utc < first_utc ? depth_frame_vec_vec[i].front().utc : first_utc;

		last_utc = depth_frame_vec_vec[i].back().utc > last_utc ? depth_frame_vec_vec[i].back().utc : last_utc;

		std::cout<<depth_frame_vec_vec[i].back().utc<<" "<<depth_frame_vec_vec[i].back().utc<<"\n";
	}

	std::cout<<first_utc<<" "<<last_utc<<"\n";

	double cur_utc = first_utc;

	std::vector<int> cur_frame_idx_vec(4, 0);

	std::vector<int> cur_row_idx_vec(4, 0);

	float vox_viz_size = 0.04f;

	

	pcl::VoxelGrid<PointT> vox;
	vox.setLeafSize(vox_size, vox_size, vox_size);

	while(cur_utc != last_utc)
	{
		for(int cam_id=0; cam_id<4; cam_id++)
		{
			std::vector<DepthFrame> depth_frame_vec = depth_frame_vec_vec[cam_id];

			int frame_idx = cur_frame_idx_vec[cam_id];

			if(frame_idx > row_frame_range_vec_vec[cam_id][cur_row_idx_vec[cam_id]+1])
			{
				cur_row_idx_vec[cam_id] = cur_row_idx_vec[cam_id] + 2 < row_frame_range_vec_vec[cam_id].size() ? cur_row_idx_vec[cam_id] + 2 : -1;
			}

			
			for(; frame_idx != -1 && depth_frame_vec[frame_idx].utc <= cur_utc; frame_idx++)
			{

				if(frame_idx < row_frame_range_vec_vec[cam_id][cur_row_idx_vec[cam_id]] 
				   || frame_idx > row_frame_range_vec_vec[cam_id][cur_row_idx_vec[cam_id]+1])
					continue;

				DepthFrame* df = &depth_frame_vec[frame_idx];

				//if(df->invalid)	continue;

				double t = (double)cv::getTickCount();
	
				readDataBinaryFile(df->file_path, bgrd_buffer);
				PointCloudT::Ptr cloud(new PointCloudT);
				readBGRD2PointCloud(bgrd_buffer, cloud, frame_idx%4, 0.5f, 1.524f, false);

				if(cloud->points.size() < 400) continue;

				PointCloudT::Ptr plant_cloud(new PointCloudT);
				NormalCloudT::Ptr plant_normals(new NormalCloudT);
				PointCloudT::Ptr soil_cloud(new PointCloudT);
				PointCloudT::Ptr tmp_cloud(new PointCloudT);
				pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);

				std::vector<pcl::ModelCoefficients::Ptr> stem_lines;

				bool found_soil = false;

#if 1
				// bottom cameras, fit soil plane
				if(cam_id == 0 || cam_id == 2)
				{
					Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();		
					transform.block<3,3>(0,0) = transform.block<3,3>(0,0)*Eigen::AngleAxisf(7./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();

					if(separatePlantsFromSoil(cloud, transform, plant_cloud, plant_normals, soil_cloud, coefficients_plane, stem_lines, viewer))
					{
						*plant_cloud += *soil_cloud;

						Eigen::Matrix4f cam_to_world = (*(df->world2cam))*transform.inverse();

						pcl::transformPointCloud(*plant_cloud, *tmp_cloud, cam_to_world);

				
						for(auto & l : stem_lines)
						{
							Eigen::Vector4f p(l->values[0], l->values[1], l->values[2], 1.0f);

							Eigen::Vector3f dir(l->values[3], l->values[4], l->values[5]);

							p = cam_to_world * p;

							dir = cam_to_world.topLeftCorner<3,3>() * dir;

							l->values[0] = p(0); l->values[1] = p(1); l->values[2] = p(2);
							l->values[3] = dir(0); l->values[4] = dir(1); l->values[5] = dir(2);
						}
					}
					else
					{
						pcl::transformPointCloud(*cloud, *tmp_cloud, (*(df->world2cam)));
					}

					
				}
				else	//top cameras
				{
					pcl::transformPointCloud(*cloud, *tmp_cloud, (*(df->world2cam)));
					
				}
#endif


//				vox.setInputCloud(tmp_cloud);
//			  	vox.filter(*cloud);

				//std::vector<pcl::ModelCoefficients::Ptr> cylinders;

				//detectStems(cloud, plant_normals, viewer, cylinders);


//				viewer->addPointCloud(cloud, "cloud"+std::to_string(cv::getTickCount()), 0);

//				for(auto & l : stem_lines)
//					viewer->addLine(*l, "stem"+std::to_string(cv::getTickCount()), 0);

				//std::cout <<"time:"<< ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;
		
//				viewer->spinOnce(pcl_view_time);
			}


			cur_frame_idx_vec[cam_id] = frame_idx < depth_frame_vec.size() ? frame_idx : -1;

		}

		double min_dist = 1e10;
		int next_cam_idx = -1;

		for(int cam_id=0; cam_id<4; cam_id++)
		{
			double tmp_dist = depth_frame_vec_vec[cam_id][cur_frame_idx_vec[cam_id]].utc - cur_utc;
			
			if(tmp_dist < min_dist)
			{
				min_dist = tmp_dist;
				next_cam_idx = cam_id;
			}
		}

		cur_utc = depth_frame_vec_vec[next_cam_idx][cur_frame_idx_vec[next_cam_idx]].utc;		
	}


	viewer->spinOnce(pcl_view_time);

	for(int row_id=1; row_id<row_frame_range_vec.size(); row_id += 2) 
	{
		int end_id = row_frame_range_vec[row_id];
		int start_id = row_frame_range_vec[row_id-1];
		if( end_id == -1) continue;


		std::cout<<"gps :"<<gps_cloud->points[start_id]<<gps_cloud->points[end_id]<<"\n";

		std::cout<<"odo :"<<depth_frame_vec[start_id].odo_x<<" "<<depth_frame_vec[start_id].odo_y<<"\n"<<depth_frame_vec[end_id].odo_x<<" "<<depth_frame_vec[end_id].odo_y<<"\n\n";

		std::cout<<"num points "<<end_id-start_id+1<<"\n";
		

		#pragma omp parallel for
		for(int i=start_id; i<end_id; i++)
		{
			DepthFrame* df = &depth_frame_vec[i];

			//if(df->invalid)	continue;

			double t = (double)cv::getTickCount();

			BGRD buffer[depth_size];
	
			readDataBinaryFile(df->file_path, buffer);
			PointCloudT::Ptr cloud(new PointCloudT);
			readBGRD2PointCloud(buffer, cloud, i%4, 0.5f, 1.524f, false);

			if(cloud->points.size() < 100) continue;

			PointCloudT::Ptr plant_cloud(new PointCloudT);
			NormalCloudT::Ptr plant_normal(new NormalCloudT);
			PointCloudT::Ptr soil_cloud(new PointCloudT);
			PointCloudT::Ptr tmp_cloud(new PointCloudT);
			pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients);

			Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();		
			transform.block<3,3>(0,0) = transform.block<3,3>(0,0)*Eigen::AngleAxisf(10./180.*M_PI, Eigen::Vector3f::UnitY()).matrix();

	/*		if(!separatePlantsFromSoil(cloud, transform, plant_cloud, plant_normal, soil_cloud, coefficients_plane))
			{
				std::cout<<"ground not found\n";
				continue;
			}
*/

			std::vector<pcl::ModelCoefficients::Ptr> cylinders;

//			detectStems(plant_cloud, plant_normal, viewer, cylinders);
//			continue;

			for(auto & c : cylinders)
			{
				Eigen::Vector4f p;
				p << c->values[0], c->values[1], c->values[2], 1.0;
				Eigen::Vector3f v;
				v << c->values[3], c->values[4], c->values[5];

				p = (*df->world2cam)*transform.inverse()*p;
				v = (*df->world2cam).topLeftCorner(3,3)*transform.inverse().topLeftCorner(3,3)*v;

				c->values[0] = p(0); c->values[1] = p(1); c->values[2] = p(2); 
				c->values[3] = v(0); c->values[4] = v(1); c->values[5] = v(2); 
				viewer->addCylinder(*c, "cylinder"+std::to_string(cv::getTickCount()), 0);
			}

			pcl::transformPointCloud(*cloud, *tmp_cloud, *(df->world2cam));

			pcl::copyPointCloud(*tmp_cloud, *cloud);
			

#if 1

			pcl::VoxelGrid<PointT> vox;
			vox.setInputCloud(cloud);
	  		vox.setLeafSize(0.04f, 0.04f, 0.04f);
		  	vox.filter(*tmp_cloud);
			pcl::copyPointCloud(*tmp_cloud, *cloud);

			#pragma omp critical
			{
				viewer->addPointCloud(cloud, "cloud"+std::to_string(cv::getTickCount()), 0);
			}

			std::cout <<"time:"<< ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;

		//	viewer->addPointCloud(plant_cloud, "plant_cloud", 0);
		//	viewer->addPointCloud(soil_cloud, "soil_cloud", 0);
		//	Eigen::Affine3f affine;
		//	affine.matrix() = *(df->world2cam);
		//	viewer->addCoordinateSystem(0.15, affine, "cam_pose"+std::to_string(i), 0);
		//	viewer->spinOnce(pcl_view_time);
		//	viewer->removePointCloud("plant_cloud", 0);
		//	viewer->removePointCloud("soil_cloud", 0);			
#endif

		}

		viewer->spin();	
	}

	return 0;
}
