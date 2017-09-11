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
#include <pcl/surface/mls.h>
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
int orthogonal_LSQ(const PointCloud3 &pc, Vector3d* a, Vector3d* b)
{
	int rc = 0;

	// anchor point is mean value
	*a = pc.meanValue();

	// copy points to libeigen matrix
	Eigen::MatrixXf points = Eigen::MatrixXf::Constant(pc.points.size(), 3, 0);
	for (int i = 0; i < points.rows(); i++)
	{
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
		double opt_dx = 0.04, int opt_nlines = 4, int opt_minvotes = 2, int granularity = 4)
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
	if (d == 0.0) 
	{
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

	try 
	{
		hough = new Hough(minPshifted, maxPshifted, opt_dx, granularity);
	} 
	catch (const std::exception &e) 
	{
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
	double rc;
	unsigned int nvotes;
	int nlines = 0;
	do 
	{
		Vector3d a; // anchor point of line
		Vector3d b; // direction of line

		hough->subtract(Y); // do it here to save one call

		nvotes = hough->getLine(&a, &b);
		
		X.pointsCloseToLine(a, b, opt_dx, &Y);

		rc = orthogonal_LSQ(Y, &a, &b);
		if (rc==0.0) break;

		X.pointsCloseToLine(a, b, opt_dx, &Y);
		nvotes = Y.points.size();
		if (nvotes < (unsigned int)opt_minvotes) break;

		rc = orthogonal_LSQ(Y, &a, &b);
		if (rc==0.0) break;

		//check if line vertical enough
		if ( std::abs(b.x) < 0.95 )
		{
			X.removePoints(Y);
			nlines++;
			continue;	
		}

		a = a + X.shift;

		nlines++;

		if(b.x > 0.) b = b*-1.;

		double t = (0.65 - a.x)/b.x;

		a.x = 0.65;
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
	float convexity;
	//float x, y, z;
	//PointCloudT cloud;
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

struct DepthFrame
{
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

	DepthFrame()
	{
		world2cam = new Eigen::Matrix4f();
		world2robot = new Eigen::Matrix4f();
	}
};

struct BGRD
{
	unsigned char b;
	unsigned char g;
	unsigned char r;
	unsigned short d;
};

bool compareByUTC(const DepthFrame & a, const DepthFrame & b)
{
	return a.utc < b.utc;
}

const int depth_width = 512;
const int depth_height = 424;
const int depth_size = depth_width*depth_height;

void readDataBinaryFile(std::string file_name, BGRD* buffer)
{
	std::ifstream in(file_name, std::ios::in | std::ios::binary);

	in.read((char*)&buffer[0], depth_size*sizeof(BGRD));

	in.close();
}



cv::Mat cur_depth_cv, pre_depth_cv, pre_pre_depth_cv, depth_canvas_cv;


void spectralCluster(PointCloudT::Ptr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{
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

	//PointLCloudT::Ptr color_cloud = super.getLabeledCloud();

	sv_graph_t sv_graph;

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

		uint32_t s_l = supervoxel_adjacency_list[s];
		uint32_t t_l = supervoxel_adjacency_list[t];

		sv_vertex_t sv_s = (label_ID_map.find(s_l))->second;
		sv_vertex_t sv_t = (label_ID_map.find(t_l))->second;

		sv_edge_t sv_edge;
		bool edge_added;

		boost::tie(sv_edge, edge_added) = boost::add_edge(sv_s, sv_t, sv_graph);

		if(edge_added)
		{
			pcl::Supervoxel<PointT>::Ptr svs = supervoxel_clusters.at(s_l);
			pcl::Supervoxel<PointT>::Ptr svt = supervoxel_clusters.at(t_l);

			sv_graph[sv_edge].weight = supervoxel_adjacency_list[edge];//*( 1.0f- std::abs(dir(0)) );
		}

	}


	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
	{
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

	Eigen::MatrixXf W = Eigen::MatrixXf::Constant(num_v, num_v, 0);

	Eigen::MatrixXf D = Eigen::MatrixXf::Constant(num_v, num_v, 0);

	BGL_FORALL_EDGES(edge, sv_graph, sv_graph_t)
	{
		int s = boost::source(edge, sv_graph);
		int t = boost::target(edge, sv_graph);

		const float weight = 1.0f;//exp(-sv_graph[edge].weight);

		W(s, t) = W(t, s) = weight;

		D(t, t) += weight;
		D(s, s) += weight;
	}

	Eigen::MatrixXf L = D - W;

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

	for(int i=0; i<num_v; i++)		
	{
		PointT p;
		p.x = eigvecs(i, 1);
		p.y = eigvecs(i, 2);
		p.z = eigvecs(i, 3);
		p.r = p.g = p.b = 255;

		spectral_cloud->push_back(p);

		pcl::Supervoxel<PointT>::Ptr sv = supervoxel_clusters.at(sv_graph[i].supervoxel_label);

		PointT p1;
		p1.x = sv->centroid_.x;
		p1.y = sv->centroid_.y;
		p1.z = sv->centroid_.z;

		viewer->addLine(p1, p, "c"+to_string(i));
		
	}

	viewer->addPointCloud(spectral_cloud, "spectral");


	viewer->spin();

	viewer->removePointCloud("spectral");
	viewer->removeAllShapes();

}


void readBGRD2PointCloud(BGRD* buffer, PointCloudT::Ptr cloud, int color, bool color_hist_equalization = true)
{
	cloud->points.clear();

	// BGRD is flipped horizontally
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

		cv::imshow("color"+std::to_string(cv::getTickCount()),color); cv::waitKey(0);

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

	for (int y = 1; y < depth_height-1; y++)
	{
		for (int x = 1; x < depth_width-1; x++)
		{
			int centerIdx = y*depth_width + x;
		
			int maxJump = 0;

			int centerD = depth_buffer[centerIdx];

			for (int h = -1; h <= 1; h+=1)
			{
				for (int w = -1; w <= 1; w+=1)
				{
					if(h == 0 && w == 0) continue;

					int neighborD = std::abs(centerD - depth_buffer[centerIdx + h*depth_width + w]);

					if(neighborD > maxJump) maxJump = neighborD;
				}
			}

			if(maxJump > max_d_jump) buffer[centerIdx].d = 10000;
		}
	}
#endif		

#if 0
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

	cv::Mat binary;
	binary = depth_u16 < 1524;
	
	cv::imshow("binary", binary);

	cv::imshow("cur_depth", depth_cv); 
//	cv::imshow("pre_depth", pre_depth_cv);
//	cv::imshow("pre_pre_depth", pre_pre_depth_cv);
	cv::waitKey(0);
//	pre_depth_cv.copyTo(pre_pre_depth_cv);
//	depth_cv.copyTo(pre_depth_cv);
#endif
	
	BGRD* ptr = buffer;
	for(int y=0; y<depth_height; y++)
	{
		for(int x=0; x<depth_width; x++, ptr++)
		{
			PointT p;

			p.z = ptr->d*0.001f;

			if(p.z > 1.524f || p.z < 0.5f) continue;

			if(color_hist_equalization)
			{
				p.b = img_hist_equalized.at<cv::Vec3b>(y,x)[0];
				p.g = img_hist_equalized.at<cv::Vec3b>(y,x)[1];
				p.r = img_hist_equalized.at<cv::Vec3b>(y,x)[2];
			}
			else
			{
				if(color == 0)
				{
					p.r = 255; p.b = 0; p.g = 0;
				}
				else if(color == 1)
				{ 	
					p.r = 0; p.b =0; p.g = 255;
				}
				else if(color == 2)
				{ 	
					p.r = 0; p.b =255; p.g = 0;
				}
				else if(color == 3)
				{
					p.r = 255; p.b =0; p.g = 255;
				}
	
				//p.b = ptr->b; p.g = ptr->g; p.r = ptr->r;
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

	Eigen::VectorXf plane_eigen(4);
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

	for(auto & p : soil_cloud->points)
		p.r = p.g = p.b = 100;

	cout<<"plant cloud size "<<plant_cloud->size()<<"\n";

	if(plant_cloud->size() < 2000) return true;


#if 1
	pcl::StatisticalOutlierRemoval<PointT> sor;
	sor.setInputCloud (plant_cloud);
	sor.setMeanK (sor_meank);
	sor.setStddevMulThresh (sor_std);
	sor.filter (*tmp_cloud);

//	*tmp_cloud = *plant_cloud;

	// Create a KD-Tree
	pcl::search::KdTree<PointT>::Ptr tree_mls (new pcl::search::KdTree<PointT>);

	// Output has the PointNormal type in order to store the normals calculated by MLS
	pcl::PointCloud<PointT> mls_points;

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

	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(mls_points, indices);

	cout<<"nan size: "<<mls_points.size()-indices.size()<<"\n";

	pcl::copyPointCloud(mls_points, *tmp_cloud);

	// convexity feature test
#if 0
	pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
	super.setInputCloud(tmp_cloud);
	super.setColorImportance (color_importance);
	super.setSpatialImportance (spatial_importance);
	super.setNormalImportance (normal_importance);

	std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

	super.extract(supervoxel_clusters);

	if(supervoxel_clusters.size() <= 100) return true;

	PointLCloudT::Ptr labeled_cloud = super.getLabeledCloud();

	//viewer->addPointCloud(labeled_cloud, "label");

	Graph pcl_sv_graph;

	super.getSupervoxelAdjacencyList(pcl_sv_graph);

	sv_graph_t sv_graph;

	std::map<uint32_t, sv_vertex_t> label_ID_map;

	BGL_FORALL_VERTICES(vertex, pcl_sv_graph, Graph)
	{
		sv_vertex_t v = boost::add_vertex(sv_graph);

		sv_graph[v].supervoxel_label = pcl_sv_graph[vertex];
		sv_graph[v].convexity = 0.f;

		label_ID_map.insert(std::make_pair(pcl_sv_graph[vertex], v));
	}

	BGL_FORALL_EDGES(edge, pcl_sv_graph, Graph)
	{
		Voxel s = boost::source(edge, pcl_sv_graph);
		Voxel t = boost::target(edge, pcl_sv_graph);

		uint32_t s_l = pcl_sv_graph[s];
		uint32_t t_l = pcl_sv_graph[t];

		sv_vertex_t sv_s = (label_ID_map.find(s_l))->second;
		sv_vertex_t sv_t = (label_ID_map.find(t_l))->second;

		sv_edge_t sv_edge;
		bool edge_added;

		boost::tie(sv_edge, edge_added) = boost::add_edge(sv_s, sv_t, sv_graph);

		if(edge_added)
		{
			//pcl::Supervoxel<PointT>::Ptr svs = supervoxel_clusters.at(s_l);
			//pcl::Supervoxel<PointT>::Ptr svt = supervoxel_clusters.at(t_l);

			sv_graph[sv_edge].weight = pcl_sv_graph[edge];//*( 1.0f- std::abs(dir(0)) );
		}
	}

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

		if(one_ring_cloud->size() < 10 ) continue;

		pcl::VoxelGrid<PointT> vox;
		vox.setLeafSize(voxel_resolution,voxel_resolution,voxel_resolution);
		vox.setInputCloud(one_ring_cloud);
		PointCloudT::Ptr filtered_cloud(new PointCloudT);
		vox.filter(*filtered_cloud);


		pcl::ConvexHull<PointT> chull;
		chull.setDimension(3);
		chull.setComputeAreaVolume(true);

		chull.setInputCloud (filtered_cloud);	

		std::vector<pcl::Vertices> vertices_chull;
		PointCloudT::Ptr cloud_hull(new PointCloudT);

		chull.reconstruct (*cloud_hull, vertices_chull);

		double H = std::pow(voxel_resolution, 3.0)*filtered_cloud->size()/chull.getTotalVolume();

		sv_graph[vertex].convexity = H;

		cout<<"H "<<H<<"\n";

		for(auto & p : sv->voxels_->points)
		{
			p.r = p.g = p.b = (unsigned char)std::min(H/1.5*255, 255.);
		}		
		
		//double area = chull.getTotalArea();


		if(H > 1.)
		{
		viewer->addPolygonMesh<PointT>(cloud_hull, vertices_chull, "convex hull"+std::to_string(cv::getTickCount()));

		viewer->addPointCloud(filtered_cloud, "ring"+to_string(cv::getTickCount()));

		viewer->spin();

		viewer->removeAllPointClouds();
		viewer->removeAllShapes();
		}

	}

	viewer->spin();

	viewer->removeAllPointClouds();
	viewer->removeAllShapes();

	return true;
#endif


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

	//viewer->addPointCloud(colored_cloud,"color", 0);
	//viewer->spin();
	//viewer->removeAllPointClouds();

	PointCloudT::Ptr init_stem_cloud(new PointCloudT);

	std::vector<int> canopy_indices;

	for(auto & indices : clusters)
	{
		PointCloudT::Ptr rr_segment_cloud(new PointCloudT);

		pcl::copyPointCloud(*tmp_cloud, indices, *rr_segment_cloud);

		//*init_stem_cloud += *rr_segment_cloud;
		

/*		pcl::octree::OctreePointCloud<PointT> octree(vox_size);
		octree.setInputCloud(rr_segment_cloud);
		octree.addPointsFromInputCloud();

		std::vector<PointT, Eigen::aligned_allocator<PointT> > voxelCenters;
  		int num_voxels = octree.getOccupiedVoxelCenters (voxelCenters);

		pcl::ConvexHull<PointT> chull;
		chull.setDimension(3);
		chull.setComputeAreaVolume(true);

		chull.setInputCloud (rr_segment_cloud);	

		std::vector<pcl::Vertices> vertices_chull;
		PointCloudT::Ptr cloud_hull(new PointCloudT);

		chull.reconstruct (*cloud_hull, vertices_chull);

		double H = std::pow(vox_size, 3.0)*rr_segment_cloud->size()/chull.getTotalVolume();
*/
		//cout<<H<<"\n";

		//if(H > 1.) continue;

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

		//check root position
		if( a_vec[i](2) > 1.3f || a_vec[i](2) < 0.7f) valid_line = false;
		
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

		if(!valid_line)
		{
			pcl::PointXYZ p1(a_vec[i](0), a_vec[i](1), a_vec[i](2));

			Eigen::Vector3f vector = a_vec[i] + line_len*b_vec[i];

			pcl::PointXYZ p2(vector(0), vector(1), vector(2));

			viewer->addLine(p1, p2, 0, 1, 0, "line"+to_string(cv::getTickCount()));	

			continue;
		}		

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

		pcl::PointXYZ p1(a_vec[i](0), a_vec[i](1), a_vec[i](2));

		Eigen::Vector3f vector = a_vec[i] + line_len*b_vec[i];

		pcl::PointXYZ p2(vector(0), vector(1), vector(2));

		//viewer->spin();


		PointCloudT::Ptr super_cloud_in(new PointCloudT);

		pcl::copyPointCloud(*colored_cloud, inliers_line, *super_cloud_in);

		cout<<"super_cloud_in size "<<super_cloud_in->size()<<"\n";


	//	viewer->removeAllShapes();
	//	spectralCluster(super_cloud_in, viewer);	continue;

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
		sv_graph_t sv_graph;

		std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;


		viewer->removeAllShapes();

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
	
			BGL_FORALL_VERTICES(vertex, sv_graph, sv_graph_t)
			{
				pcl::Supervoxel<PointT>::Ptr supervoxel = sv_graph[vertex].supervoxel;

				Eigen::Vector3f centroid = supervoxel->centroid_.getVector3fMap();

				int num_neighbors = boost::out_degree(vertex, sv_graph);

				bool triangle = false;

				if(num_neighbors == 0)
					continue;				
				else if(num_neighbors == 1)
				{
					*super_cloud_in += *supervoxel->voxels_;
					continue;
				}
				else if(num_neighbors == 2)
				{
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
					  ) 
					{
						*super_cloud_in += *supervoxel->voxels_;
						continue;
					}
				}


				PointACloudT::Ptr neighbor_centroids(new PointACloudT);

				neighbor_centroids->push_back(supervoxel->centroid_);

				Eigen::Vector3f neighbor_mean(0,0,0);

				float weight_sum = 0.0f;

				BGL_FORALL_ADJ(vertex, adj_v, sv_graph, sv_graph_t)
				{
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
		viewer->addPointCloud(colored_cloud, "super", 0);
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


bool processDepthFrameFileName(std::string data_folder, std::string section, std::vector<std::vector<DepthFrame>> & depth_frame_vec_vec,
			       std::vector<PointCloudT::Ptr> & gps_cloud_vec, std::vector<std::vector<int>> & row_frame_range_vec_vec)
{
	std::vector<path> sensor_folder_vec;

	for(int i=1; i<=4; i++)
	{
		path sensor_folder = data_folder + "/" + std::to_string(i) + section;

		if(!exists(sensor_folder)) 
		{
			std::cout<< "sensor "<<i<<section<<" folder does not exists\n";
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

int main()
{



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

	fs.release();

	std::cout<<"parameters loaded\n";

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//	viewer->addCoordinateSystem(0.25);
	viewer->registerPointPickingCallback(&pp_callback);
	viewer->setSize(900,700);
//	viewer->setBackgroundColor(0.8, 0.8, 0.8, 0);

	depth_canvas_cv.create(depth_height, 3*depth_width, CV_8U);
	pre_depth_cv.create(depth_width, depth_height, CV_8U);
	cur_depth_cv.create(depth_width, depth_height, CV_8U);
	pre_pre_depth_cv.create(depth_width, depth_height, CV_8U);

	
	std::string data_folder = "Data/081215";

	std::vector<std::vector<DepthFrame>> depth_frame_vec_vec;

	std::vector<PointCloudT::Ptr> gps_cloud_vec;

	std::vector<std::vector<int>> row_frame_range_vec_vec;

	double t = (double)cv::getTickCount();
	processDepthFrameFileName(data_folder, "a", depth_frame_vec_vec, gps_cloud_vec, row_frame_range_vec_vec);
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

	BGRD bgrd_buffer[depth_size];

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
				readBGRD2PointCloud(bgrd_buffer, cloud, frame_idx%4, false);

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
			readBGRD2PointCloud(buffer, cloud, i%4, false);

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
