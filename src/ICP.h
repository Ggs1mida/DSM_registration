/*
Iterative Closest Point

Performs the iterative closest point algorithm. A point cloud 'dynamicPointCloud' is transformed such that
it best "matches" the point cloud 'staticPointCloud'.

this program uses the method outlined in [1].

1. for each point in the dynamic point cloud, find the closest point in the static point cloud.
2. solve for an affine transform which minimizes the distance between all dynamic points and their respective static points.
3. transform the dynamic points.
4. goto -> 1 until stopping conditions are met.

affine transforms are solved using SVD.

@author Greg Smith 2017

[1] Arun, K. Somani, Thomas S. Huang, and Steven D. Blostein. "Least-squares fitting of two 3-D point sets." 
	IEEE Transactions on pattern analysis and machine intelligence 5 (1987): 698-700.

*/

#pragma once
#include <stdio.h>
#include <vector>
#include <random>
#include "kdtree.h"
#include <ctime>
#include "svd.h"
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <omp.h>


namespace gs
{
	enum ICP_TYPE {Point2Point,Point2Plane};
	enum REJECT_TYPE {Trimmed_Ratio,Trimmed_Distance};
	enum TRANS_TYPE {RIGID,SHIFT};
	class Point;
	class Correspondence;
	class ICPResult;

	class Point
	{
	public:
		Point() {
			this->pos[0] = 0.0;
			this->pos[1] = 0.0;
			this->pos[2] = 0.0;
		};
		Point(double x, double y, double z) {
			this->pos[0] = x;
			this->pos[1] = y;
			this->pos[2] = z;
		};
		Point(const Point& p1) {
			this->pos[0] = p1.pos[0];
			this->pos[1] = p1.pos[1];
			this->pos[2] = p1.pos[2];

		}
		inline Point(Point& rhs)
		{
			this->pos[0] = rhs.pos[0];
			this->pos[1] = rhs.pos[1];
			this->pos[2] = rhs.pos[2];
		}

		inline Point operator+(const Point& rhs)
		{
			Point r;
			r.pos[0] = pos[0] + rhs.pos[0];
			r.pos[1] = pos[1] + rhs.pos[1];
			r.pos[2] = pos[2] + rhs.pos[2];
			return r;
		}
		inline Point operator-(const Point& rhs)
		{
			Point r;
			r.pos[0] = pos[0] - rhs.pos[0];
			r.pos[1] = pos[1] - rhs.pos[1];
			r.pos[2] = pos[2] - rhs.pos[2];
			return r;
		}
		inline Point& operator=(const Point& rhs)
		{
			this->pos[0] = rhs.pos[0];
			this->pos[1] = rhs.pos[1];
			this->pos[2] = rhs.pos[2];

			return *this;
		}

		double pos[3];
	};

	class Correspondence
	{
	public:
		Correspondence() {
			this->d = -1;
			this->s = -1;
		};
		Correspondence(int id_s, int id_d) {
			this->d = id_d;
			this->s = id_s;
		};
		int s;
		int d;
	};
	
	class ICPResult
	{
	public:
		ICPResult();
		std::vector<Correspondence> correspondence_set;
		double fitness;
		double inlier_rmse;
		double rmse;
		double rotationMatrix[9];
		double translation[3];
	};
	
	class ICP_Para
	{
	public:
		ICP_Para();
		double sample_ratio;
		double trimmed_ratio;
		double rmse_threshold;
		double max_distance;
		double init_rotation[9];
		double init_translation[3];
		int max_iterations;
		ICP_TYPE icp_type;
		TRANS_TYPE trans_type;
		REJECT_TYPE reject_type;
	};

	/*
	Compute the correspondence based on current transformation and RMSE error
	*/
	ICPResult GetRegistrationResultAndCorrespondences(std::vector<gs::Point*>& dynamicPointCloud,
		std::vector<gs::Point*>& staticPointCloud,
		kdtree* tree,
		std::unordered_map<kdnode*, int>& kd2id,
		double sample_ratio,
		double max_distance, 
		ICPResult& result);
	/*
	Copmute the transformation matrix based on correspondence. Point-to-Point objective function
	*/
	void ComputeTransformation_Point2Point(std::vector<Point*>& dynamicPointCloud,
		std::vector<Point*>& staticPointCloud,
		std::vector<Correspondence>& corrs,
		double* rotationMatrix,
		double* translation);

	/*
	Copmute the transformation matrix based on correspondence. Point-to-Point objective function
	*/
	void ComputeTransformation_Point2Point_SHIFT(std::vector<Point*>& dynamicPointCloud,
		std::vector<Point*>& staticPointCloud,
		std::vector<Correspondence>& corrs,
		double* translation);
	/*
	Reject the outlier corespondence
	*/
	void RejectCorrespondence(std::vector<Point*>& dynamicPointCloud,
		std::vector<Point*>& staticPointCloud,
		std::vector<Correspondence>& corrs,
		REJECT_TYPE reject_type,
		double trimmed_ratio);


	/*
	void icp
		transforms the point cloud 'dynamicPointCloud' such that it best matches 'staticPointCloud'

	@param std::vector<Point*> dynamicPointCloud : point cloud to be rotated and translated to match 'staticPointCloud'
	@param std::vector<Point*> staticPointCloud : reference point cloud.
	*/
	void icp(std::vector<Point*> &dynamicPointCloud, 
		std::vector<Point*> &staticPointCloud, 
		ICP_Para icp_para,
		ICPResult& result);

	inline void computeBBOX(std::vector<Point*>& cloud, gs::Point* mean,double* bbox) {
		bbox[0] = cloud[0]->pos[0];
		bbox[1] = cloud[0]->pos[0];
		bbox[2] = cloud[0]->pos[1];
		bbox[3] = cloud[0]->pos[1];
		bbox[4] = cloud[0]->pos[2];
		bbox[5] = cloud[0]->pos[2];
		for (int i = 0; i < cloud.size(); ++i) {
			bbox[0] = (cloud[i]->pos[0] < bbox[0]) ? cloud[i]->pos[0] : bbox[0];
			bbox[1] = (cloud[i]->pos[0] > bbox[1]) ? cloud[i]->pos[0] : bbox[1];
			bbox[2] = (cloud[i]->pos[1] < bbox[2]) ? cloud[i]->pos[1] : bbox[2];
			bbox[3] = (cloud[i]->pos[1] > bbox[3]) ? cloud[i]->pos[1] : bbox[3];
			bbox[4] = (cloud[i]->pos[2] < bbox[4]) ? cloud[i]->pos[2] : bbox[4];
			bbox[5] = (cloud[i]->pos[2] > bbox[5]) ? cloud[i]->pos[2] : bbox[5];
		}
		mean->pos[0] = (bbox[0] + bbox[1]) / 2.0f;
		mean->pos[1] = (bbox[2] + bbox[3]) / 2.0f;
		mean->pos[2] = (bbox[4] + bbox[5]) / 2.0f;
	}

	inline void computeCloudMean(std::vector<Point*> &cloud, gs::Point* mean)
	{
		mean->pos[0] = 0.0;
		mean->pos[1] = 0.0;
		mean->pos[2] = 0.0;
		for (int i = 0; i < cloud.size(); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				mean->pos[j] += cloud[i]->pos[j];
			}
		}
		mean->pos[0] = mean->pos[0] / (double)cloud.size();
		mean->pos[1] = mean->pos[1] / (double)cloud.size();
		mean->pos[2] = mean->pos[2] / (double)cloud.size();
	}

	inline void clearTranslation(double* translation)
	{
		translation[0] = 0.0;
		translation[1] = 0.0;
		translation[2] = 0.0;
	}

	inline void clearRotation(double* rotation)
	{
		rotation[0] = 1.0;
		rotation[1] = 0.0;
		rotation[2] = 0.0;

		rotation[3] = 0.0;
		rotation[4] = 1.0;
		rotation[5] = 0.0;

		rotation[6] = 0.0;
		rotation[7] = 0.0;
		rotation[8] = 1.0;
	}

	inline void clearMatrix(double* mat)
	{
		mat[0] = 0.0;
		mat[1] = 0.0;
		mat[2] = 0.0;

		mat[3] = 0.0;
		mat[4] = 0.0;
		mat[5] = 0.0;

		mat[6] = 0.0;
		mat[7] = 0.0;
		mat[8] = 0.0;
	}

	inline void initDiagonal(double* mat) {
		mat[0] = 1.0;
		mat[1] = 0.0;
		mat[2] = 0.0;

		mat[3] = 0.0;
		mat[4] = 1.0;
		mat[5] = 0.0;

		mat[6] = 0.0;
		mat[7] = 0.0;
		mat[8] = 1.0;
	}

	inline void rotate(gs::Point* p, double* rotationMatrix, gs::Point* result)
	{
		result->pos[0] = p->pos[0] * rotationMatrix[0] + p->pos[1] * rotationMatrix[1] + p->pos[2] * rotationMatrix[2];
		result->pos[1] = p->pos[0] * rotationMatrix[3] + p->pos[1] * rotationMatrix[4] + p->pos[2] * rotationMatrix[5];
		result->pos[2] = p->pos[0] * rotationMatrix[6] + p->pos[1] * rotationMatrix[7] + p->pos[2] * rotationMatrix[8];
	}

	inline void rotate_mat(double* p, double* rotationMatrix, double* result)
	{
		result[0] = p[0] * rotationMatrix[0] + p[1] * rotationMatrix[1] + p[2] * rotationMatrix[2];
		result[1] = p[0] * rotationMatrix[3] + p[1] * rotationMatrix[4] + p[2] * rotationMatrix[5];
		result[2] = p[0] * rotationMatrix[6] + p[1] * rotationMatrix[7] + p[2] * rotationMatrix[8];
	}

	inline void translate(gs::Point* p, double* translationVector, gs::Point* result)
	{
		result->pos[0] = p->pos[0] + translationVector[0];
		result->pos[1] = p->pos[1] + translationVector[1];
		result->pos[2] = p->pos[2] + translationVector[2];
	}

	inline double determinant_2by2(double a11, double a12, double a21, double a22) {
		return a11 * a22 - a12 * a21;

	};
	
	inline double determinant_3by3(double** a) {
		return a[0][0] * determinant_2by2(a[1][1], a[1][2], a[2][1], a[2][2])
			- a[0][1] * determinant_2by2(a[1][0], a[1][2], a[2][0], a[2][2])
			+ a[0][2] * determinant_2by2(a[1][0], a[1][1], a[2][0], a[2][1]);
	}

	inline void outerProduct(gs::Point* a, gs::Point* b, double* mat)
	{
		mat[0] = a->pos[0] * b->pos[0];
		mat[1] = a->pos[0] * b->pos[1];
		mat[2] = a->pos[0] * b->pos[2];

		mat[3] = a->pos[1] * b->pos[0];
		mat[4] = a->pos[1] * b->pos[1];
		mat[5] = a->pos[1] * b->pos[2];

		mat[6] = a->pos[2] * b->pos[0];
		mat[7] = a->pos[2] * b->pos[1];
		mat[8] = a->pos[2] * b->pos[2];
	}

	inline void matrixMult(double* a, double* b, double* result)
	{
		result[0] = a[0] * b[0] + a[1] * b[3] + a[2] * b[6];
		result[1] = a[0] * b[1] + a[1] * b[4] + a[2] * b[7];
		result[2] = a[0] * b[2] + a[1] * b[5] + a[2] * b[8];

		result[3] = a[3] * b[0] + a[4] * b[3] + a[5] * b[6];
		result[4] = a[3] * b[1] + a[4] * b[4] + a[5] * b[7];
		result[5] = a[3] * b[2] + a[4] * b[5] + a[5] * b[8];

		result[6] = a[6] * b[0] + a[7] * b[3] + a[8] * b[6];
		result[7] = a[6] * b[1] + a[7] * b[4] + a[8] * b[7];
		result[8] = a[6] * b[2] + a[7] * b[5] + a[8] * b[8];
	}

	inline void transpose(double* a)
	{
		double temp;

		temp = a[1];
		a[1] = a[3];
		a[3] = temp;

		temp = a[2];
		a[2] = a[6];
		a[6] = temp;

		temp = a[5];
		a[5] = a[7];
		a[7] = temp;
	}

	inline void addMatrix(double* a, double* b, double* result)
	{
		result[0] = a[0] + b[0];
		result[1] = a[1] + b[1];
		result[2] = a[2] + b[2];

		result[3] = a[3] + b[3];
		result[4] = a[4] + b[4];
		result[5] = a[5] + b[5];

		result[6] = a[6] + b[6];
		result[7] = a[7] + b[7];
		result[8] = a[8] + b[8];
	}

	inline double square_error(Point* ps, Point* pd)
	{
		double err = pow(ps->pos[0] - pd->pos[0], double(2.0));
		err += pow(ps->pos[1] - pd->pos[1], double(2.0));
		err += pow(ps->pos[2] - pd->pos[2], double(2.0));
		return err;
	}

	inline void copyMatToUV(double* mat, double** result)
	{
		result[0][0] = mat[0];
		result[0][1] = mat[1];
		result[0][2] = mat[2];

		result[1][0] = mat[3];
		result[1][1] = mat[4];
		result[1][2] = mat[5];

		result[2][0] = mat[6];
		result[2][1] = mat[7];
		result[2][2] = mat[8];
	}

	inline void copyUVtoMat(double** mat, double* result)
	{
		result[0] = mat[0][0];
		result[1] = mat[0][1];
		result[2] = mat[0][2];

		result[3] = mat[1][0];
		result[4] = mat[1][1];
		result[5] = mat[1][2];

		result[6] = mat[2][0];
		result[7] = mat[2][1];
		result[8] = mat[2][2];
	}
}