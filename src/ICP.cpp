#include "ICP.h"
#include <algorithm>
gs::ICPResult::ICPResult() {
	std::vector<Correspondence> empty_set;
	this->correspondence_set = empty_set;
	this->rmse = -9999;
	this->inlier_rmse = -9999;
	this->fitness = -9999;
	clearTranslation(this->translation);
	clearRotation(this->rotationMatrix);
};

gs::ICP_Para::ICP_Para() {
	this->sample_ratio = 0.3;
	this->trimmed_ratio = 0.5;
	this->rmse_threshold = 0.001;
	this->max_iterations = 20;
	this->max_distance = 30;
	clearRotation(this->init_rotation);
	clearTranslation(this->init_translation);
	this->icp_type = Point2Point;
	this->trans_type = RIGID;
	this->reject_type = Trimmed_Distance;

};

void cout_matrix(double* mat) {
	std::cout << mat[0] << "," << mat[1] << "," << mat[2] << "," <<
		mat[3] << "," << mat[4] << "," << mat[5] << "," <<
		mat[6] << "," << mat[7] << "," << mat[8] << std::endl;
}

void cout_matrix3(double* mat) {
	std::cout << mat[0] << "," << mat[1] << "," << mat[2] <<std::endl;
}

void cout_matrix2d(double** mat) {
	std::cout << mat[0][0] << "," << mat[0][1] << "," << mat[0][2] << "," <<
		mat[1][0] << "," << mat[1][1] << "," << mat[1][2] << "," <<
		mat[2][0] << "," << mat[2][1] << "," << mat[2][2] << std::endl;
}

void check_corr(std::vector<gs::Correspondence> corr_set) {
	int cnt = 0;
	for (int i = 0; i < corr_set.size(); ++i) {
		if (corr_set[i].d != corr_set[i].s) {
			cnt++;
		}
	}
	std::cout << cnt << std::endl;
}

gs::ICPResult gs::GetRegistrationResultAndCorrespondences(std::vector<gs::Point*>& dynamicPointCloud,
	std::vector<gs::Point*>& staticPointCloud,
	kdtree* tree,
	std::unordered_map<kdnode*, int>& kd2id,
	double sample_ratio,
	double max_distance,
	gs::ICPResult& result)
{
	//double sample_ratio = 1.0;
	double cost = 0;
	std::vector<gs::Correspondence> corrs_set;
	int numSamples = int(sample_ratio * dynamicPointCloud.size());

#pragma omp parallel
	{
		double cost_private = 0;
		double distance = 0;
		std::vector<gs::Correspondence> corrs_set_private;
		gs::Point p, x;
		int s_id = 0;
		kdnode* kdnode_ = 0;
#pragma omp for nowait
		for (int i = 0; i < dynamicPointCloud.size(); i++)
		{
			//std::cout << i << std::endl;
			int randSample = std::rand() % dynamicPointCloud.size();
			randSample = i;
			p = *dynamicPointCloud[randSample];
			kdres* kdset1 = kd_nearest3f(tree, p.pos[0], p.pos[1], p.pos[2]);
			kd_res_node(kdset1, &kdnode_);
			s_id = kd2id[kdnode_];
			x = *staticPointCloud[s_id];
			distance= std::sqrt(square_error(&x, &p));
			if (distance < max_distance) {
				cost_private += pow(distance, 2.0);
				corrs_set_private.push_back(gs::Correspondence(s_id, randSample));
			}
			kd_res_free(kdset1);
		}
#pragma omp critical
		{
			for (int i = 0; i < corrs_set_private.size(); ++i) {
				corrs_set.push_back(corrs_set_private[i]);
			}
			cost += cost_private;
		}
	}

	result.fitness = (double)corrs_set.size() / (double)numSamples;
	result.rmse = std::sqrt(cost / numSamples);
	result.inlier_rmse = std::sqrt(cost / numSamples);
	result.correspondence_set = corrs_set;
	return result;

}


/*
Based on https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
and Eigen::umeyama fucntion
*/
void gs::ComputeTransformation_Point2Point(std::vector<Point*>& dynamicPointCloud,
	std::vector<Point*>& staticPointCloud,
	std::vector<Correspondence>& corrs,
	double* rotationMatrix,
	double* translation){
	if (corrs.size() < 3) { // return identiy matrix
		clearRotation(rotationMatrix);
		clearTranslation(translation);
		return;
	}
	else {
		Point dynamicMid,d_raw,s_raw,d_delta,s_delta;
		Point staticMid(0, 0, 0);
		computeCloudMean(dynamicPointCloud, &dynamicMid);
		for (int i = 0; i < corrs.size(); ++i) {
			staticMid =staticMid + *staticPointCloud[corrs[i].s];
		}
		staticMid.pos[0] = staticMid.pos[0] / corrs.size();
		staticMid.pos[1] = staticMid.pos[1] / corrs.size();
		staticMid.pos[2] = staticMid.pos[2] / corrs.size();

		double w[9];
		double U[9]; /// Covariance matrix of Dynamic and Static data.
		double sigma[3];
		double V[9];
		double Diagonal[9];

		double** uSvd = new double* [3];
		double** vSvd = new double* [3];
		uSvd[0] = new double[3];
		uSvd[1] = new double[3];
		uSvd[2] = new double[3];

		vSvd[0] = new double[3];
		vSvd[1] = new double[3];
		vSvd[2] = new double[3];
		clearMatrix(U);
		clearMatrix(V);
		clearMatrix(w);
		for (int i = 0; i < corrs.size(); i++)
		{
			d_raw = *dynamicPointCloud[corrs[i].d];
			s_raw = *staticPointCloud[corrs[i].s];
			d_delta = d_raw - dynamicMid;
			s_delta = s_raw - staticMid;

			//outerProduct(&d_delta, &s_delta, w);
			outerProduct(&s_delta, &d_delta, w);
			addMatrix(w, U, U);
		}
		//U[0] /= corrs.size();
		//U[1] /= corrs.size();
		//U[2] /= corrs.size();
		//U[3] /= corrs.size();
		//U[4] /= corrs.size();
		//U[5] /= corrs.size();
		//U[6] /= corrs.size();
		//U[7] /= corrs.size();
		//U[8] /= corrs.size();
		copyMatToUV(U, uSvd);
		dsvd(uSvd, 3, 3, sigma, vSvd);
		copyUVtoMat(uSvd, U);
		copyUVtoMat(vSvd, V);

		/// Compute Rotation matrix
		initDiagonal(Diagonal);
		if (determinant_3by3(uSvd) * determinant_3by3(vSvd) < 0) {
			Diagonal[8] = -1;
		}
		//transpose(U);
		transpose(V);
		double tmp[9];
		clearMatrix(tmp);
		//matrixMult(V, Diagonal,tmp);
		//matrixMult(tmp, U, rotationMatrix);
		matrixMult(U, Diagonal, tmp);
		matrixMult(tmp, V, rotationMatrix);

		gs::Point t(0.0, 0.0, 0.0);
		rotate(&dynamicMid, rotationMatrix, &t);
		translation[0] = staticMid.pos[0] - t.pos[0];
		translation[1] = staticMid.pos[1] - t.pos[1];
		translation[2] = staticMid.pos[2] - t.pos[2];
		delete[] uSvd[0];
		delete[] uSvd[1];
		delete[] uSvd[2];
		delete[] uSvd;

		delete[] vSvd[0];
		delete[] vSvd[1];
		delete[] vSvd[2];
		delete[] vSvd;
	}
}

void gs::ComputeTransformation_Point2Point_SHIFT(std::vector<Point*>& dynamicPointCloud,
	std::vector<Point*>& staticPointCloud,
	std::vector<Correspondence>& corrs,
	double* translation) {
	if (corrs.size() < 1) { // return identiy matrix
		clearTranslation(translation);
		return;
	}
	else {
		Point dynamicMid(0, 0, 0);
		Point staticMid(0,0,0);
		for (int i = 0; i < corrs.size(); i++)
		{
			dynamicMid= dynamicMid + *dynamicPointCloud[corrs[i].d];
			staticMid = staticMid + *staticPointCloud[corrs[i].s];
		}
		dynamicMid.pos[0] /= (double)corrs.size();
		dynamicMid.pos[1] /= (double)corrs.size();
		dynamicMid.pos[2] /= (double)corrs.size();
		staticMid.pos[0] /= (double)corrs.size();
		staticMid.pos[1] /= (double)corrs.size();
		staticMid.pos[2] /= (double)corrs.size();

		translation[0] = staticMid.pos[0] - dynamicMid.pos[0];
		translation[1] = staticMid.pos[1] - dynamicMid.pos[1];
		translation[2] = staticMid.pos[2] - dynamicMid.pos[2];
	}
}

bool cmp_corr(std::pair<float, int>& a, std::pair<float, int>& b) {
	return a.first < b.first;
}

void gs::RejectCorrespondence(std::vector<Point*>& dynamicPointCloud,
	std::vector<Point*>& staticPointCloud,
	std::vector<Correspondence>& corrs,
	REJECT_TYPE reject_type,
	double trimmed_ratio) {
	if (reject_type == Trimmed_Distance) {
		return;
	}
	else if(reject_type == Trimmed_Ratio) {
		int num_trimmed = (int)corrs.size() * trimmed_ratio;
		std::vector<Correspondence> corr_tmp;
		std::vector<std::pair<float, int>> corr_dis_index;
		float dis;
		Point d, s;
		for (int i = 0; i < corrs.size(); ++i) {
			d = *dynamicPointCloud[corrs[i].d];
			s = *staticPointCloud[corrs[i].s];
			dis = std::sqrt(pow(d.pos[0] - s.pos[0], 2.0) + 
							pow(d.pos[1] - s.pos[1], 2.0) + 
							pow(d.pos[2] - s.pos[2], 2.0));
			corr_dis_index.push_back(std::make_pair(dis, i));
		}
		std::sort(corr_dis_index.begin(), corr_dis_index.end(), cmp_corr);
		for (int i = 0; i < num_trimmed; ++i) {
			corr_tmp.push_back(corrs[corr_dis_index[i].second]);
		}
		corrs = corr_tmp;

	}
}

void gs::icp(std::vector<Point*> &dynamicPointCloud, 
	std::vector<Point*> &staticPointCloud,
	gs::ICP_Para icp_para,
	gs::ICPResult& result)
{
	
	double sample_ratio = icp_para.sample_ratio;
	double trimmed_ratio = icp_para.trimmed_ratio;
	int maxIterations = icp_para.max_iterations;
	int numRandomSamples = int(dynamicPointCloud.size()*sample_ratio);
	const double eps = icp_para.rmse_threshold;
	gs::Point dynamicMid(0.0, 0.0, 0.0);
	gs::Point staticMid(0.0, 0.0, 0.0);
	Point p;
	double init_rotationMatrix[9];
	double init_translation[3];
	double rotationMatrix_tmp[9];
	double translation_tmp[3];
	double tmp_vec3[3];

	std::copy(&icp_para.init_rotation[0], &icp_para.init_rotation[9], &init_rotationMatrix[0]);
	std::copy(&icp_para.init_translation[0], &icp_para.init_translation[3], &init_translation[0]);
	std::copy(&icp_para.init_rotation[0], &icp_para.init_rotation[9], &result.rotationMatrix[0]);
	std::copy(&icp_para.init_translation[0], &icp_para.init_translation[3], &result.translation[0]);
	clearTranslation(translation_tmp);
	clearRotation(rotationMatrix_tmp);
	
	// construct a kd-tree index:
	kdtree* kdtree_= kd_create(3);
	kdnode* kdnode_=0;
	std::unordered_map<kdnode*, int> kd2id;
	//computeCloudMean(staticPointCloud, &staticMid);

	// Apply Global Shit
	//double GLOBAL_TRANS[3];
	//gs::Point static_bbox_mid(0.0, 0.0, 0.0);
	//double bbox[6];
	//computeBBOX(staticPointCloud, &static_bbox_mid,bbox);
	//GLOBAL_TRANS[0] = -static_bbox_mid.pos[0];
	//GLOBAL_TRANS[1] = -static_bbox_mid.pos[1];
	//GLOBAL_TRANS[2] = -static_bbox_mid.pos[2];
	//for (int i = 0; i < dynamicPointCloud.size(); i++)
	//{
	//	translate(dynamicPointCloud[i], GLOBAL_TRANS, dynamicPointCloud[i]);
	//}
	//for (int i = 0; i < staticPointCloud.size(); i++)
	//{
	//	translate(staticPointCloud[i], GLOBAL_TRANS, staticPointCloud[i]);
	//}

	// create the kd tree
	for (int i = 0; i < staticPointCloud.size(); i++)
	{
		kd_insert3(kdtree_, staticPointCloud[i]->pos[0], staticPointCloud[i]->pos[1], staticPointCloud[i]->pos[2], 0, &kdnode_);
		kd2id.insert(std::make_pair(kdnode_, i));
	}

	gs::GetRegistrationResultAndCorrespondences(dynamicPointCloud, staticPointCloud, kdtree_, kd2id, sample_ratio,icp_para.max_distance,result);
	for (int iter = 0; iter < maxIterations; iter++)
	{	
		std::ostringstream ss;
		ss << "Iteration:" << iter << " RMSE Cost:" << result.rmse<<"\n";
		std::string s(ss.str());
		printf(s.c_str());

		if (icp_para.trans_type == RIGID) {
			// Reject outlier correspondence
			gs::RejectCorrespondence(dynamicPointCloud, staticPointCloud, result.correspondence_set, icp_para.reject_type, icp_para.trimmed_ratio);
			/// Compute the transformation matrix
			gs::ComputeTransformation_Point2Point(dynamicPointCloud, staticPointCloud, result.correspondence_set, rotationMatrix_tmp, translation_tmp);
			/// Update current transformation matrix
			matrixMult(rotationMatrix_tmp, result.rotationMatrix, init_rotationMatrix);
			rotate_mat(result.translation, rotationMatrix_tmp, tmp_vec3);
			init_translation[0] = tmp_vec3[0] + translation_tmp[0];
			init_translation[1] = tmp_vec3[1] + translation_tmp[1];
			init_translation[2] = tmp_vec3[2] + translation_tmp[2];
			std::copy(&init_rotationMatrix[0], &init_rotationMatrix[9], &result.rotationMatrix[0]);
			std::copy(&init_translation[0], &init_translation[3], &result.translation[0]);
			//update the point cloud
			for (int i = 0; i < dynamicPointCloud.size(); i++)
			{
				rotate(dynamicPointCloud[i], rotationMatrix_tmp, &p);
				translate(&p, translation_tmp, dynamicPointCloud[i]);
			}
		}
		else{
			// Reject outlier correspondence
			gs::RejectCorrespondence(dynamicPointCloud, staticPointCloud, result.correspondence_set, icp_para.reject_type, icp_para.trimmed_ratio);
			gs::ComputeTransformation_Point2Point_SHIFT(dynamicPointCloud, staticPointCloud, result.correspondence_set, translation_tmp);
			/// Update current transformation matrix
			init_translation[0] = result.translation[0] + translation_tmp[0];
			init_translation[1] = result.translation[1] + translation_tmp[1];
			init_translation[2] = result.translation[2] + translation_tmp[2];
			std::copy(&init_translation[0], &init_translation[3], &result.translation[0]);
			//update the point cloud
			for (int i = 0; i < dynamicPointCloud.size(); i++)
			{
				translate(dynamicPointCloud[i], translation_tmp, dynamicPointCloud[i]);
			}
		}

		double pre_rmse = result.rmse;
		gs::GetRegistrationResultAndCorrespondences(dynamicPointCloud, staticPointCloud, kdtree_, kd2id, sample_ratio,icp_para.max_distance,result);
		if (std::abs(result.rmse - pre_rmse) < icp_para.rmse_threshold) {
			break;
		}
	}

	/// Apply reverse Global transform
	//GLOBAL_TRANS[0] = -GLOBAL_TRANS[0];
	//GLOBAL_TRANS[1] = -GLOBAL_TRANS[1];
	//GLOBAL_TRANS[2] = -GLOBAL_TRANS[2];
	//for (int i = 0; i < dynamicPointCloud.size(); i++)
	//{
	//	translate(dynamicPointCloud[i], GLOBAL_TRANS, dynamicPointCloud[i]);
	//}

	kd_free(kdtree_);
}