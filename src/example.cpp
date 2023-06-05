#include "ICP.h"
#include <fstream>
#include <iomanip>

using namespace gs;

/*
Create a test box of points.
*/
void createPoints(std::vector<Point*>& points)
{
	for (double i = 0; i < 450; ++i) {
		for (double j = 0; j < 450; ++j) {
			points.push_back(new Point(i/10, j/10, -0.5f));
		}
	}
}

/*
Apply an afine transofrm to a point cloud
*/
void applyAffineTransform(std::vector<Point*>& points, float* rotationMatrix, float* translation)
{
	Point pRot;
	for (int i = 0; i < points.size(); i++)
	{
		pRot.pos[0] = rotationMatrix[0] * points[i]->pos[0] + rotationMatrix[1] * points[i]->pos[1] + rotationMatrix[2] * points[i]->pos[2] + translation[0];
		pRot.pos[1] = rotationMatrix[3] * points[i]->pos[0] + rotationMatrix[4] * points[i]->pos[1] + rotationMatrix[5] * points[i]->pos[2] + translation[1];
		pRot.pos[2] = rotationMatrix[6] * points[i]->pos[0] + rotationMatrix[7] * points[i]->pos[1] + rotationMatrix[8] * points[i]->pos[2] + translation[2];

		*points[i] = pRot;
	}
}

/*
Load point cloud data from txt file
*/
void load_txt(std::vector<Point*>& points, std::string file_path) {
	std::ifstream infile(file_path);
	float x, y, z;
	while (infile >> x >> y >> z) {
		points.push_back(new Point(x, y, z));
	}
	infile.close();
}

void write_txt(std::vector<Point*>& points, std::string file_path) {
	std::ofstream outfile(file_path);
	//float x, y, z;
	for (int i = 0; i < points.size(); ++i) {
		outfile << std::fixed << std::setw(11) << std::setprecision(11) << points[i]->pos[0] << " " <<
			std::fixed << std::setw(11) << std::setprecision(11) << points[i]->pos[1] << " " <<
			std::fixed << std::setw(11) << std::setprecision(11) << points[i]->pos[2] << " " << "\n";

		//outfile << points[i]->pos[0] <<" "<< points[i]->pos[1]<<" " << points[i]->pos[2]<<"\n";
	}
	outfile.close();
}

void write_trans(gs::ICPResult& result, std::string file_path) {
	std::ofstream outfile(file_path);
	outfile << std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[0] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[1] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[2] << " " << 
		std::fixed << std::setw(11) << std::setprecision(11) << result.translation[0] << " " <<"\n"

		<< std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[3] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[4] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[5] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.translation[1] << " " << "\n"
		<< std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[6] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[7] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.rotationMatrix[8] << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << result.translation[2] << " " << "\n"
		<< std::fixed << std::setw(11) << std::setprecision(11) << 0 << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << 0 << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << 0 << " " <<
		std::fixed << std::setw(11) << std::setprecision(11) << 1 << " " << "\n";
	outfile.close();
}

/*
ICP is used to minimize the distance between 2 point clouds.

This example computes a point cloud of a unit box (Static point cloud)
and a second unit box with a linear transform applied (dynamic point cloud).
ICP is used to transform the dynamic point cloud to best match the static point cloud.
*/


void icpExample()
{
	//create a static box point cloud used as a reference.
	std::vector<Point*> staticPointCloud;
	createPoints(staticPointCloud);

	//create a dynamic box point cloud.
	//this point cloud is transformed to match the static point cloud.
	std::vector<Point*> dynamicPointCloud;
	createPoints(dynamicPointCloud);

	//apply an artitrary rotation and translation to the dynamic point cloud to misalign the point cloud.
	float rotation[] = { 1.0f, 0.0f, 0.0f,	0.0f, 0.70710678f, -0.70710678f,	0.0f, 0.70710678f, 0.70710678f };
	float translation[] = { 100.0f, 0.0f, 1.0f };
	applyAffineTransform(dynamicPointCloud, rotation, translation);

	//printf("Static point Cloud: \n");
	//for (int i = 0; i < staticPointCloud.size(); i++)
	//{
	//	printf("%0.2f, %0.2f, %0.2f \n", staticPointCloud[i]->pos[0], staticPointCloud[i]->pos[1], staticPointCloud[i]->pos[2]);
	//}
	//printf("\n");

	//printf("Dynamic point Cloud: \n");
	//for (int i = 0; i < dynamicPointCloud.size(); i++)
	//{
	//	printf("%0.2f, %0.2f, %0.2f \n", dynamicPointCloud[i]->pos[0], dynamicPointCloud[i]->pos[1], dynamicPointCloud[i]->pos[2]);
	//}
	//printf("\n");

	//use iterative closest point to transform the dynamic point cloud to best align the static point cloud.
	ICP_Para icp_para;
	ICPResult result;
	icp(dynamicPointCloud, staticPointCloud,icp_para,result);

	//printf("Dynamic point Cloud Transformed: \n");
	//for (int i = 0; i < dynamicPointCloud.size(); i++)
	//{
	//	printf("%0.2f, %0.2f, %0.2f \n", dynamicPointCloud[i]->pos[0], dynamicPointCloud[i]->pos[1], dynamicPointCloud[i]->pos[2]);
	//}
	//printf("\n");

	float alignmentError = 0.0f;
	for (int i = 0; i < dynamicPointCloud.size(); i++)
	{
		alignmentError += pow(dynamicPointCloud[i]->pos[0] - staticPointCloud[i]->pos[0], 2.0f);
		alignmentError += pow(dynamicPointCloud[i]->pos[1] - staticPointCloud[i]->pos[1], 2.0f);
		alignmentError += pow(dynamicPointCloud[i]->pos[2] - staticPointCloud[i]->pos[2], 2.0f);
	}

	alignmentError /= (float)dynamicPointCloud.size();

	printf("Alignment Error: %0.5f \n", alignmentError);
}

//void main()
//{
//	/// ICP test
//	std::string pts1_path = "J:\\xuningli\\3DDataRegistration\\REG\\icp_2010\\data\\als.txt";
//	std::string pts2_path = "J:\\xuningli\\3DDataRegistration\\REG\\icp_2010\\data\\uav.txt";
//	std::vector<Point*> staticPointCloud;
//	std::vector<Point*> dynamicPointCloud;
//	load_txt(staticPointCloud, pts1_path);
//	load_txt(dynamicPointCloud, pts2_path);
//	ICP_Para icp_para;
//	icp_para.sample_ratio = 1;
//	icp_para.max_distance = 30;
//	icp_para.trans_type = RIGID;
//	ICPResult result;
//	icp(dynamicPointCloud, staticPointCloud,icp_para,result);
//	write_txt(dynamicPointCloud, "J:\\xuningli\\3DDataRegistration\\REG\\icp_2010\\data\\uav_reg.txt");
//	write_trans(result, "J:\\xuningli\\3DDataRegistration\\REG\\icp_2010\\data\\est_trans.txt");
//	
//	//icpExample();
//
//
//
//	system("pause");
//}