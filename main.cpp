////gdal
#include "gdal_priv.h"
#include "gdalwarper.h"
#include "ogr_core.h"
#include "ogr_api.h"
#include "cpl_conv.h" // for CPLMalloc()
#include <ogrsf_frmts.h>

#include "iostream"
#include <algorithm>
#include <sstream>
#include <iterator>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ICP.h>
#include <strutil.h>
#include <math.h>
#include <Eigen/Geometry>
#include <numeric>
#include <argparse.hpp>

using namespace std;
namespace fs = std::experimental::filesystem;
using namespace std::chrono;
#define PI 3.14159265;
class TILE_INFO {
public:
    TILE_INFO() {
        this->utm_bbox[0] = 0;
        this->utm_bbox[1] = 0;
        this->utm_bbox[2] = 0;
        this->utm_bbox[3] = 0;
        this->ref_var = 0;
        this->src_var = 0;
        this->tile_id = 0;
        this->src_lambda1 = 0;
        this->src_lambda2 = 0;
        this->src_R=0;
        this->ref_lambda1 = 0;
        this->ref_lambda2 = 0;
        this->ref_R = 0;
    }
    double utm_bbox[4];
    double ref_var;
    double src_var;
    double ref_valid;
    double src_valid;
    double src_lambda1;
    double src_lambda2;
    double src_R;
    double ref_lambda1;
    double ref_lambda2;
    double ref_R;
    int tile_id;
    double rotation[9];
    double translation[3];
};

class CORR {
public:
    CORR() {};
    CORR(gs::Point src,gs::Point ref,double weight) {
        this->src = src;
        this->ref = ref;
        this->w = weight;
    }
    CORR(gs::Point src, gs::Point ref,std::vector<float> src_norm, std::vector<float> ref_norm, double weight) {
        this->src = src;
        this->ref = ref;
        this->w = weight;
        this->src_norm = src_norm;
        this->ref_norm = ref_norm;
    }
    gs::Point src;
    gs::Point ref;
    vector<float> src_norm;
    vector<float> ref_norm;
    double w;
};

struct Parameters {
    std::string ref_path_, src_path_;
    int ref_width_, ref_height_, src_width_, src_height_;
    double GLOBAL_OFFSET_X_m_;
    double GLOBAL_OFFSET_Y_m_;
    double resolution_;
    double ref_utm_bbox_[4];
    double src_utm_bbox_[4];
    double aoi_utm_bbox_[4];
    int aoi_ref_bbox_pixel_[4];
    int aoi_src_bbox_pixel_[4];
    int num_tile_x_, num_tile_y_;
    double tilesize_m_;
    std::vector<TILE_INFO> tile_infos_all_, tile_infos_filter_, tile_infos_icp_;
    double valid_ratio_threshold_, var_min_;
    int verbose_;
    // for General 
    bool plane_or_not_;
    bool multi_or_not_;
    bool brute_force_search_or_not_;
    double xmin_m_;
    double xmax_m_;
    double ymin_m_;
    double ymax_m_;
    double rough_step_m_;
    int rough_step_min_resolution_;
    int rough_num_mins_;
    double fine_step_m_;
    // for rough align
    double valid_ratio_threshold_rough_;
    double rough_rotation_[9];
    double rough_translation_[3];
    vector<vector<double>> rough_translations_;
    double rough_icp_max_iter_;
    double rough_icp_rmse_threshold_;
    double trimmed_ratio_;
    double rough_icp_percentage_threshold_;
    double rough_rmse_;
    // for fine align
    double HARRIS_step_;
    double HARRIS_k_;
    double fine_translation_[3];
    int n_blocks_col_;
    int n_blocks_row_;
    int num_filter_tiles_;
    double fine_icp_max_iter_;
    double fine_icp_rmse_threshold_;
    double fine_icp_percentage_threshold_;
    double fine_search_half_pixels_;
    int fine_resolution_factor_;
    double fine_rmse_;
    double fine_rotation_[9];

    double final_rot_[9];
    double final_trans_[3];

    // FOR rough align
    int rough_num_pts_;
    int search_half_size_pixel_;
    std::vector<gs::Point> rough_src_pts_;
    std::vector<std::pair<int, int>> rough_src_index_; // in source image pixel coordinates
};

void parse_shp(string shp_path, vector<string>& tif_paths, vector<double>& tif_bboxs) {
    GDALDataset* shps;
    shps = (GDALDataset*)GDALOpenEx(shp_path.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    int a = shps->GetLayerCount();
    OGRLayer* l1 = shps->GetLayer(0);
    OGRFeature* poFeature;
    l1->ResetReading();
    while ((poFeature = l1->GetNextFeature()) != NULL)
    {
        int c = poFeature->GetFieldCount();
        OGRGeometry* shape = poFeature->GetGeometryRef();
        OGREnvelope psEnvelope;
        shape->getBoundary()->getEnvelope(&psEnvelope);
        auto&& field = (*poFeature)[0];
        string tif_path = field.GetAsString();
        tif_bboxs.push_back(psEnvelope.MinX);
        tif_bboxs.push_back(psEnvelope.MaxX);
        tif_bboxs.push_back(psEnvelope.MinY);
        tif_bboxs.push_back(psEnvelope.MaxY);
        tif_paths.push_back(tif_path);
    }
    GDALClose(shps);
}

void export_tfw(string path, double* transform) {
    std::ofstream out(path.c_str());
    out << std::fixed << std::setprecision(8) << transform[1] << endl;
    out << std::fixed << std::setprecision(8) << transform[2] << endl;
    out << std::fixed << std::setprecision(8) << transform[4] << endl;
    out << std::fixed << std::setprecision(8) << transform[5] << endl;
    out << std::fixed << std::setprecision(8) << transform[0] << endl;
    out << std::fixed << std::setprecision(8) << transform[3] << endl;
    out.close();

}

void get_aoi_bbox_pixels(double* aoi_utm_bbox, double* ref_bbox, double* src_bbox, int ref_width, int ref_height, int src_width, int src_height, int* aoi_ref_bbox, int* aoi_src_bbox) {
    aoi_ref_bbox[0] = floor((aoi_utm_bbox[0] - ref_bbox[0]+0.25) / 0.5);
    aoi_ref_bbox[1] = floor((aoi_utm_bbox[1] - ref_bbox[0]+0.25) / 0.5);
    aoi_ref_bbox[2] = floor((ref_bbox[3]-aoi_utm_bbox[2]+ 0.25) / 0.5);
    aoi_ref_bbox[3] = floor((ref_bbox[3]-aoi_utm_bbox[3]+0.25) / 0.5);
    aoi_src_bbox[0] = floor((aoi_utm_bbox[0] - src_bbox[0]+0.25) / 0.5);
    aoi_src_bbox[1] = floor((aoi_utm_bbox[1] - src_bbox[0]+0.25) / 0.5);
    aoi_src_bbox[2] = floor((src_bbox[3]-aoi_utm_bbox[2]+0.25) / 0.5);
    aoi_src_bbox[3] = floor((src_bbox[3]-aoi_utm_bbox[3]+0.25) / 0.5);
    aoi_ref_bbox[0] = max(aoi_ref_bbox[0], 0);
    aoi_ref_bbox[0] = min(aoi_ref_bbox[0], ref_width - 1);
    aoi_ref_bbox[1] = max(aoi_ref_bbox[1], 0);
    aoi_ref_bbox[1] = min(aoi_ref_bbox[1], ref_width - 1);
    aoi_ref_bbox[2] = min(aoi_ref_bbox[2], ref_height - 1);
    aoi_ref_bbox[2] = max(aoi_ref_bbox[2], 0);
    aoi_ref_bbox[3] = min(aoi_ref_bbox[3], ref_height - 1);
    aoi_ref_bbox[3] = max(aoi_ref_bbox[3], 0);
    aoi_src_bbox[0] = max(aoi_src_bbox[0], 0);
    aoi_src_bbox[0] = min(aoi_src_bbox[0], src_width - 1);
    aoi_src_bbox[1] = max(aoi_src_bbox[1], 0);
    aoi_src_bbox[1] = min(aoi_src_bbox[1], src_width - 1);
    aoi_src_bbox[2] = min(aoi_src_bbox[2], src_height - 1);
    aoi_src_bbox[2] = max(aoi_src_bbox[2], 0);
    aoi_src_bbox[3] = min(aoi_src_bbox[3], src_height - 1);
    aoi_src_bbox[3] = max(aoi_src_bbox[3], 0);
}

void compute_tile_var(double* data, int width, int height,double& var, double& valid_ratio) {
    double mean=0;
    var = 0;
    int valid_cnt = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (isnan(data[i * width + j])) {
                continue;
            }
            mean += data[i * width + j];
            valid_cnt++;
        }
    }
    mean /=valid_cnt;
    valid_ratio = (double)valid_cnt / (double)(width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (isnan(data[i * width + j])) {
                continue;
            }
            var += pow(data[i * width + j]-mean,2.0);
        }
    }
    var/= valid_cnt;
    var = std::sqrt(var);
}

void write_ptcloud(double* data, int width, int height, double resolution, string out_path,double offx,double offy) {
    std::ofstream outfile(out_path);
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            if (isnan(data[col + row * width])) {
                continue;
            }
            outfile << std::fixed << std::setw(11) << std::setprecision(11) << (double)col * resolution+offx << " " <<
                std::fixed << std::setw(11) << std::setprecision(11) << (double)(height-row) * resolution+offy << " " <<
                std::fixed << std::setw(11) << std::setprecision(11) << data[col + row * width] << "\n";
        }
    }
    outfile.close();
}

void write_ptcloud_corrs(std::vector<pair<gs::Point*, gs::Point*>>& corrs, string out_ref_path, string out_src_path) {
    std::ofstream out_ref(out_ref_path);
    std::ofstream out_src(out_src_path);
    for (int i = 0; i < corrs.size(); ++i) {
        out_src << std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].first->pos[0] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].first->pos[1] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].first->pos[2] << "\n";
    }
    for (int i = 0; i < corrs.size(); ++i) {
        out_ref << std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].second->pos[0] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].second->pos[1] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)corrs[i].second->pos[2] << "\n";
    }

    out_ref.close();
    out_src.close();
}

void write_ptcloud_gs(std::vector<gs::Point>& pts, string out_path,double offx,double offy) {
    std::ofstream outfile(out_path);
    for (int i= 0; i < pts.size(); ++i) {
        outfile << std::fixed << std::setw(11) << std::setprecision(11) << (double)pts[i].pos[0]+offx << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)pts[i].pos[1]+offy << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)pts[i].pos[2] << "\n";

    }
    outfile.close();
}

void convert_ptcloud(double* data, int width, int height,double resolution,int resolution_factor, vector<gs::Point>& ptcloud, double offx, double offy) {
    ptcloud.clear();
    for (int row = 0; row < height; row+=resolution_factor) {
        for (int col = 0; col < width; col+=resolution_factor) {
            if (isnan(data[col + row * width])) {
                continue;
            }
            gs::Point pt((double)col * resolution+offx, (double)(- row) * resolution+offy, data[col + row * width]);
            ptcloud.push_back(pt);
        }
    }

}

void update_ptcloud(vector<gs::Point>& ptcloud, double* rot, double* trans,double offx_in,double offy_in,double offx_rot,double offy_rot) {
    double tmp[3];
    for (int i = 0; i < ptcloud.size(); ++i) {
        ptcloud[i].pos[0] = ptcloud[i].pos[0] + offx_in - offx_rot;
        ptcloud[i].pos[1] = ptcloud[i].pos[1] + offy_in - offy_rot;

        gs::rotate_mat(ptcloud[i].pos, rot, tmp);
        ptcloud[i].pos[0] = tmp[0] + trans[0];
        ptcloud[i].pos[1] = tmp[1] + trans[1];
        ptcloud[i].pos[2] = tmp[2] + trans[2];
        
        ptcloud[i].pos[0] = ptcloud[i].pos[0]+offx_rot - offx_in;
        ptcloud[i].pos[1] = ptcloud[i].pos[1]+offy_rot - offy_in;
    }
}

void update_ptcloud_offset(vector<gs::Point>& ptcloud, double offx_in, double offy_in, double offx_out, double offy_out) {
    double tmp[3];
    for (int i = 0; i < ptcloud.size(); ++i) {
        ptcloud[i].pos[0] = ptcloud[i].pos[0] + offx_in - offx_out;
        ptcloud[i].pos[1] = ptcloud[i].pos[1] + offy_in - offy_out;
    }
}

void load_rmse(std::string path, double** rmses) {
    ifstream ifs(path);
    int x, y;
    double rmse;
    char line[1000];
    string line_str;
    while (!ifs.eof()) {
        //ifs >> line;
        getline(ifs, line_str);
        sscanf(line_str.c_str(), "%d %d %lf", &x, &y, &rmse);
        rmses[y][x] = rmse;
    }
}

void write_shp(std::vector<TILE_INFO> tile_infos, string out_path) {
    GDALAllRegister();
    const char* pszDriverName = "ESRI Shapefile";
    GDALDriver* poDriver;
    poDriver = GetGDALDriverManager()->GetDriverByName(pszDriverName);
    if (poDriver == NULL)
    {
        printf("%s driver not available.\n", pszDriverName);
        exit(1);
    }
    GDALDataset* poDS;
    poDS = poDriver->Create(out_path.c_str(), 0, 0, 0, GDT_Unknown, NULL);
    if (poDS == NULL)
    {
        printf("Creation of output file failed.\n");
        exit(1);
    }
    OGRLayer* poLayer;

    poLayer = poDS->CreateLayer("tiles", NULL, wkbPolygon, NULL);
    if (poLayer == NULL)
    {
        printf("Layer creation failed.\n");
        exit(1);
    }
    OGRFieldDefn oField1("Ref_Var", OFTString);
    OGRFieldDefn oField2("Src_Var", OFTString);
    OGRFieldDefn oField3("Tile_ID", OFTString);
    OGRFieldDefn oField4("Ref_Valid", OFTString);
    OGRFieldDefn oField5("Src_Valid", OFTString);
    OGRFieldDefn oField6("Src_lmd1", OFTString);
    OGRFieldDefn oField7("Src_lmd2", OFTString);
    OGRFieldDefn oField8("Src_R", OFTString);
    OGRFieldDefn oField9("Ref_lmd1", OFTString);
    OGRFieldDefn oField10("Ref_lmd2", OFTString);
    OGRFieldDefn oField11("Ref_R", OFTString);
    oField1.SetWidth(32);
    oField2.SetWidth(32);
    oField3.SetWidth(32);
    oField4.SetWidth(32);
    oField5.SetWidth(32);
    oField6.SetWidth(32);
    oField7.SetWidth(32);
    oField8.SetWidth(32);
    oField9.SetWidth(32);
    oField10.SetWidth(32);
    oField11.SetWidth(32);

    if (poLayer->CreateField(&oField1) != OGRERR_NONE || poLayer->CreateField(&oField2) != OGRERR_NONE || 
        poLayer->CreateField(&oField3) != OGRERR_NONE || poLayer->CreateField(&oField4) != OGRERR_NONE || 
        poLayer->CreateField(&oField5) != OGRERR_NONE || poLayer->CreateField(&oField6) != OGRERR_NONE ||
        poLayer->CreateField(&oField7) != OGRERR_NONE || poLayer->CreateField(&oField8) != OGRERR_NONE ||
        poLayer->CreateField(&oField9) != OGRERR_NONE || poLayer->CreateField(&oField10) != OGRERR_NONE || 
        poLayer->CreateField(&oField11) != OGRERR_NONE)
    {
        printf("Creating Name field failed.\n");
        exit(1);
    }
    for (int i = 0; i < tile_infos.size(); ++i) {
        OGRFeature* poFeature;
        poFeature = OGRFeature::CreateFeature(poLayer->GetLayerDefn());
        string var_str= to_string(tile_infos[i].ref_var);
        string id_str = to_string(tile_infos[i].tile_id);
        char* cstr = new char[var_str.length() + 1];
        var_str.copy(cstr, var_str.length());
        cstr[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Ref_Var"), cstr);
        delete[] cstr;

        var_str = to_string(tile_infos[i].src_var);
        char* cstr2 = new char[var_str.length() + 1];
        var_str.copy(cstr2, var_str.length());
        cstr2[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Src_Var"), cstr2);
        delete[] cstr2;

        char* cstr1 = new char[id_str.length() + 1];
        id_str.copy(cstr1, id_str.length());
        cstr1[id_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Tile_ID"), cstr1);
        delete[] cstr1;

        var_str = to_string(tile_infos[i].ref_valid);
        char* cstr3 = new char[var_str.length() + 1];
        var_str.copy(cstr3, var_str.length());
        cstr3[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Ref_Valid"), cstr3);
        delete[] cstr3;

        var_str = to_string(tile_infos[i].src_valid);
        char* cstr4 = new char[var_str.length() + 1];
        var_str.copy(cstr4, var_str.length());
        cstr4[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Src_Valid"), cstr4);
        delete[] cstr4;

        var_str = to_string(tile_infos[i].src_lambda1);
        char* cstr5 = new char[var_str.length() + 1];
        var_str.copy(cstr5, var_str.length());
        cstr5[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Src_Lmd1"), cstr5);
        delete[] cstr5;

        var_str = to_string(tile_infos[i].src_lambda2);
        char* cstr6 = new char[var_str.length() + 1];
        var_str.copy(cstr6, var_str.length());
        cstr6[var_str.length()] = '\0';
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Src_Lmd2"), cstr6);
        delete[] cstr6;

        var_str = to_string(tile_infos[i].src_R);
        char* cstr7 = new char[var_str.length() + 1];
        var_str.copy(cstr7, var_str.length());
        cstr7[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Src_R"), cstr7);
        delete[] cstr7;

        var_str = to_string(tile_infos[i].ref_lambda1);
        char* cstr8 = new char[var_str.length() + 1];
        var_str.copy(cstr8, var_str.length());
        cstr8[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Ref_Lmd1"), cstr8);
        delete[] cstr8;

        var_str = to_string(tile_infos[i].ref_lambda2);
        char* cstr9 = new char[var_str.length() + 1];
        var_str.copy(cstr9, var_str.length());
        cstr9[var_str.length()] = '\0';
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Ref_Lmd2"), cstr9);
        delete[] cstr9;

        var_str = to_string(tile_infos[i].ref_R);
        char* cstr10 = new char[var_str.length() + 1];
        var_str.copy(cstr10, var_str.length());
        cstr10[var_str.length()] = '\0';
        //poFeature->SetField("Variance", cstr);
        OGR_F_SetFieldString(poFeature, OGR_F_GetFieldIndex(poFeature, "Ref_R"), cstr10);
        delete[] cstr10;

        OGRGeometryH geo = OGR_G_CreateGeometry(wkbLinearRing);
        OGR_G_SetPoint_2D(geo, 0, tile_infos[i].utm_bbox[0], tile_infos[i].utm_bbox[2]);
        OGR_G_SetPoint_2D(geo, 1, tile_infos[i].utm_bbox[0], tile_infos[i].utm_bbox[3]);
        OGR_G_SetPoint_2D(geo, 2, tile_infos[i].utm_bbox[1], tile_infos[i].utm_bbox[3]);
        OGR_G_SetPoint_2D(geo, 3, tile_infos[i].utm_bbox[1], tile_infos[i].utm_bbox[2]);
        OGR_G_CloseRings(geo);
        geo = OGR_G_ForceToPolygon(geo);
        OGR_F_SetGeometry(poFeature, geo);
        OGR_G_DestroyGeometry(geo);
        if (poLayer->CreateFeature(poFeature) != OGRERR_NONE)
        {
            printf("Failed to create feature in shapefile.\n");
            exit(1);
        }

        OGRFeature::DestroyFeature(poFeature);
    }

    GDALClose(poDS);

}

void write_tile_info_txt(std::vector<TILE_INFO> tile_infos, string out_path) {
    std::ofstream outfile(out_path);
    for (int  i= 0; i < tile_infos.size(); ++i) {
        outfile << tile_infos[i].tile_id <<"\n";
        
    }
    outfile.close();
}

void write_shift(std::vector<TILE_INFO> tile_infos, string out_path) {
    std::ofstream outfile(out_path);
    for (int i=0;i<tile_infos.size();++i){
        outfile << std::fixed << std::setw(11) << std::setprecision(11) << (double)tile_infos[i].translation[0] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)tile_infos[i].translation[1] << " " <<
            std::fixed << std::setw(11) << std::setprecision(11) << (double)tile_infos[i].translation[2] << "\n";
        
    }
    outfile.close();
}

bool cmp_var(pair<float, int>& a, pair<float, int>& b) {
    return a.first > b.first;
}

void ESTIMATE_TRANSFORMATION_CORRS(std::vector<pair<gs::Point, gs::Point>>& corrs, double* rotation, double* translation) {
    gs::Point dynamicMid(0, 0, 0), d_raw, s_raw, d_delta, s_delta;
    gs::Point staticMid(0, 0, 0);
    for (int i = 0; i < corrs.size(); ++i) {
        dynamicMid = dynamicMid + corrs[i].first;
        staticMid = staticMid + corrs[i].second;

    }
    dynamicMid.pos[0] = dynamicMid.pos[0] / corrs.size();
    dynamicMid.pos[1] = dynamicMid.pos[1] / corrs.size();
    dynamicMid.pos[2] = dynamicMid.pos[2] / corrs.size();
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
    gs::clearMatrix(U);
    gs::clearMatrix(V);
    gs::clearMatrix(w);
    for (int i = 0; i < corrs.size(); i++)
    {
        d_raw = corrs[i].first;
        s_raw = corrs[i].second;
        d_delta = d_raw - dynamicMid;
        s_delta = s_raw - staticMid;

        //outerProduct(&d_delta, &s_delta, w);
        gs::outerProduct(&s_delta, &d_delta, w);
        gs::addMatrix(w, U, U);
    }
    gs::copyMatToUV(U, uSvd);
    dsvd(uSvd, 3, 3, sigma, vSvd);
    gs::copyUVtoMat(uSvd, U);
    gs::copyUVtoMat(vSvd, V);

    /// Compute Rotation matrix
    gs::initDiagonal(Diagonal);
    if (gs::determinant_3by3(uSvd) * gs::determinant_3by3(vSvd) < 0) {
        Diagonal[8] = -1;
    }
    //transpose(U);
    gs::transpose(V);
    double tmp[9];
    double tmp_vec3[3];
    gs::clearMatrix(tmp);
    //matrixMult(V, Diagonal,tmp);
    //matrixMult(tmp, U, rotationMatrix);
    gs::matrixMult(U, Diagonal, tmp);
    gs::matrixMult(tmp, V, rotation);

    gs::Point t(0.0, 0.0, 0.0);
    gs::rotate(&dynamicMid, rotation, &t);
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

void update_utm_bbox(double* utm_bbox, double* rotation, double* translation) {
    utm_bbox[0] += translation[0];
    utm_bbox[1] += translation[0];
    utm_bbox[2] += translation[1];
    utm_bbox[3] += translation[1];
}

void gen_rotation(double* rot, double x_angle,double y_angle,double z_angle) { // in degree
    gs::initDiagonal(rot);
    double rot_tmp1[9];
    double rot_tmp2[9];
    gs::initDiagonal(rot_tmp1);
    gs::initDiagonal(rot_tmp2);

    x_angle = x_angle * 3.1415926 / 180.0;
    y_angle = y_angle * 3.1415926 / 180.0;
    z_angle = z_angle * 3.1415926 / 180.0;
    //Apply x rot
    rot_tmp1[4] = cos(x_angle);
    rot_tmp1[5] = -sin(x_angle);
    rot_tmp1[7] = sin(x_angle);
    rot_tmp1[8] = cos(x_angle);
    gs::matrixMult(rot_tmp1, rot, rot_tmp2);
    //Apply y rot
    gs::initDiagonal(rot_tmp1);
    rot_tmp1[0] = cos(y_angle);
    rot_tmp1[2] = sin(y_angle);
    rot_tmp1[6] = -sin(y_angle);
    rot_tmp1[8] = cos(y_angle);
    gs::matrixMult(rot_tmp1, rot_tmp2, rot);
    //Apply z rot
    gs::initDiagonal(rot_tmp1);
    rot_tmp1[0] = cos(z_angle);
    rot_tmp1[1] = -sin(z_angle);
    rot_tmp1[3] = sin(z_angle);
    rot_tmp1[4] = cos(z_angle);
    gs::matrixMult(rot_tmp1, rot, rot_tmp2);
    std::copy(&rot_tmp2[0], &rot_tmp2[9], &rot[0]);
}
void gen_rotation_zaxis(double* rot, double angle) { // in degree
    gs::initDiagonal(rot);
    angle = angle * 3.1415926 / 180.0;
    rot[0] = cos(angle);
    rot[1] = -sin(angle);
    rot[3] = sin(angle);
    rot[4] = cos(angle);
}

/* [R2,T2]*[R1,T1]=[Rout,Tout]
*/
void merge_two_trans(double* rot1, double* trans1, double* rot2, double* trans2, double* out_rot, double* out_trans) {
	double mid[3];
	gs::matrixMult(rot2, rot1, out_rot);
	gs::rotate_mat(trans1,rot2, mid);
	out_trans[0] = mid[0] + trans2[0];
	out_trans[1] = mid[1] + trans2[1];
	out_trans[2] = mid[2] + trans2[2];
}

void create_dsm(string out_path) {
    int out_width = 70;
    int out_height = 70;
    int min = 45;
    int max = 50;
    GDALAllRegister();
    GDALDriver* pdriver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* out_dataset_ = pdriver->Create(out_path.c_str(), out_width, out_height, 1, GDT_Float64, NULL);
    GDALRasterBand* out_band_ = out_dataset_->GetRasterBand(1);
    double* write_value = new double[out_width*out_height];
    int id = 0;
    for (int row = 0; row < out_height; ++row) {
        for (int col = 0; col < out_width; ++col) {
            id = col + row * out_width;
            if (col < max && col>min && row < max && row>min) {
                write_value[id] = 10.0;
            }
            else {
                write_value[id] = 0.0;
            }
        }
    }
    out_band_->RasterIO(GF_Write, 0, 0, out_width, out_height,
        write_value, out_width, out_height, GDT_Float64, 0, 0);
    GDALClose(out_dataset_);
}

class DSM_REG {
public:

    DSM_REG(std::string ref_path, std::string src_path) {
        //General 
        plane_or_not_ = false;
        multi_or_not_ = false;
        brute_force_search_or_not_ = false;
        xmin_m_ = -3; // for brute-force search bounding
        xmax_m_ = 3;
        ymin_m_ = -3;
        ymax_m_ = 3;
        rough_step_m_ = 0.5;
        rough_step_min_resolution_ = 3; //local_minimal_within_range=rough_step_min_resolution_*rough_step_m_
        rough_num_mins_ = 3; // number of local minimal
        fine_step_m_= 0.5;
        verbose_ = 1;
        //Rough

        rough_icp_percentage_threshold_ = 1e-5;
        valid_ratio_threshold_rough_ = 0.5;
        rough_num_pts_ = 24000;
        search_half_size_pixel_ = 20;
        trimmed_ratio_ = 0.6;
        rough_icp_rmse_threshold_ = 1e-5;
        rough_icp_max_iter_ = 500;
        // Fine
        n_blocks_col_ = 5;
        n_blocks_row_ = 5;
        fine_resolution_factor_ = 5;
        fine_search_half_pixels_ = 10;
        fine_icp_rmse_threshold_ = 1e-5;
        fine_icp_max_iter_ = 1500;
        fine_icp_percentage_threshold_ = 1e-5;
        num_filter_tiles_ = 100; // (>0) minimal is n_blocks_col_*n_blocks_row_*1
        HARRIS_k_ = 0.05;
        HARRIS_step_ = 3;
        resolution_ = 0.5;
        var_min_ = 4.0;
        valid_ratio_threshold_ = 0.5;
        tilesize_m_ = 100;
        ref_path_ = ref_path;
        src_path_ = src_path;
        gs::initDiagonal(rough_rotation_);
        rough_translation_[0] = 0.0;
        rough_translation_[1] = 0.0;
        rough_translation_[2] = 0.0;
        fine_translation_[0] = 0.0;
        fine_translation_[1] = 0.0;
        fine_translation_[2] = 0.0;
        double transform[6];
        int xSize, ySize;

        GDALAllRegister();
        pdriver_ = GetGDALDriverManager()->GetDriverByName("GTiff");
        ref_dataset_ = (GDALDataset*)GDALOpen(ref_path.c_str(), GA_ReadOnly);
        ref_dataset_->GetGeoTransform(transform);
        xSize = ref_dataset_->GetRasterXSize();
        ySize = ref_dataset_->GetRasterYSize();
        ref_width_ = xSize;
        ref_height_ = ySize;
        ref_utm_bbox_[0] = 0.0;
        ref_utm_bbox_[1] = 0.0 + (xSize - 1) * transform[1];
        ref_utm_bbox_[2] = 0.0 - (ySize - 1) * transform[1];
        ref_utm_bbox_[3] = 0.0;
        GLOBAL_OFFSET_X_m_ = transform[0];
        GLOBAL_OFFSET_Y_m_ = transform[3];


        src_dataset_ = (GDALDataset*)GDALOpen(src_path.c_str(), GA_ReadOnly);
        src_dataset_->GetGeoTransform(transform);
        xSize = src_dataset_->GetRasterXSize();
        ySize = src_dataset_->GetRasterYSize();
        src_width_ = xSize;
        src_height_ = ySize;
        src_utm_bbox_[0] = transform[0]- GLOBAL_OFFSET_X_m_;
        src_utm_bbox_[1] = transform[0]- GLOBAL_OFFSET_X_m_ + (xSize - 1) * transform[1];
        src_utm_bbox_[2] = transform[3]- GLOBAL_OFFSET_Y_m_ - (ySize - 1) * transform[1];
        src_utm_bbox_[3] = transform[3]- GLOBAL_OFFSET_Y_m_;

        aoi_utm_bbox_[0] = max(ref_utm_bbox_[0], src_utm_bbox_[0]);
        aoi_utm_bbox_[1] = min(ref_utm_bbox_[1], src_utm_bbox_[1]);
        aoi_utm_bbox_[2] = max(ref_utm_bbox_[2], src_utm_bbox_[2]);
        aoi_utm_bbox_[3] = min(ref_utm_bbox_[3], src_utm_bbox_[3]);

        get_aoi_bbox_pixels(aoi_utm_bbox_, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, aoi_ref_bbox_pixel_, aoi_src_bbox_pixel_);

        num_tile_x_ = ceil((aoi_utm_bbox_[1] - aoi_utm_bbox_[0]) / tilesize_m_);
        num_tile_y_ = ceil((aoi_utm_bbox_[3] - aoi_utm_bbox_[2]) / tilesize_m_);

        ref_band_ = ref_dataset_->GetRasterBand(1);
        src_band_ = src_dataset_->GetRasterBand(1);
        std::cout << "################ DSM_REG Paras ################### " << std::endl;
		std::cout << "ICP point-to-plane: " << plane_or_not_ << std::endl;
        std::cout << "Rough Align Paras: "<< std::endl;
        std::cout << "# Sampling Pts: "<< rough_num_pts_ << std::endl;
        std::cout << "NN search half size (pixels): " << search_half_size_pixel_ << std::endl;
        std::cout << "Valid Threshold: " << valid_ratio_threshold_rough_ << std::endl;
        std::cout << "Rough ICP Max Iters: " << rough_icp_max_iter_ <<'\n'<< std::endl;
        std::cout << "Fine Align Paras: " << std::endl;
        std::cout << "# Filtered TIles: " << num_filter_tiles_ << std::endl;
        std::cout << "Tiel size (m): " << tilesize_m_ << std::endl;
        std::cout << "Variance Minimal Threshold: " << var_min_ << std::endl;
        std::cout << "Valid Threshold: " << valid_ratio_threshold_ <<'\n'<< std::endl;


    };

    std::string ref_path_, src_path_;
    GDALDriver* pdriver_;
    GDALDataset* ref_dataset_;
    GDALDataset* src_dataset_;
    GDALRasterBand* ref_band_;
    GDALRasterBand* src_band_;
    int ref_width_, ref_height_, src_width_, src_height_;
    double GLOBAL_OFFSET_X_m_;
    double GLOBAL_OFFSET_Y_m_;
    double resolution_;
    double ref_utm_bbox_[4];
    double src_utm_bbox_[4];
    double aoi_utm_bbox_[4];
    int aoi_ref_bbox_pixel_[4];
    int aoi_src_bbox_pixel_[4];
    int num_tile_x_, num_tile_y_;
    double tilesize_m_;
    std::vector<TILE_INFO> tile_infos_all_, tile_infos_filter_, tile_infos_icp_;
    double valid_ratio_threshold_, var_min_;
    int verbose_;
    // for General 
    bool plane_or_not_;
    bool multi_or_not_;
    bool brute_force_search_or_not_;
    double xmin_m_;
    double xmax_m_;
    double ymin_m_;
    double ymax_m_;
    double rough_step_m_;
    int rough_step_min_resolution_;
    int rough_num_mins_;
    double fine_step_m_;
    // for rough align
    double valid_ratio_threshold_rough_;
    double rough_rotation_[9];
    double rough_translation_[3];
    vector<vector<double>> rough_translations_;
    double rough_icp_max_iter_;
    double rough_icp_rmse_threshold_;
    double trimmed_ratio_;
    double rough_icp_percentage_threshold_;
    double rough_rmse_;
    // for fine align
    double HARRIS_step_;
    double HARRIS_k_;
    double fine_translation_[3];
    int n_blocks_col_;
    int n_blocks_row_;
    int num_filter_tiles_;
    double fine_icp_max_iter_;
    double fine_icp_rmse_threshold_;
    double fine_icp_percentage_threshold_;
    double fine_search_half_pixels_;
    int fine_resolution_factor_;
    double fine_rmse_;
    double fine_rotation_[9];

	double final_rot_[9];
	double final_trans_[3];

    // FOR rough align
    int rough_num_pts_;
    int search_half_size_pixel_;
    std::vector<gs::Point> rough_src_pts_;
    std::vector<std::pair<int, int>> rough_src_index_; // in source image pixel coordinates
    
    static bool cmp_ascend(std::pair<float, int>& a, std::pair<float, int>& b) {
        return a.first < b.first;
    }
    static bool cmp_descend(std::pair<float, int>& a, std::pair<float, int>& b) {
        return a.first > b.first;
    }
    static bool cmp_descend_double(std::pair<double, int>& a, std::pair<double, int>& b) {
        return a.first > b.first;
    }
    static bool cmp_descend_only_double(double& a, double& b) {
        return a > b;
    }
    template <typename T>
    static vector<size_t> sort_indexes(const vector<T>& v) {

        // initialize original index locations
        vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values 
        stable_sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });

        return idx;
    }
    
    void ROUGH_FIND_CORRESPONDENCE(std::vector<pair<gs::Point, gs::Point>>& corrs,double& RMSE) {
        //clear corrs
        corrs.clear();
        RMSE = 0;
        double utm_x, utm_y;
        int grid_x, grid_y;
        int search_xmin, search_xmax, search_ymax, search_ymin;
        int search_width, search_height;
        double* ref_search_area = new double[(2*search_half_size_pixel_+1)* (2 * search_half_size_pixel_ + 1)];
        for (int i = 0; i < rough_src_pts_.size(); ++i) {
            gs::Point src_pts;
            src_pts.pos[0] = rough_src_pts_[i].pos[0];
            src_pts.pos[1] = rough_src_pts_[i].pos[1];
            src_pts.pos[2] = rough_src_pts_[i].pos[2];
            utm_x = src_pts.pos[0];
            utm_y = src_pts.pos[1];
            grid_x = floor((utm_x - ref_utm_bbox_[0] + 0.25) / resolution_);
            grid_y = floor((ref_utm_bbox_[3] - utm_y + 0.25) / resolution_);
            if (grid_x < 0 || grid_x >= ref_width_ || grid_y < 0 || grid_y >= ref_height_) {
                continue;
            }
            search_xmin = grid_x - search_half_size_pixel_;
            search_xmax = grid_x + search_half_size_pixel_;
            search_ymin = grid_y - search_half_size_pixel_;
            search_ymax = grid_y + search_half_size_pixel_;
            search_xmin = (search_xmin < 0) ? 0 : search_xmin;
            search_xmax = (search_xmax >= ref_width_) ? ref_width_-1 : search_xmax;
            search_ymin = (search_ymin < 0) ? 0 : search_ymin;
            search_ymax = (search_ymax >= ref_height_) ? ref_height_-1 : search_ymax;
            search_width = search_xmax - search_xmin+1;
            search_height = search_ymax - search_ymin + 1;


            ref_band_->RasterIO(GF_Read, search_xmin, search_ymin,
                search_width, search_height,
                ref_search_area, search_width, search_height, GDT_Float64, 0, 0);
            int best_row = 0;
            int best_col = 0;
            double best_dis = 9999;
            double cur_dis;
            double ref_x, ref_y;
            int valid_cnt = 0;
            double valid_ratio = 0;
            for (int row = 0; row < search_height; ++row) {
                for (int col = 0; col < search_width; ++col) {
                    if (isnan(ref_search_area[col + row * search_width])) {
                        continue;
                    }
                    valid_cnt++;
                    ref_x = ref_utm_bbox_[0] + (search_xmin+ col) * resolution_;
                    ref_y = ref_utm_bbox_[3] - (search_ymin+ row) * resolution_;

                    cur_dis = pow(src_pts.pos[0] - ref_x, 2.0) + pow(src_pts.pos[1] - ref_y, 2.0) + pow(src_pts.pos[2] - ref_search_area[col + row * search_width], 2.0);
                    if (cur_dis < best_dis) {
                        best_row = row;
                        best_col = col;
                        best_dis = cur_dis;
                    }
                }
            }
            valid_ratio = (double)valid_cnt / (double)(search_width*search_height);
            if (best_dis == 9999 || valid_ratio < valid_ratio_threshold_rough_) {
                continue;
            }
            gs::Point ref_pts(ref_utm_bbox_[0] + (search_xmin + best_col) * resolution_,
                ref_utm_bbox_[3] - (search_ymin + best_row) * resolution_,
                ref_search_area[best_col + best_row * search_width]);
            corrs.push_back(make_pair(src_pts,ref_pts));
            
        }
        // reject outlier
        CORR_REJECTOR(corrs);
        //compute rmse
        for (int i = 0; i < corrs.size(); ++i) {
            RMSE += pow(corrs[i].first.pos[0] - corrs[i].second.pos[0], 2.0) +
                pow(corrs[i].first.pos[1] - corrs[i].second.pos[1], 2.0) +
                pow(corrs[i].first.pos[2] - corrs[i].second.pos[2], 2.0);
        }
        RMSE = sqrt(RMSE / (double)corrs.size());
        delete[] ref_search_area;
    }
    
    void FIND_MULTI_LINK_CORRESPONDENCE(vector<gs::Point> src_pts, Eigen::MatrixXd& corr_src, Eigen::MatrixXd& corr_dst, Eigen::VectorXd& corr_w,
                                        int search_half_size_pixel, double percentage_threshold, bool multi_or_not,
                                        int max_iter, double& RMSE, bool plane_or_not) {

        RMSE = 0;
        double utm_x, utm_y;
        int grid_x, grid_y;
        int search_xmin, search_xmax, search_ymax, search_ymin;
        int search_width, search_height;
        double* ref_search_area = new double[(2 * search_half_size_pixel + 1) * (2 * search_half_size_pixel + 1)];
        double src_pts_num = src_pts.size();
        double valid_src_pts = 0.0;
        std::vector<CORR> corrs;
        for (int i = 0; i < src_pts.size(); ++i) {
            gs::Point src_pt = src_pts[i];

            utm_x = src_pt.pos[0];
            utm_y = src_pt.pos[1];
            grid_x = floor((utm_x - ref_utm_bbox_[0] + 0.25) / resolution_);
            grid_y = floor((ref_utm_bbox_[3] - utm_y + 0.25) / resolution_);
            if (grid_x < 0 || grid_x >= ref_width_ || grid_y < 0 || grid_y >= ref_height_) {
                continue;
            }
            search_xmin = grid_x - search_half_size_pixel;
            search_xmax = grid_x + search_half_size_pixel;
            search_ymin = grid_y - search_half_size_pixel;
            search_ymax = grid_y + search_half_size_pixel;
            search_xmin = (search_xmin < 0) ? 0 : search_xmin;
            search_xmax = (search_xmax >= ref_width_) ? ref_width_ - 1 : search_xmax;
            search_ymin = (search_ymin < 0) ? 0 : search_ymin;
            search_ymax = (search_ymax >= ref_height_) ? ref_height_ - 1 : search_ymax;
            search_width = search_xmax - search_xmin + 1;
            search_height = search_ymax - search_ymin + 1;


            ref_band_->RasterIO(GF_Read, search_xmin, search_ymin,
                search_width, search_height,
                ref_search_area, search_width, search_height, GDT_Float64, 0, 0);
            int best_row = 0;
            int best_col = 0;
            double best_dis = 9999;
            double cur_dis;
            double ref_x, ref_y, ref_z;
            int valid_cnt = 0;
            double valid_ratio = 0;
            double sum_guass = 0;
            double guass = 0;
            double percentage = 0;
            int idx = 0;
            vector<int> candidate_index_index;
            std::vector<pair<double, int>> candidate_index;
            for (int row = 1; row < search_height - 1; ++row) {
                for (int col = 1; col < search_width - 1; ++col) {
                    idx = col + row * search_width;
                    if (isnan(ref_search_area[idx])) {
                        continue;
                    }
                    if (plane_or_not && (isnan(ref_search_area[idx - 1]) ||
                        isnan(ref_search_area[idx + 1]) ||
                        isnan(ref_search_area[idx - search_width]) ||
                        isnan(ref_search_area[idx + search_width]))) {
                        continue;
                    }
                    valid_cnt++;
                    ref_x = ref_utm_bbox_[0] + (search_xmin + col) * resolution_;
                    ref_y = ref_utm_bbox_[3] - (search_ymin + row) * resolution_;
                    ref_z = ref_search_area[idx];
                    cur_dis = pow(src_pt.pos[0] - ref_x, 2.0) + pow(src_pt.pos[1] - ref_y, 2.0) + pow(src_pt.pos[2] - ref_z, 2.0);
                    guass = exp(-cur_dis);
                    //sum_guass += guass;
                    candidate_index.push_back(make_pair(guass, idx));
                    if (cur_dis < best_dis) {
                        best_row = row;
                        best_col = col;
                        best_dis = cur_dis;
                    }
                }
            }
            valid_ratio = (double)valid_cnt / (double)(search_width * search_height);
            if (valid_ratio < valid_ratio_threshold_rough_) { continue; }
            // start go multi-correspondence
            if (multi_or_not) {
                sort(candidate_index.begin(), candidate_index.end(), cmp_descend_double);
                int col, row;
                sum_guass = 0.0;
                //num_multi_link = candidate_index.size();
                for (int iter = 0; iter < candidate_index.size(); ++iter) {
                    guass = candidate_index[iter].first;
                    sum_guass += guass;
                }
                for (int iter = 0; iter < candidate_index.size(); ++iter) {
                    col = candidate_index[iter].second % search_width;
                    row = candidate_index[iter].second / search_width;
                    guass = candidate_index[iter].first;
                    percentage = guass / sum_guass;
                    if (percentage < percentage_threshold || isnan(ref_search_area[candidate_index[iter].second]) || isnan(guass)) {
                        continue;
                    }
                    candidate_index_index.push_back(iter);
                }
                sum_guass = 0.0;
                for (int iter = 0; iter < candidate_index_index.size(); ++iter) {
                    guass = candidate_index[candidate_index_index[iter]].first;
                    sum_guass += guass;
                }
                if (sum_guass < 1e-5) {
                    continue;
                }
                for (int iter = 0; iter < candidate_index_index.size(); ++iter) {
                    idx = candidate_index_index[iter];
                    col = candidate_index[idx].second % search_width;
                    row = candidate_index[idx].second / search_width;
                    ref_x = ref_utm_bbox_[0] + (search_xmin + col) * resolution_;
                    ref_y = ref_utm_bbox_[3] - (search_ymin + row) * resolution_;
                    gs::Point ref_pt(ref_x,
                        ref_y,
                        ref_search_area[candidate_index[idx].second]);
                    guass = candidate_index[idx].first;
                    CORR corr;
                    if (plane_or_not) {
                        float ref_dx, ref_dy, ref_dz, magnitude;
                        vector<float> src_norm, ref_norm;
                        ref_dx = (ref_search_area[candidate_index[idx].second + 1] - ref_search_area[candidate_index[idx].second - 1]) / 2.0;
                        ref_dy = (ref_search_area[candidate_index[idx].second - search_width] - ref_search_area[candidate_index[idx].second + search_width]) / 2.0;
                        magnitude = sqrt(pow(ref_dx, 2.0) + pow(ref_dy, 2) + 1);
                        ref_dx /= magnitude;
                        ref_dy /= magnitude;
                        ref_dz = 1.0 / magnitude;
                        ref_norm.push_back(ref_dx);
                        ref_norm.push_back(ref_dy);
                        ref_norm.push_back(ref_dz);
                        corr.ref = ref_pt;
                        corr.src = src_pt;
                        corr.ref_norm = ref_norm;
                        corr.w = guass / sum_guass;
                    }
                    else {
                        corr.ref = ref_pt;
                        corr.src = src_pt;
                        corr.w = guass / sum_guass;
                    }

                    corrs.push_back(corr);
                }
                valid_src_pts++;
            }
            else {
                if (best_dis == 9999) {
                    continue;
                }
                gs::Point ref_pt(ref_utm_bbox_[0] + (search_xmin + best_col) * resolution_,
                    ref_utm_bbox_[3] - (search_ymin + best_row) * resolution_,
                    ref_search_area[best_col + best_row * search_width]);
                CORR corr;
                if (plane_or_not) {
                    float ref_dx, ref_dy, ref_dz, magnitude;
                    vector<float> src_norm, ref_norm;
                    ref_dx = (ref_search_area[best_col + 1 + best_row * search_width] - ref_search_area[best_col - 1 + best_row * search_width] / 2.0);
                    ref_dy = (ref_search_area[best_col + (best_row - 1) * search_width] - ref_search_area[best_col + (best_row + 1) * search_width] / 2.0);
                    magnitude = sqrt(pow(ref_dx, 2.0) + pow(ref_dy, 2) + 1);
                    ref_dx /= magnitude;
                    ref_dy /= magnitude;
                    ref_dz = 1.0 / magnitude;
                    ref_norm.push_back(ref_dx);
                    ref_norm.push_back(ref_dy);
                    ref_norm.push_back(ref_dz);
                    corr.ref = ref_pt;
                    corr.src = src_pt;
                    corr.ref_norm = ref_norm;
                    corr.w = 1;
                }
                else {
                    corr.ref = ref_pt;
                    corr.src = src_pt;
                    corr.w = 1;
                }
                corrs.push_back(corr);
                valid_src_pts++;
            }
        }
        //reweight the corr
        for (int i = 0; i < corrs.size(); ++i) {
            corrs[i].w /= valid_src_pts;
        }
        // reject outlier
        if (!multi_or_not) {
            CORR_REJECTOR1(corrs, 1.0);
        }

        //compute rmse
        if (plane_or_not) {
            for (int i = 0; i < corrs.size(); ++i) {
                RMSE += corrs[i].w * (
                    pow((corrs[i].src.pos[0] - corrs[i].ref.pos[0]) * corrs[i].ref_norm[0], 2) +
                    pow((corrs[i].src.pos[1] - corrs[i].ref.pos[1]) * corrs[i].ref_norm[1], 2) +
                    pow((corrs[i].src.pos[2] - corrs[i].ref.pos[2]) * corrs[i].ref_norm[2], 2)
                    );
            }
            RMSE = sqrt(RMSE);
        }
        else {
            for (int i = 0; i < corrs.size(); ++i) {
                RMSE += corrs[i].w * (pow(corrs[i].src.pos[0] - corrs[i].ref.pos[0], 2.0) +
                    pow(corrs[i].src.pos[1] - corrs[i].ref.pos[1], 2.0) +
                    pow(corrs[i].src.pos[2] - corrs[i].ref.pos[2], 2.0));
            }
            RMSE = sqrt(RMSE);
        }
        if (corrs.size() == 0) {
            RMSE = 9999;
        }

        //transform to eigen type
        corr_src.resize(3,corrs.size());
        corr_dst.resize(3,corrs.size());
        corr_w.resize(corrs.size());
        for (int i = 0; i < corrs.size(); ++i) {
            corr_src(0,i) = corrs[i].src.pos[0];
            corr_src(1,i) = corrs[i].src.pos[1];
            corr_src(2,i) = corrs[i].src.pos[2];
            corr_dst(0,i) = corrs[i].ref.pos[0];
            corr_dst(1,i) = corrs[i].ref.pos[1];
            corr_dst(2,i) = corrs[i].ref.pos[2];
            corr_w(i) = corrs[i].w;
        }


        delete[] ref_search_area;
    }


    void FIND_MULTI_LINK_CORRESPONDENCE(std::vector<gs::Point>& src_pts,std::vector<CORR>& corrs, 
                                        int search_half_size_pixel, double percentage_threshold,bool multi_or_not, 
                                        int max_iter, double& RMSE, bool plane_or_not) {
        //clear corrs
        corrs.clear();
        RMSE = 0;
        double utm_x, utm_y;
        int grid_x, grid_y;
        int search_xmin, search_xmax, search_ymax, search_ymin;
        int search_width, search_height;
        double* ref_search_area = new double[(2 * search_half_size_pixel + 1) * (2 * search_half_size_pixel + 1)];
        double src_pts_num = src_pts.size();
        double valid_src_pts = 0.0;
        for (int i = 0; i < src_pts.size(); ++i) {
            gs::Point src_pt= src_pts[i];
            utm_x = src_pt.pos[0];
            utm_y = src_pt.pos[1];
            grid_x = floor((utm_x - ref_utm_bbox_[0] + 0.25) / resolution_);
            grid_y = floor((ref_utm_bbox_[3] - utm_y + 0.25) / resolution_);
            if (grid_x < 0 || grid_x >= ref_width_ || grid_y < 0 || grid_y >= ref_height_) {
                continue;
            }
            if (i == 1630) {
                int a = 0;
            }
            search_xmin = grid_x - search_half_size_pixel;
            search_xmax = grid_x + search_half_size_pixel;
            search_ymin = grid_y - search_half_size_pixel;
            search_ymax = grid_y + search_half_size_pixel;
            search_xmin = (search_xmin < 0) ? 0 : search_xmin;
            search_xmax = (search_xmax >= ref_width_) ? ref_width_ - 1 : search_xmax;
            search_ymin = (search_ymin < 0) ? 0 : search_ymin;
            search_ymax = (search_ymax >= ref_height_) ? ref_height_ - 1 : search_ymax;
            search_width = search_xmax - search_xmin + 1;
            search_height = search_ymax - search_ymin + 1;


            ref_band_->RasterIO(GF_Read, search_xmin, search_ymin,
                search_width, search_height,
                ref_search_area, search_width, search_height, GDT_Float64, 0, 0);
            int best_row = 0;
            int best_col = 0;
            double best_dis = 9999;
            double cur_dis;
            double ref_x, ref_y,ref_z;
            int valid_cnt = 0;
            double valid_ratio = 0;
            double sum_guass = 0;
            double guass = 0;
            double percentage = 0;
            int idx = 0;
            vector<int> candidate_index_index;
            std::vector<pair<double, int>> candidate_index;
            for (int row = 1; row < search_height-1; ++row) {
                for (int col = 1; col < search_width-1; ++col) {
                    idx = col + row * search_width;
					if (isnan(ref_search_area[idx])) {
						continue;
					}
                    if (plane_or_not && (isnan(ref_search_area[idx-1]) || 
										isnan(ref_search_area[idx+1]) || 
										isnan(ref_search_area[idx-search_width]) || 
										isnan(ref_search_area[idx+search_width]))) {
                        continue;
                    }
                    valid_cnt++;
                    ref_x = ref_utm_bbox_[0] + (search_xmin + col) * resolution_;
                    ref_y = ref_utm_bbox_[3] - (search_ymin + row) * resolution_;
                    ref_z = ref_search_area[idx];
                    cur_dis = pow(src_pt.pos[0] - ref_x, 2.0) + pow(src_pt.pos[1] - ref_y, 2.0) + pow(src_pt.pos[2] - ref_z, 2.0);
                    guass = exp(-cur_dis);
                    //sum_guass += guass;
                    candidate_index.push_back(make_pair(guass, idx));
                    if (cur_dis < best_dis) {
                        best_row = row;
                        best_col = col;
                        best_dis = cur_dis;
                    }
                }
            }
            valid_ratio = (double)valid_cnt / (double)(search_width * search_height);
            if (valid_ratio < valid_ratio_threshold_rough_) { continue; }
            // start go multi-correspondence
            if (multi_or_not) {
                sort(candidate_index.begin(), candidate_index.end(), cmp_descend_double);
                int col, row;
                sum_guass = 0.0;
                //num_multi_link = candidate_index.size();
                for (int iter = 0; iter < candidate_index.size(); ++iter) {
                    guass = candidate_index[iter].first;
                    sum_guass += guass;
                }
                for (int iter = 0; iter < candidate_index.size(); ++iter) {
                    col = candidate_index[iter].second % search_width;
                    row = candidate_index[iter].second / search_width;
                    guass = candidate_index[iter].first;
                    percentage = guass / sum_guass;
                    if (percentage < percentage_threshold || isnan(ref_search_area[candidate_index[iter].second]) || isnan(guass)) {
                        continue;
                    }
                    candidate_index_index.push_back(iter);
                }
                sum_guass = 0.0;
                for (int iter = 0; iter < candidate_index_index.size(); ++iter) {
                    guass = candidate_index[candidate_index_index[iter]].first;
                    sum_guass+=guass;
                }
                if (sum_guass <1e-5) {
                    continue;
                }
                for (int iter = 0; iter < candidate_index_index.size(); ++iter) {
                    idx = candidate_index_index[iter];
                    col = candidate_index[idx].second % search_width;
                    row = candidate_index[idx].second / search_width;
                    ref_x = ref_utm_bbox_[0] + (search_xmin + col) * resolution_;
                    ref_y = ref_utm_bbox_[3] - (search_ymin + row) * resolution_;
                    gs::Point ref_pt(ref_x,
                        ref_y,
                        ref_search_area[candidate_index[idx].second]);
                    guass = candidate_index[idx].first;
                    CORR corr;
                    if (plane_or_not) {
                        float ref_dx, ref_dy,ref_dz,magnitude;
                        vector<float> src_norm, ref_norm;
                        ref_dx = (ref_search_area[candidate_index[idx].second + 1] - ref_search_area[candidate_index[idx].second - 1]) / 2.0;
                        ref_dy = (ref_search_area[candidate_index[idx].second - search_width] - ref_search_area[candidate_index[idx].second + search_width]) / 2.0;
                        magnitude = sqrt(pow(ref_dx,2.0) + pow(ref_dy,2) + 1);
                        ref_dx /= magnitude;
                        ref_dy /= magnitude;
                        ref_dz = 1.0 / magnitude;
                        ref_norm.push_back(ref_dx);
                        ref_norm.push_back(ref_dy);
                        ref_norm.push_back(ref_dz);
                        corr.ref = ref_pt;
                        corr.src = src_pt;
                        corr.ref_norm = ref_norm;
                        corr.w = guass / sum_guass;
                    }
                    else {
                        corr.ref = ref_pt;
                        corr.src = src_pt;
                        corr.w = guass / sum_guass;
                    }

                    //CORR corr(src_pt, ref_pt, guass / (sum_guass));
                    if (corrs.size() == 24416) {
                        int a = 0;
                    }
                    corrs.push_back(corr);
                }
                valid_src_pts++;
            }
            else {
                if (best_dis == 9999) {
                    continue;
                }
                gs::Point ref_pt(ref_utm_bbox_[0] + (search_xmin + best_col) * resolution_,
                    ref_utm_bbox_[3] - (search_ymin + best_row) * resolution_,
                    ref_search_area[best_col + best_row * search_width]);
                CORR corr;
                if (plane_or_not) {
                    float ref_dx, ref_dy, ref_dz, magnitude;
                    vector<float> src_norm, ref_norm;
                    ref_dx = (ref_search_area[best_col+1 + best_row * search_width] - ref_search_area[best_col-1 + best_row * search_width] / 2.0);
                    ref_dy = (ref_search_area[best_col + (best_row-1) * search_width] - ref_search_area[best_col + (best_row+1) * search_width] / 2.0);
                    magnitude = sqrt(pow(ref_dx, 2.0) + pow(ref_dy, 2) + 1);
                    ref_dx /= magnitude;
                    ref_dy /= magnitude;
                    ref_dz = 1.0 / magnitude;
                    ref_norm.push_back(ref_dx);
                    ref_norm.push_back(ref_dy);
                    ref_norm.push_back(ref_dz);
                    corr.ref = ref_pt;
                    corr.src = src_pt;
                    corr.ref_norm = ref_norm;
                    corr.w = 1;                 
                }
                else {
                    corr.ref = ref_pt;
                    corr.src = src_pt;
                    corr.w = 1;
                }
                corrs.push_back(corr);
                valid_src_pts++;
            }
        }
        //reweight the corr
        for (int i = 0; i < corrs.size(); ++i) {
            corrs[i].w /= valid_src_pts;
        }
        // reject outlier
        if (!multi_or_not) {
            CORR_REJECTOR1(corrs,1.0);
        }

        //compute rmse
        if (plane_or_not) {
            for (int i = 0; i < corrs.size(); ++i) {
                RMSE += corrs[i].w * (
                    pow((corrs[i].src.pos[0] - corrs[i].ref.pos[0]) * corrs[i].ref_norm[0], 2) +
                    pow((corrs[i].src.pos[1] - corrs[i].ref.pos[1]) * corrs[i].ref_norm[1], 2) +
                    pow((corrs[i].src.pos[2] - corrs[i].ref.pos[2]) * corrs[i].ref_norm[2], 2)
                    );
            }
            RMSE = sqrt(RMSE);
        }
        else {
            for (int i = 0; i < corrs.size(); ++i) {
                RMSE += corrs[i].w * (pow(corrs[i].src.pos[0] - corrs[i].ref.pos[0], 2.0) +
                    pow(corrs[i].src.pos[1] - corrs[i].ref.pos[1], 2.0) +
                    pow(corrs[i].src.pos[2] - corrs[i].ref.pos[2], 2.0));
            }
            RMSE = sqrt(RMSE);
        }
		if (corrs.size() == 0) {
			RMSE = 9999;
		}
        delete[] ref_search_area;
    }
    

    /// <summary>
    /// 
    /// </summary>
    /// <param name="X">source points, 3xN</param>
    /// <param name="Y">target points, 3xN</param>
    /// <param name="W">weight, N</param>
    /// <param name="R">estimate rotation, 3x3</param>
    /// <param name="T">estimate translation, 3</param>
    void ESTIMATE_TRANSFORMATION_WEIGHTED_CORRS(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, Eigen::VectorXd& W, 
                                                Eigen::Matrix3d& R, Eigen::Vector3d& T) {
        int dim = X.rows();
        /// Normalize weight vector
        Eigen::VectorXd w_normalized = W / W.sum();
        /// De-mean
        Eigen::VectorXd X_mean(dim), Y_mean(dim);
        for (int i = 0; i < dim; ++i) {
            X_mean(i) = (X.row(i).array() * w_normalized.transpose().array()).sum();
            Y_mean(i) = (Y.row(i).array() * w_normalized.transpose().array()).sum();
        }
        X.colwise() -= X_mean;
        Y.colwise() -= Y_mean;

        /// Compute transformation
        Eigen::MatrixXd sigma = X * w_normalized.asDiagonal() * Y.transpose();
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
            Eigen::VectorXd S = Eigen::VectorXd::Ones(dim); S(dim - 1) = -1.0;
            R = svd.matrixV() * S.asDiagonal() * svd.matrixU().transpose();
        }
        else {
            R = svd.matrixV() * svd.matrixU().transpose();
        }
        T = Y_mean - R * X_mean;
        /// Re-apply mean
        X.colwise() += X_mean;
        Y.colwise() += Y_mean;
    }

    void ESTIMATE_TRANSFORMATION_WEIGHTED_CORRS(std::vector<CORR>& corrs, double* rotation, double* translation, bool plane_or_not) {
		gs::initDiagonal(rotation);
		translation[0] = 0;
		translation[1] = 0;
		translation[2] = 0;

		if (corrs.size() < 3) {
			std::cout << "[ERROR] # correspondences less than 3" << std::endl;
			return;
		}
		// compute estimation Point-to-Plane cost function
        // ref: https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/transformation_estimation_point_to_plane_lls_weighted.hpp
        if (plane_or_not) {
            using Vector6d = Eigen::Matrix<double, 6, 1>;
            using Matrix6d = Eigen::Matrix<double, 6, 6>;

            Matrix6d ATA;
            Vector6d ATb;
            ATA.setZero();
            ATb.setZero();

            for (int i=0;i<corrs.size();++i){


                const float& sx = corrs[i].src.pos[0];
                const float& sy = corrs[i].src.pos[1];
                const float& sz = corrs[i].src.pos[2];
                const float& dx = corrs[i].ref.pos[0];
                const float& dy = corrs[i].ref.pos[1];
                const float& dz = corrs[i].ref.pos[2];
                const float& nx = corrs[i].ref_norm[0] * corrs[i].w;
                const float& ny = corrs[i].ref_norm[1] * corrs[i].w;
                const float& nz = corrs[i].ref_norm[2] * corrs[i].w;

                double a = nz * sy - ny * sz;
                double b = nx * sz - nz * sx;
                double c = ny * sx - nx * sy;

                //    0  1  2  3  4  5
                //    6  7  8  9 10 11
                //   12 13 14 15 16 17
                //   18 19 20 21 22 23
                //   24 25 26 27 28 29
                //   30 31 32 33 34 35

                ATA.coeffRef(0) += a * a;
                ATA.coeffRef(1) += a * b;
                ATA.coeffRef(2) += a * c;
                ATA.coeffRef(3) += a * nx;
                ATA.coeffRef(4) += a * ny;
                ATA.coeffRef(5) += a * nz;
                ATA.coeffRef(7) += b * b;
                ATA.coeffRef(8) += b * c;
                ATA.coeffRef(9) += b * nx;
                ATA.coeffRef(10) += b * ny;
                ATA.coeffRef(11) += b * nz;
                ATA.coeffRef(14) += c * c;
                ATA.coeffRef(15) += c * nx;
                ATA.coeffRef(16) += c * ny;
                ATA.coeffRef(17) += c * nz;
                ATA.coeffRef(21) += nx * nx;
                ATA.coeffRef(22) += nx * ny;
                ATA.coeffRef(23) += nx * nz;
                ATA.coeffRef(28) += ny * ny;
                ATA.coeffRef(29) += ny * nz;
                ATA.coeffRef(35) += nz * nz;

                double d = nx * dx + ny * dy + nz * dz - nx * sx - ny * sy - nz * sz;
                ATb.coeffRef(0) += a * d;
                ATb.coeffRef(1) += b * d;
                ATb.coeffRef(2) += c * d;
                ATb.coeffRef(3) += nx * d;
                ATb.coeffRef(4) += ny * d;
                ATb.coeffRef(5) += nz * d;
            }

            ATA.coeffRef(6) = ATA.coeff(1);
            ATA.coeffRef(12) = ATA.coeff(2);
            ATA.coeffRef(13) = ATA.coeff(8);
            ATA.coeffRef(18) = ATA.coeff(3);
            ATA.coeffRef(19) = ATA.coeff(9);
            ATA.coeffRef(20) = ATA.coeff(15);
            ATA.coeffRef(24) = ATA.coeff(4);
            ATA.coeffRef(25) = ATA.coeff(10);
            ATA.coeffRef(26) = ATA.coeff(16);
            ATA.coeffRef(27) = ATA.coeff(22);
            ATA.coeffRef(30) = ATA.coeff(5);
            ATA.coeffRef(31) = ATA.coeff(11);
            ATA.coeffRef(32) = ATA.coeff(17);
            ATA.coeffRef(33) = ATA.coeff(23);
            ATA.coeffRef(34) = ATA.coeff(29);

            // Solve A*x = b
            Vector6d x = static_cast<Vector6d>(ATA.inverse() * ATb);

            // Construct the transformation matrix from x
            Eigen::Matrix4d transformation_matrix;
            double alpha, beta, gamma, tx, ty, tz;
            alpha = x(0);
            beta = x(1);
            gamma = x(2);
            tx = x(3);
            ty = x(4);
            tz = x(5);
            transformation_matrix = Eigen::Matrix<double, 4, 4>::Zero();
            rotation[0] = static_cast<double>(std::cos(gamma) * std::cos(beta));
            rotation[1] = static_cast<double>(
                -sin(gamma) * std::cos(alpha) + std::cos(gamma) * sin(beta) * sin(alpha));
            rotation[2] = static_cast<double>(
                sin(gamma) * sin(alpha) + std::cos(gamma) * sin(beta) * std::cos(alpha));
            rotation[3] = static_cast<double>(sin(gamma) * std::cos(beta));
            rotation[4] = static_cast<double>(
                std::cos(gamma) * std::cos(alpha) + sin(gamma) * sin(beta) * sin(alpha));
            rotation[5] = static_cast<double>(
                -std::cos(gamma) * sin(alpha) + sin(gamma) * sin(beta) * std::cos(alpha));
            rotation[6] = static_cast<double>(-sin(beta));
            rotation[7] = static_cast<double>(std::cos(beta) * sin(alpha));
            rotation[8] = static_cast<double>(std::cos(beta) * std::cos(alpha));

            translation[0] = static_cast<double>(tx);
            translation[1] = static_cast<double>(ty);
            translation[2] = static_cast<double>(tz);
        }
        // compute estimation Point-to-Point cost function
        else {
            gs::Point dynamicMid(0, 0, 0), d_raw, s_raw, d_delta, s_delta;
            gs::Point staticMid(0, 0, 0);
            for (int i = 0; i < corrs.size(); ++i) {
                dynamicMid.pos[0] = dynamicMid.pos[0] + corrs[i].w * corrs[i].src.pos[0];
                dynamicMid.pos[1] = dynamicMid.pos[1] + corrs[i].w * corrs[i].src.pos[1];
                dynamicMid.pos[2] = dynamicMid.pos[2] + corrs[i].w * corrs[i].src.pos[2];
                staticMid.pos[0] = staticMid.pos[0] + corrs[i].w * corrs[i].ref.pos[0];
                staticMid.pos[1] = staticMid.pos[1] + corrs[i].w * corrs[i].ref.pos[1];
                staticMid.pos[2] = staticMid.pos[2] + corrs[i].w * corrs[i].ref.pos[2];
            }
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
            gs::clearMatrix(U);
            gs::clearMatrix(V);
            gs::clearMatrix(w);
            for (int i = 0; i < corrs.size(); i++)
            {
                d_raw = corrs[i].src;
                s_raw = corrs[i].ref;
                d_delta = d_raw - dynamicMid;
                s_delta = s_raw - staticMid;

                //outerProduct(&d_delta, &s_delta, w);
                gs::outerProduct(&s_delta, &d_delta, w);
                w[0] *= corrs[i].w;
                w[1] *= corrs[i].w;
                w[2] *= corrs[i].w;
                w[3] *= corrs[i].w;
                w[4] *= corrs[i].w;
                w[5] *= corrs[i].w;
                w[6] *= corrs[i].w;
                w[7] *= corrs[i].w;
                w[8] *= corrs[i].w;
                gs::addMatrix(w, U, U);
            }
            gs::copyMatToUV(U, uSvd);
            dsvd(uSvd, 3, 3, sigma, vSvd);
            gs::copyUVtoMat(uSvd, U);
            gs::copyUVtoMat(vSvd, V);

            /// Compute Rotation matrix
            gs::initDiagonal(Diagonal);
            if (gs::determinant_3by3(uSvd) * gs::determinant_3by3(vSvd) < 0) {
                Diagonal[8] = -1;
            }
            //transpose(U);
            gs::transpose(V);
            double tmp[9];
            double tmp_vec3[3];
            gs::clearMatrix(tmp);
            //matrixMult(V, Diagonal,tmp);
            //matrixMult(tmp, U, rotationMatrix);
            gs::matrixMult(U, Diagonal, tmp);
            gs::matrixMult(tmp, V, rotation);

            gs::Point t(0.0, 0.0, 0.0);
            gs::rotate(&dynamicMid, rotation, &t);
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

    void CORR_REJECTOR(std::vector<pair<gs::Point, gs::Point>>& corrs) {
        int num_trimmed = (int)corrs.size() * trimmed_ratio_;
        std::vector<pair<gs::Point, gs::Point>> corrs_tmp;
        std::vector<std::pair<float, int>> corr_dis_index;
        float dis;
        for (int i = 0; i < corrs.size(); ++i) {
            dis = std::sqrt(pow(corrs[i].first.pos[0] - corrs[i].second.pos[0], 2.0) +
                pow(corrs[i].first.pos[1] - corrs[i].second.pos[1], 2.0) +
                pow(corrs[i].first.pos[2] - corrs[i].second.pos[2], 2.0));
            corr_dis_index.push_back(std::make_pair(dis, i));
        }
        std::sort(corr_dis_index.begin(), corr_dis_index.end(), cmp_ascend);
        for (int i = 0; i < num_trimmed; ++i) {
            corrs_tmp.push_back(corrs[corr_dis_index[i].second]);
        }
        corrs.clear();
        corrs = corrs_tmp;
    }
    void CORR_REJECTOR1(std::vector<CORR>& corrs,double trimmed_ratio) {
        int num_trimmed = (int)corrs.size() * trimmed_ratio;
        std::vector<CORR> corrs_tmp;
        std::vector<std::pair<float, int>> corr_dis_index;
        float dis;
        for (int i = 0; i < corrs.size(); ++i) {
            dis = std::sqrt(pow(corrs[i].src.pos[0] - corrs[i].ref.pos[0], 2.0) +
                pow(corrs[i].src.pos[1] - corrs[i].ref.pos[1], 2.0) +
                pow(corrs[i].src.pos[2] - corrs[i].ref.pos[2], 2.0));
            corr_dis_index.push_back(std::make_pair(dis, i));
        }
        std::sort(corr_dis_index.begin(), corr_dis_index.end(), cmp_ascend);
        for (int i = 0; i < num_trimmed; ++i) {
            corrs_tmp.push_back(corrs[corr_dis_index[i].second]);
        }
        corrs.clear();
        corrs = corrs_tmp;
    }
    
    void START(std::string debug_dir) {
        print_info();
        STEP1_1_ROUGH_ALIGN(debug_dir);
        STEP2_1_COLLECT_TILE_INFOS();
        STEP2_2_FILTER_TILE_INFOS(debug_dir);
        STEP2_3_FINE_ALIGN(debug_dir);
        STEP_END();
    }

    void STEP1_1_ROUGH_ALIGN_old(string out_path) {
        std::cout << "\n[STEP1_1] Rough Align" << std::endl;
        // Sample src pts
        double margin_ratio = 0.1;
        int aoi_src_bbox_pixel_margin[4];
        int margin_width = (aoi_src_bbox_pixel_[1] - aoi_src_bbox_pixel_[0]) * margin_ratio / 2;
        int margin_height = (aoi_src_bbox_pixel_[2] - aoi_src_bbox_pixel_[3]) * margin_ratio / 2;
        aoi_src_bbox_pixel_margin[0] = aoi_src_bbox_pixel_[0] + margin_width;
        aoi_src_bbox_pixel_margin[1] = aoi_src_bbox_pixel_[1] - margin_width;
        aoi_src_bbox_pixel_margin[2] = aoi_src_bbox_pixel_[2] - margin_height;
        aoi_src_bbox_pixel_margin[3] = aoi_src_bbox_pixel_[3] + margin_height;

        double widht_height_ratio = (aoi_utm_bbox_[1] - aoi_utm_bbox_[0]) / (aoi_utm_bbox_[3] - aoi_utm_bbox_[2]);
        int grid_height = ceil(std::sqrt((double)rough_num_pts_ / widht_height_ratio));
        int grid_width = (int)((double)grid_height * widht_height_ratio);
        int grid_step_x_pixel = (aoi_src_bbox_pixel_margin[1] - aoi_src_bbox_pixel_margin[0]) / grid_width;
        int grid_step_y_pixel = (aoi_src_bbox_pixel_margin[2] - aoi_src_bbox_pixel_margin[3])/ grid_height;
        int grid_x, grid_y;
        double utm_x, utm_y;
        double* pts_z = new double[1];
        for (int row = 0; row < grid_height; ++row) {
            for (int col = 0; col < grid_width; ++col) {
                grid_x = aoi_src_bbox_pixel_margin[0] + col * grid_step_x_pixel;
                grid_y = aoi_src_bbox_pixel_margin[3] + row * grid_step_y_pixel;
                utm_x = src_utm_bbox_[0] + grid_x * resolution_;
                utm_y = src_utm_bbox_[3] - grid_y * resolution_;
                src_band_->RasterIO(GF_Read, grid_x, grid_y, 1, 1, pts_z, 1, 1, GDT_Float64, 0, 0);
                if (isnan(*pts_z)) {
                    continue;
                }
                rough_src_index_.push_back(make_pair(grid_x, grid_y));
                gs::Point src_pt(utm_x, utm_y, *pts_z);
                rough_src_pts_.push_back(src_pt);

            }
        }
        delete[] pts_z;
        // Find correspondence on reference data
        std::vector<pair<gs::Point, gs::Point>> corrs;
        double corr_rmse = 0;
        ROUGH_FIND_CORRESPONDENCE(corrs, corr_rmse);

        //write_ptcloud_corrs(corrs, "N:\\tasks\\reg\\ref_pts.txt", "N:\\tasks\\reg\\src_pts.txt");
        //Estmate Transformation
        
        for (int iter = 0; iter < rough_icp_max_iter_; ++iter) {
            std::cout << "Rough Align #Iteration: " << iter <<"# corrs: "<<corrs.size() << " RMSE: " << corr_rmse << std::endl;
            double rotation[9];
            double translation[3];
            ESTIMATE_TRANSFORMATION_CORRS(corrs, rotation, translation);
            double tmp[9];
            double tmp_vec3[3];
            gs::matrixMult(rotation, rough_rotation_, tmp);
            gs::rotate_mat(rough_translation_, rotation, tmp_vec3);
            rough_translation_[0] = tmp_vec3[0] + translation[0];
            rough_translation_[1] = tmp_vec3[1] + translation[1];
            rough_translation_[2] = tmp_vec3[2] + translation[2];
            std::copy(&tmp[0], &tmp[9], &rough_rotation_[0]);

            // Update source pts
            update_ptcloud(rough_src_pts_, rotation, translation, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_);

            // FInd correspondence
            double pre_rmse = corr_rmse;
            ROUGH_FIND_CORRESPONDENCE(corrs, corr_rmse);
            if (abs(pre_rmse - corr_rmse) < rough_icp_rmse_threshold_) {
                break;
            }
        }
        
        //print
        std::cout << "\nRough rotation: " << rough_rotation_[0] << "," <<
            rough_rotation_[1] << "," <<
            rough_rotation_[2] << "," <<
            rough_rotation_[3] << "," <<
            rough_rotation_[4] << "," <<
            rough_rotation_[5] << "," <<
            rough_rotation_[6] << "," <<
            rough_rotation_[7] << "," <<
            rough_rotation_[8] << "," << "\n"<<
            "Rough translation" << rough_translation_[0] << "," <<
            rough_translation_[1] << "," <<
            rough_translation_[2] << endl;
        if (out_path != "") {
            ofstream ofs(out_path);
            ofs << rough_rotation_[0] << " " <<
                rough_rotation_[1] << " " <<
                rough_rotation_[2] << " " <<
                rough_translation_[0] << "\n" <<
                rough_rotation_[3] << " " <<
                rough_rotation_[4] << " " <<
                rough_rotation_[5] << " " <<
                rough_translation_[0] << "\n" <<
                rough_rotation_[6] << " " <<
                rough_rotation_[7] << " " <<
                rough_rotation_[8] << " " <<
                rough_translation_[0] << "\n"<<
                0<<" "<<0<<" "<<0<<" "<<1;
            ofs.close();
        }
    }
    void STEP1_1_ROUGH_ALIGN(string dir_path) {
        std::cout << "\n[STEP1_1] Rough Align" << std::endl;
        // Sample src pts
        double margin_ratio = 0.1;
        int aoi_src_bbox_pixel_margin[4];
        int margin_width = (aoi_src_bbox_pixel_[1] - aoi_src_bbox_pixel_[0]) * margin_ratio / 2;
        int margin_height = (aoi_src_bbox_pixel_[2] - aoi_src_bbox_pixel_[3]) * margin_ratio / 2;
        aoi_src_bbox_pixel_margin[0] = aoi_src_bbox_pixel_[0] + margin_width;
        aoi_src_bbox_pixel_margin[1] = aoi_src_bbox_pixel_[1] - margin_width;
        aoi_src_bbox_pixel_margin[2] = aoi_src_bbox_pixel_[2] - margin_height;
        aoi_src_bbox_pixel_margin[3] = aoi_src_bbox_pixel_[3] + margin_height;

        double widht_height_ratio = (aoi_utm_bbox_[1] - aoi_utm_bbox_[0]) / (aoi_utm_bbox_[3] - aoi_utm_bbox_[2]);
        int grid_height = ceil(std::sqrt((double)rough_num_pts_ / widht_height_ratio));
        int grid_width = (int)((double)grid_height * widht_height_ratio);
        int grid_step_x_pixel = (aoi_src_bbox_pixel_margin[1] - aoi_src_bbox_pixel_margin[0]) / grid_width;
        int grid_step_y_pixel = (aoi_src_bbox_pixel_margin[2] - aoi_src_bbox_pixel_margin[3]) / grid_height;
        int grid_x, grid_y;
        double utm_x, utm_y;
        double* pts_z = new double[1];
        for (int row = 0; row < grid_height; ++row) {
            for (int col = 0; col < grid_width; ++col) {
                grid_x = aoi_src_bbox_pixel_margin[0] + col * grid_step_x_pixel;
                grid_y = aoi_src_bbox_pixel_margin[3] + row * grid_step_y_pixel;
                utm_x = src_utm_bbox_[0] + grid_x * resolution_;
                utm_y = src_utm_bbox_[3] - grid_y * resolution_;
                src_band_->RasterIO(GF_Read, grid_x, grid_y, 1, 1, pts_z, 1, 1, GDT_Float64, 0, 0);
                if (isnan(*pts_z)) {
                    continue;
                }
                rough_src_index_.push_back(make_pair(grid_x, grid_y));
                gs::Point src_pt(utm_x, utm_y, *pts_z);
                rough_src_pts_.push_back(src_pt);
            }
        }
        delete[] pts_z;

        // Rough ICP : Adjust Rotation, Z
        string rough_icp_log = dir_path + "rough_icp.log";
        string rough_trans_txt = dir_path + "rough_icp_trans.txt";
        double rough_rmse;
		gs::initDiagonal(rough_rotation_);
		rough_translation_[0] = 0;
		rough_translation_[1] = 0;
		rough_translation_[2] = 0;
        MULTI_ICP(rough_src_pts_, search_half_size_pixel_,rough_icp_percentage_threshold_, multi_or_not_, rough_icp_max_iter_, rough_icp_rmse_threshold_,rough_rotation_,rough_translation_,plane_or_not_, rough_rmse,rough_icp_log);

        // If no brute force search on X_Y plane, continue to the fine registration
        if (!brute_force_search_or_not_) {
            rough_translations_.clear();
            vector<double> trans;
            trans.push_back(rough_translation_[0]);
            trans.push_back(rough_translation_[1]);
            trans.push_back(rough_translation_[2]);
            rough_translations_.push_back(trans);
            return;
        }
        // Brute-Force search on x-y plane
		std::cout << "Brute-Force seach on XY-plane" << std::endl;
        string xy_search_log = dir_path + "xysearch.log";
        vector<double> rough_grid_rmses;
        vector<pair<int, int>> rough_grid_idx;
        vector<pair<double, double>> rough_trans_xy_list;
        double rot[9];
        double grid_trans[3];
        gs::initDiagonal(rot);
        grid_trans[0] = 0.0;
        grid_trans[1] = 0.0;
        grid_trans[2] = 0.0;
        grid_x = (int)((xmax_m_ - xmin_m_) / rough_step_m_) + 1;
        grid_y = (int)((ymax_m_ - ymin_m_) / rough_step_m_) + 1;
        std::vector<CORR> corrs;
        double** corr_rmses = new double*[grid_y];
        for (int i = 0; i < grid_y; ++i) {
            corr_rmses[i] = new double[grid_x];
        }
        ofstream ofs;
        double corr_rmse = 0;
        if (dir_path != "" && verbose_ > 0) {
            ofs.open(xy_search_log);
        }
        // first apply rough_trans to rough_src_pts_
		std::cout << "Search grid(x,y) : " << grid_x << "," << grid_y << std::endl;
        update_ptcloud(rough_src_pts_, rough_rotation_, rough_translation_, 0, 0, 0, 0);
        for (int cnt_x = 0; cnt_x < grid_x; ++cnt_x) {
            for (int cnt_y = 0; cnt_y < grid_y; ++cnt_y) {
                if (cnt_x == 0 && cnt_y == 0) {
                    grid_trans[0] = xmin_m_;
                    grid_trans[1] = ymin_m_;
                }
                else if (cnt_y == 0) {
                    grid_trans[0] = rough_step_m_;
                    grid_trans[1] = -(grid_y - 1) * rough_step_m_;
                }
                else {
                    grid_trans[0] = 0.0;
                    grid_trans[1] = rough_step_m_;
                }
                rough_trans_xy_list.push_back(make_pair(xmin_m_+cnt_x*rough_step_m_, ymin_m_ + cnt_y * rough_step_m_));
                rough_grid_idx.push_back(make_pair(cnt_x, cnt_y));
                update_ptcloud(rough_src_pts_, rot, grid_trans, 0, 0, 0, 0);
                FIND_MULTI_LINK_CORRESPONDENCE(rough_src_pts_, corrs, search_half_size_pixel_, rough_icp_percentage_threshold_, multi_or_not_, rough_icp_max_iter_, corr_rmse,plane_or_not_);
                rough_grid_rmses.push_back(corr_rmse);
                corr_rmses[cnt_y][cnt_x] = corr_rmse;
                //cout << setprecision(11) << "Grid X,Y: " << cnt_x << "," << cnt_y << ", RMSE: " << corr_rmse << std::endl;
                if (dir_path != "" && verbose_ > 0) { ofs << setprecision(11) << cnt_x << " " << cnt_y << " " << corr_rmse << std::endl; }
            }
        }
        if (dir_path != "" && verbose_ > 0) { ofs.close(); }
        
        // Non-max supression
        vector<pair<int, int>> mins_ids;
        non_min_suppresion(grid_x, grid_y, corr_rmses, rough_step_min_resolution_, mins_ids);
        for (int i = 0; i < grid_y; ++i) {
            delete[] corr_rmses[i];
        }
        delete[] corr_rmses;

        // Generate Rough translation candidate
        for (int i = 0; i < mins_ids.size(); ++i) {
			std::cout << "Top #" << i << " grid (x,y): " << mins_ids[i].second << "," << mins_ids[i].first << std::endl;
            vector<double> trans;
            trans.push_back(xmin_m_ + mins_ids[i].second * rough_step_m_+rough_translation_[0]);
            trans.push_back(ymin_m_ + mins_ids[i].first * rough_step_m_+rough_translation_[1]);
            trans.push_back(rough_translation_[2]);
            rough_translations_.push_back(trans);
			std::cout << "Rough Align Translation candidate: " << std::endl;
			std::cout << "#" << i << ": " << trans[0] << "," << trans[1] << "," << trans[2] << std::endl;
			if (i >= rough_num_mins_-1) {
				break;
			}
        }
    }

    void STEP2_1_COLLECT_TILE_INFOS() {
        std::cout << "\n[STEP2_1] Collect Tile Infos" << std::endl;
        std::cout << "Total Num of Tiles: " << num_tile_x_ * num_tile_y_ << std::endl;
#pragma omp parallel
        {
            int tile_id = 0;
            double tile_utm_bbox[4];
            int tile_ref_bbox[4];
            int tile_src_bbox[4];
            vector<TILE_INFO> tile_infos_private;
#pragma omp for collapse(2) nowait
            for (int row = 0; row < num_tile_y_; ++row) {
                for (int col = 0; col < num_tile_x_; ++col) {

                    TILE_INFO tile_info;
                    tile_id = col + row * num_tile_x_;
                    //std::cout << "Tiles:" << tile_id << std::endl;
                    tile_utm_bbox[0] = aoi_utm_bbox_[0] + col * tilesize_m_;
                    tile_utm_bbox[1] = aoi_utm_bbox_[0] + (col + 1) * tilesize_m_;
                    tile_utm_bbox[2] = aoi_utm_bbox_[3] - (row + 1) * tilesize_m_;
                    tile_utm_bbox[3] = aoi_utm_bbox_[3] - row * tilesize_m_;

                    get_aoi_bbox_pixels(tile_utm_bbox, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, tile_ref_bbox, tile_src_bbox);

                    double* ref_data = new double[(tile_ref_bbox[1] - tile_ref_bbox[0] + 1) * (tile_ref_bbox[2] - tile_ref_bbox[3] + 1)];
                    double* src_data = new double[(tile_src_bbox[1] - tile_src_bbox[0] + 1) * (tile_src_bbox[2] - tile_src_bbox[3] + 1)];
                    ref_band_->RasterIO(GF_Read, tile_ref_bbox[0],
                        tile_ref_bbox[3],
                        tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                        tile_ref_bbox[2] - tile_ref_bbox[3] + 1,
                        ref_data,
                        tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                        tile_ref_bbox[2] - tile_ref_bbox[3] + 1, GDT_Float64, 0, 0);
                    src_band_->RasterIO(GF_Read, tile_src_bbox[0],
                        tile_src_bbox[3],
                        tile_src_bbox[1] - tile_src_bbox[0] + 1,
                        tile_src_bbox[2] - tile_src_bbox[3] + 1,
                        src_data,
                        tile_src_bbox[1] - tile_src_bbox[0] + 1,
                        tile_src_bbox[2] - tile_src_bbox[3] + 1, GDT_Float64, 0, 0);
                    


                    tile_info.utm_bbox[0] = tile_utm_bbox[0]+ GLOBAL_OFFSET_X_m_;
                    tile_info.utm_bbox[1] = tile_utm_bbox[1]+ GLOBAL_OFFSET_X_m_;
                    tile_info.utm_bbox[2] = tile_utm_bbox[2]+ GLOBAL_OFFSET_Y_m_;
                    tile_info.utm_bbox[3] = tile_utm_bbox[3]+ GLOBAL_OFFSET_Y_m_;
                    tile_info.tile_id = tile_id;
                    double ref_valid_ratio;
                    double src_valid_ratio;
                    if (tile_info.tile_id == 564) {
                        int a = 0;
                    }
                    compute_tile_var(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, tile_info.ref_var, tile_info.ref_valid);
                    compute_tile_var(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, tile_info.src_var, tile_info.src_valid);
                    //HARRIS(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1,HARRIS_step_,HARRIS_k_, tile_info.ref_lambda1, tile_info.ref_lambda2, tile_info.ref_R);
                    //HARRIS(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1,HARRIS_step_,HARRIS_k_, tile_info.src_lambda1, tile_info.src_lambda2, tile_info.src_R);
                    delete[] ref_data;
                    delete[] src_data;
                    tile_infos_private.push_back(tile_info);
                }
            }
#pragma omp critical
            {
                for (int i = 0; i < tile_infos_private.size(); ++i) {
                    tile_infos_all_.push_back(tile_infos_private[i]);

                }
            }
        }
    }
    void STEP2_2_FILTER_TILE_INFOS_old() {
        std::cout << "\n[STEP2_2] Filter out tiles" << std::endl;
        //write_shp(tile_infos,shp_out_all_path);
        vector<TILE_INFO> tile_infos_only, tile_infos_icp;
        vector<pair<float, int>> tiles_var_index;
        vector<pair<float, int>> tiles_similar_index;
        vector<pair<float, int>> tiles_sum_index;
        for (int i = 0; i < tile_infos_all_.size(); ++i) {
            if (tile_infos_all_[i].ref_valid < valid_ratio_threshold_ || tile_infos_all_[i].src_valid < valid_ratio_threshold_ || 
                tile_infos_all_[i].src_var < var_min_ || tile_infos_all_[i].ref_var < var_min_) {
                tiles_sum_index.push_back(make_pair(0, i));
                continue;
            }
            tile_infos_filter_.push_back(tile_infos_all_[i]);
            tiles_var_index.push_back(make_pair(tile_infos_all_[i].ref_var + tile_infos_all_[i].src_var, i));
            tiles_similar_index.push_back(make_pair(abs(tile_infos_all_[i].ref_var - tile_infos_all_[i].src_var), i));
            tiles_sum_index.push_back(make_pair(0, i));
        }
        sort(tiles_var_index.begin(), tiles_var_index.end(), cmp_descend);
        sort(tiles_similar_index.begin(), tiles_similar_index.end(), cmp_ascend);
        for (int iter = 0; iter < tiles_var_index.size(); ++iter) {
            tiles_sum_index[tiles_var_index[iter].second].first += iter+1;
        }
        for (int iter = 0; iter < tiles_similar_index.size(); ++iter) {
            tiles_sum_index[tiles_similar_index[iter].second].first += iter+1;
        }
        sort(tiles_sum_index.begin(), tiles_sum_index.end(), cmp_ascend);
        for (int i = 0; i < tiles_sum_index.size(); ++i) {
            if (tiles_sum_index[i].first == 0) {
                continue;
            }
            //tile_infos_icp.push_back(tile_infos[tiles_only_index[i].second]);
            tile_infos_icp_.push_back(tile_infos_all_[tiles_sum_index[i].second]);
            if (tile_infos_icp_.size() >= num_filter_tiles_) {
                break;
            }
        }
        std::cout << "# Pre-Filtered Tiles: " << tile_infos_filter_.size() << std::endl;
        std::cout << "# Filtered Tiles: "<<tile_infos_icp_.size() << std::endl;
        write_shp(tile_infos_filter_, "N:\\tasks\\reg\\rough_reg1\\tiles_filter_100m.shp");
        write_shp(tile_infos_icp_, "N:\\tasks\\reg\\rough_reg1\\tiles_icp_100m.shp");
        write_shp(tile_infos_all_, "N:\\tasks\\reg\\rough_reg1\\tiles_all_100m.shp");
        //tiles_only_index.clear();
    }
    void STEP2_2_FILTER_TILE_INFOS(std::string debug_dir) {
        std::cout << "\n[STEP2_2] Filter out tiles" << std::endl;
        int num_tiles_each_block = ceil(num_filter_tiles_ / (n_blocks_col_ * n_blocks_row_));
        int num_tiles_each_block_col = ceil((float)num_tile_x_/(float)n_blocks_col_);
        int num_tiles_each_block_row = ceil((float)num_tile_y_ / (float)n_blocks_row_);
        
        int tile_col_min, tile_col_max, tile_row_min, tile_row_max,tile_local_id,col_local,row_local;
        vector<pair<float, int>> tiles_var_index;
        vector<pair<float, int>> tiles_similar_index;
        vector<pair<float, int>> tiles_sum_index;
        map<int, int> local_tile_index; // map local tile id to global tile id
        int tile_id;
        // Iterate each block
		std::cout << "Num of Blocks (x,y)" << n_blocks_col_ << "," << n_blocks_row_ << std::endl;
		std::cout << "Num tiles each blcoks should be " << num_tiles_each_block << std::endl;
        for (int b_col = 0; b_col < n_blocks_col_; ++b_col) {
            for (int b_row = 0; b_row < n_blocks_row_; ++b_row) {
                tile_col_min = b_col * num_tiles_each_block_col;
                tile_col_max= (b_col+1) * num_tiles_each_block_col;
                tile_row_min = b_row * num_tiles_each_block_row;
                tile_row_max = (b_row + 1) * num_tiles_each_block_row;
                tile_col_max = (tile_col_max > num_tile_x_) ? num_tile_x_ : tile_col_max;
                tile_row_max = (tile_row_max > num_tile_y_) ? num_tile_y_ : tile_row_max;

                tiles_var_index.clear();
                tiles_similar_index.clear();
                tiles_sum_index.clear();
                local_tile_index.clear();
                tile_local_id = 0;
                // Iterate each tiles inside the block
                for (int tile_col = tile_col_min; tile_col < tile_col_max; ++tile_col) {
                    for (int tile_row = tile_row_min; tile_row < tile_row_max; ++tile_row) {
                        tile_id = tile_col + tile_row * num_tile_x_;
                        if (tile_infos_all_[tile_id].ref_valid < valid_ratio_threshold_ || tile_infos_all_[tile_id].src_valid < valid_ratio_threshold_ ||
                            tile_infos_all_[tile_id].src_var < var_min_ || tile_infos_all_[tile_id].ref_var < var_min_) {
                            tiles_sum_index.push_back(make_pair(0, tile_local_id));
                            local_tile_index.insert(make_pair(tile_local_id, tile_id));
                            tile_local_id++;
                            continue;
                        }
                        tile_infos_filter_.push_back(tile_infos_all_[tile_id]);
                        tiles_var_index.push_back(make_pair(tile_infos_all_[tile_id].ref_var + tile_infos_all_[tile_id].src_var, tile_local_id));
                        tiles_similar_index.push_back(make_pair(abs(tile_infos_all_[tile_id].ref_var - tile_infos_all_[tile_id].src_var), tile_local_id));
                        tiles_sum_index.push_back(make_pair(0, tile_local_id));
                        local_tile_index.insert(make_pair(tile_local_id, tile_id));
                        tile_local_id++;
                    }
                }
                sort(tiles_var_index.begin(), tiles_var_index.end(), cmp_descend);
                sort(tiles_similar_index.begin(), tiles_similar_index.end(), cmp_ascend);
                for (int iter = 0; iter < tiles_var_index.size(); ++iter) {
                    tiles_sum_index[tiles_var_index[iter].second].first += iter + 1;
                }
                for (int iter = 0; iter < tiles_similar_index.size(); ++iter) {
                    tiles_sum_index[tiles_similar_index[iter].second].first += iter + 1;
                }
                sort(tiles_sum_index.begin(), tiles_sum_index.end(), cmp_ascend);
                int cnt = 0;
                for (int i = 0; i < tiles_sum_index.size(); ++i) {
                    if (tiles_sum_index[i].first == 0) {
                        continue;
                    }
                    //tile_infos_icp.push_back(tile_infos[tiles_only_index[i].second]);
                    tile_infos_icp_.push_back(tile_infos_all_[local_tile_index[tiles_sum_index[i].second]]);
                    if (cnt >= num_tiles_each_block) {
                        break;
                    }
                    cnt++;
                }

            }
        }
        std::string shp_all_path = debug_dir + "tiles_all.shp";
        std::string shp_icp_path = debug_dir + "tiles_for_fine.shp";

        if (debug_dir != "" && verbose_ > 0) { 
            write_shp(tile_infos_all_, shp_all_path);
            write_shp(tile_infos_icp_, shp_icp_path);
        }
        std::cout << "Final # Filtered Tiles: " << tile_infos_icp_.size() << std::endl;
    }

    void STEP2_3_ICP_TILE() {
        std::cout << "\n[STEP2_3] Fine align tiles" << std::endl;
        std::vector<TILE_INFO> tile_infos_tmp;
#pragma omp parallel
        {
            double tile_utm_bbox[4];
            int tile_ref_bbox[4];
            int tile_src_bbox[4];
            gs::ICP_Para icp_para;
            icp_para.trans_type = gs::SHIFT;
            icp_para.rmse_threshold = 1e-5;
            icp_para.sample_ratio = 1;
            icp_para.max_distance = 6;
            icp_para.trans_type = gs::SHIFT;
            icp_para.reject_type = gs::Trimmed_Distance;
            icp_para.max_iterations = 7;
            vector<TILE_INFO> tile_infos_icp_private;
#pragma omp for nowait
            for (int i = 0; i < tile_infos_icp_.size(); ++i) {
                //std::cout << i << std::endl;
                //tile_infos_icp.push_back(tile_infos[tiles_only_index[i].second]);
                TILE_INFO tile_info = tile_infos_icp_[i];
                int col = tile_info.tile_id % num_tile_x_;
                int row = (tile_info.tile_id - col) / num_tile_x_;
                tile_utm_bbox[0] = aoi_utm_bbox_[0] + col * tilesize_m_;
                tile_utm_bbox[1] = aoi_utm_bbox_[0] + (col + 1) * tilesize_m_;
                tile_utm_bbox[2] = aoi_utm_bbox_[3] - (row + 1) * tilesize_m_;
                tile_utm_bbox[3] = aoi_utm_bbox_[3] - row * tilesize_m_;
                get_aoi_bbox_pixels(tile_utm_bbox, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, tile_ref_bbox, tile_src_bbox);

                double* ref_data = new double[(tile_ref_bbox[1] - tile_ref_bbox[0] + 1) * (tile_ref_bbox[2] - tile_ref_bbox[3] + 1)];
                double* src_data = new double[(tile_src_bbox[1] - tile_src_bbox[0] + 1) * (tile_src_bbox[2] - tile_src_bbox[3] + 1)];
                ref_band_->RasterIO(GF_Read, tile_ref_bbox[0],
                    tile_ref_bbox[3],
                    tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                    tile_ref_bbox[2] - tile_ref_bbox[3] + 1,
                    ref_data,
                    tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                    tile_ref_bbox[2] - tile_ref_bbox[3] + 1, GDT_Float64, 0, 0);
                src_band_->RasterIO(GF_Read, tile_src_bbox[0],
                    tile_src_bbox[3],
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1,
                    src_data,
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1, GDT_Float64, 0, 0);

                // write before rough align ptcloud
                stringstream ss;
                string tile_dir = "N:\\tasks\\reg\\rough_reg\\tile_rough\\";
                string ref_out_path;
                string src_out_path;
                //ss << tile_dir << "tile_" << tile_info.tile_id << "ref.txt";
                //string ref_out_path = ss.str();
                //ss.str("");
                //ss << tile_dir << "tile_" << tile_info.tile_id << "src_raw.txt";
                //string src_out_path = ss.str();
                //write_ptcloud(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, 0.5, ref_out_path);
                //write_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, 0.5, src_out_path, 
                //    tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);

                vector<gs::Point> d_ptcloud;
                vector<gs::Point> s_ptcloud;
                convert_ptcloud(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, resolution_, fine_resolution_factor_, s_ptcloud, tile_utm_bbox[0], tile_utm_bbox[3]);
                convert_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, resolution_, fine_resolution_factor_, d_ptcloud, tile_utm_bbox[0], tile_utm_bbox[3]);
                delete[] ref_data;
                delete[] src_data;
                //update_ptcloud(d_ptcloud, rough_rotation_, rough_translation_,tile_utm_bbox[0]+ GLOBAL_OFFSET_X_m_,tile_utm_bbox[3]+ GLOBAL_OFFSET_Y_m_,GLOBAL_OFFSET_X_m_,GLOBAL_OFFSET_Y_m_);
                //// write ptcloud
                ss.str(""); 
                tile_dir = "N:\\tasks\\reg\\rough_reg1\\tile_rough\\";
                ss << tile_dir << "tile_" << tile_info.tile_id << "ref.txt";
                ref_out_path = ss.str();
                ss.str("");
                ss << tile_dir << "tile_" << tile_info.tile_id << "src.txt";
                src_out_path = ss.str();
                //write_ptcloud_gs(d_ptcloud, src_out_path, tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);
                //write_ptcloud_gs(s_ptcloud, ref_out_path, tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);

                //write_ptcloud(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, 0.5, ref_out_path);
                //write_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, 0.5, src_out_path);


                gs::ICPResult result;
                //icp(d_ptcloud, s_ptcloud, icp_para, result);
                tile_info.translation[0] = result.translation[0];
                tile_info.translation[1] = result.translation[1];
                tile_info.translation[2] = result.translation[2];
                tile_infos_icp_private.push_back(tile_info);
                // clear data
                //for (int i = 0; i < d_ptcloud.size(); ++i) {
                //    delete[] d_ptcloud[i]->pos;
                //}
                //for (int i = 0; i < s_ptcloud.size(); ++i) {
                //    delete[] s_ptcloud[i]->pos;
                //}
            }
#pragma omp critical
            {
                tile_infos_icp_.clear();
                for (int i = 0; i < tile_infos_icp_private.size(); ++i) {
                    tile_infos_icp_.push_back(tile_infos_icp_private[i]);
                }
            }
        }

    }
    void STEP2_3_FINE_ALIGN(std::string debug_dir) {
        std::cout << "\n[STEP2_3] Fine align tiles" << std::endl;
        std::vector<gs::Point> fine_src_pts;
        int id = 0;
        double* value = new double[1];

        /*Extract source pts
        */
        std::vector<TILE_INFO> tile_infos_tmp;
        {
            double tile_utm_bbox[4];
            int tile_ref_bbox[4];
            int tile_src_bbox[4];
            vector<TILE_INFO> tile_infos_icp_private;
            for (int i = 0; i < tile_infos_icp_.size(); ++i) {
                TILE_INFO tile_info = tile_infos_icp_[i];
                int col = tile_info.tile_id % num_tile_x_;
                int row = (tile_info.tile_id - col) / num_tile_x_;
                tile_utm_bbox[0] = aoi_utm_bbox_[0] + col * tilesize_m_;
                tile_utm_bbox[1] = aoi_utm_bbox_[0] + (col + 1) * tilesize_m_;
                tile_utm_bbox[2] = aoi_utm_bbox_[3] - (row + 1) * tilesize_m_;
                tile_utm_bbox[3] = aoi_utm_bbox_[3] - row * tilesize_m_;
                get_aoi_bbox_pixels(tile_utm_bbox, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, tile_ref_bbox, tile_src_bbox);

                double* src_data = new double[(tile_src_bbox[1] - tile_src_bbox[0] + 1) * (tile_src_bbox[2] - tile_src_bbox[3] + 1)];
                double* ref_data = new double[(tile_ref_bbox[1] - tile_ref_bbox[0] + 1) * (tile_ref_bbox[2] - tile_ref_bbox[3] + 1)];

                src_band_->RasterIO(GF_Read, tile_src_bbox[0],
                    tile_src_bbox[3],
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1,
                    src_data,
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1, GDT_Float64, 0, 0);

                vector<gs::Point> d_ptcloud;
                // convert data from pixel coord to utm coor(origin is GLOBAL_OFFX, GLOABL_OFFY)
                convert_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, resolution_,fine_resolution_factor_, d_ptcloud, tile_utm_bbox[0], tile_utm_bbox[3]);
                delete[] src_data;
                // apply rough transformation to src
                //update_ptcloud(d_ptcloud, rough_rotation_, rough_translation_,0,0,0,0);

                //// write ptcloud
                fine_src_pts.insert(fine_src_pts.end(), d_ptcloud.begin(), d_ptcloud.end());
            }
        }
        std::cout << "# Src_pts for fine registration: " << fine_src_pts.size() << std::endl;

        /* Fine registration
        */
		double rough_best_trans[3];
        for (int i = 0; i < rough_translations_.size(); ++i) {
			std::cout << "Fine registration #" << i << std::endl;
            double fine_rmse;
            double translation[3];
            double rot[9];
            double fine_rot[9];
            double fine_trans[3];
            gs::initDiagonal(rot);
            if (i == 0) {
                translation[0] = rough_translations_[i][0];
                translation[1] = rough_translations_[i][1];
                translation[2] = rough_translations_[i][2];
                update_ptcloud(fine_src_pts, rough_rotation_, translation, 0, 0, 0, 0);
            }
            else {
                translation[0] = rough_translations_[i][0]- rough_translations_[i-1][0];
                translation[1] = rough_translations_[i][1]- rough_translations_[i-1][1];
                translation[2] = rough_translations_[i][2]- rough_translations_[i-1][2];
                update_ptcloud(fine_src_pts, rot, translation, 0, 0, 0, 0);
            }
			gs::initDiagonal(fine_rot);
			fine_trans[0] = 0;
			fine_trans[1] = 0;
			fine_trans[2] = 0;
            MULTI_ICP(fine_src_pts, fine_search_half_pixels_, fine_icp_percentage_threshold_, multi_or_not_, fine_icp_max_iter_, fine_icp_rmse_threshold_, fine_rot, fine_trans, plane_or_not_,fine_rmse, "");
            std::cout <<" : rough_trans[3]: " << rough_translations_[i][0] << ", " << rough_translations_[i][1] << ", " << rough_translations_[i][2] << " RMSE: " << fine_rmse << std::endl;;
            if (i == 0) {
                fine_rmse_ = fine_rmse;
				std::copy(&fine_rot[0], &fine_rot[9], &fine_rotation_[0]);
				std::copy(&fine_trans[0], &fine_trans[3], &fine_translation_[0]);
				std::copy(&rough_translations_[i][0], &rough_translations_[i][3], &rough_best_trans[0]);

            }
            else {
                if (fine_rmse < fine_rmse_) {
                    fine_rmse_ = fine_rmse;
                    std::copy(&fine_rot[0] , &fine_rot[9], &fine_rotation_[0]);
                    std::copy(&fine_trans[0] , &fine_trans[3] , &fine_translation_[0]);
					std::copy(&rough_translations_[i][0], &rough_translations_[i][3], &rough_best_trans[0]);
                }
            }
        }
        
		// Merge rough_trans and fine_trans
		merge_two_trans(rough_rotation_, rough_best_trans, fine_rotation_, fine_translation_, final_rot_, final_trans_);
		std::cout << std::endl;
		std::cout << "##################### Result ##########################" << std::endl;
		std::cout << "Rough rotation:" << rough_rotation_[0] << "," << rough_rotation_[1] << "," << rough_rotation_[2] << "\n"
			<< rough_rotation_[3] << "," << rough_rotation_[4] << "," << rough_rotation_[5] << "\n"
			<< rough_rotation_[6] << "," << rough_rotation_[7] << "," << rough_rotation_[8] << "\n"
			<< "Rough translation: " << rough_best_trans[0] << "," << rough_best_trans[1] << "," << rough_best_trans[2] << "\n\n"
			<< "Fine rotation:" << fine_rotation_[0] << "," << fine_rotation_[1] << "," << fine_rotation_[2] << "\n"
			<< fine_rotation_[3] << "," << fine_rotation_[4] << "," << fine_rotation_[5] << "\n"
			<< fine_rotation_[6] << "," << fine_rotation_[7] << "," << fine_rotation_[8] << "\n"
			<< "Fine translation: " << fine_translation_[0] << "," << fine_translation_[1] << "," << fine_translation_[2] << "\n\n"
			<< "Final rotation:" << final_rot_[0] << "," << final_rot_[1] << "," << final_rot_[2] << "\n"
			<< final_rot_[3] << "," << final_rot_[4] << "," << final_rot_[5] << "\n"
			<< final_rot_[6] << "," << final_rot_[7] << "," << final_rot_[8] << "\n"
			<< "Final translation: " << final_trans_[0] << "," << final_trans_[1] << "," << final_trans_[2] << "\n";

		std::string out_path = debug_dir + "fine_trans.txt";
		std::string rough_trans_path = debug_dir + "rough_trans.txt";
		std::string final_trans_path= debug_dir + "final_trans.txt";
        if (debug_dir != "" && verbose_>0) {
            ofstream ofs(out_path);
            ofs <<setprecision(11)<< fine_rotation_[0] << " " <<
                fine_rotation_[1] << " " <<
                fine_rotation_[2] << " " <<
                fine_translation_[0] << "\n" <<
                fine_rotation_[3] << " " <<
                fine_rotation_[4] << " " <<
                fine_rotation_[5] << " " <<
                fine_translation_[1] << "\n" <<
                fine_rotation_[6] << " " <<
                fine_rotation_[7] << " " <<
                fine_rotation_[8] << " " <<
                fine_translation_[2] << "\n" <<
                0 << " " << 0 << " " << 0 << " " << 1;
            ofs.close();

			ofs.open(rough_trans_path);
			ofs << setprecision(11) << rough_rotation_[0] << " " <<
				rough_rotation_[1] << " " <<
				rough_rotation_[2] << " " <<
				rough_best_trans[0] << "\n" <<
				rough_rotation_[3] << " " <<
				rough_rotation_[4] << " " <<
				rough_rotation_[5] << " " <<
				rough_best_trans[1] << "\n" <<
				rough_rotation_[6] << " " <<
				rough_rotation_[7] << " " <<
				rough_rotation_[8] << " " <<
				rough_best_trans[2] << "\n" <<
				0 << " " << 0 << " " << 0 << " " << 1;
			ofs.close();

			ofs.open(final_trans_path);
			ofs << setprecision(11) << final_rot_[0] << " " <<
				final_rot_[1] << " " <<
				final_rot_[2] << " " <<
				final_trans_[0] << "\n" <<
				final_rot_[3] << " " <<
				final_rot_[4] << " " <<
				final_rot_[5] << " " <<
				final_trans_[1] << "\n" <<
				final_rot_[6] << " " <<
				final_rot_[7] << " " <<
				final_rot_[8] << " " <<
				final_trans_[2] << "\n" <<
				0 << " " << 0 << " " << 0 << " " << 1;
			ofs.close();
        }
    }
    void STEP_END() {
        GDALClose(ref_dataset_);
        GDALClose(src_dataset_);
    }
    
    void MULTI_ICP(std::vector<gs::Point> src_pts,int search_half_pixels,double percentage_threshold,bool multi_or_not,int max_iter,double rmse_threshold,double* rot_out,double* trans_out,double plane_or_not,double& final_rmse,string debug_path) {
		update_ptcloud(src_pts, rot_out, trans_out, 0, 0, 0, 0);
        Eigen::MatrixXd X, Y;
        Eigen::VectorXd W;
        double corr_rmse = 0;
        FIND_MULTI_LINK_CORRESPONDENCE(src_pts,X,Y,W,search_half_pixels,percentage_threshold,multi_or_not,max_iter,corr_rmse,plane_or_not);
        final_rmse = corr_rmse;
        //write_ptcloud_corrs(corrs, "N:\\tasks\\reg\\ref_pts.txt", "N:\\tasks\\reg\\src_pts.txt");
        //Estmate Transformation
        ofstream ofs;
        if (debug_path != "" && verbose_>0) {
            ofs.open(debug_path);
        }
        for (int iter = 0; iter < max_iter; ++iter) {
            std::cout << std::setprecision(11) << "ICP #Iteration: " << iter << "# corrs: " << X.rows() << " RMSE: " << corr_rmse <<"Trans: "<<trans_out[0]<<","<<trans_out[1]<<","<<trans_out[2]<< std::endl;
            if (debug_path != "" && verbose_>0) {
                ofs << std::setprecision(11)<< corr_rmse << " " << trans_out[0] <<" "<<trans_out[1] <<" "<<trans_out[2] << std::endl;
            }
            Eigen::Matrix3d R;
            Eigen::Vector3d T;
            double rotation[9];
            double translation[3];
            ESTIMATE_TRANSFORMATION_WEIGHTED_CORRS(X, Y, W, R, T);
            //ESTIMATE_TRANSFORMATION_WEIGHTED_CORRS(corrs, rotation, translation,plane_or_not);
            rotation[0] = R(0, 0);
            rotation[1] = R(0, 1);
            rotation[2] = R(0, 2);
            rotation[3] = R(1, 0);
            rotation[4] = R(1, 1);
            rotation[5] = R(1, 2);
            rotation[6] = R(2, 0);
            rotation[7] = R(2, 1);
            rotation[8] = R(2, 2);
            translation[0] = T(0);
            translation[1] = T(1);
            translation[2] = T(2);


            double tmp[9];
            double tmp_vec3[3];
            gs::matrixMult(rotation, rot_out, tmp);
            gs::rotate_mat(trans_out, rotation, tmp_vec3);
            trans_out[0] = tmp_vec3[0] + translation[0];
            trans_out[1] = tmp_vec3[1] + translation[1];
            trans_out[2] = tmp_vec3[2] + translation[2];
            std::copy(&tmp[0], &tmp[9], &rot_out[0]);

            // Update source pts
            update_ptcloud(src_pts, rotation, translation, 0.0, 0.0, 0.0, 0.0);

            // FInd correspondence
            double pre_rmse = corr_rmse;
            //FIND_MULTI_LINK_CORRESPONDENCE(src_pts, corrs, search_half_pixels,percentage_threshold, multi_or_not, max_iter, corr_rmse,plane_or_not);
            FIND_MULTI_LINK_CORRESPONDENCE(src_pts, X, Y, W, search_half_pixels, percentage_threshold, multi_or_not, max_iter, corr_rmse, plane_or_not);
            //debug
            //if (iter == 0) {
            //    string out_file = "J:\\xuningli\\3DDataRegistration\\REG\\dsm_reg\\out\\corr.txt";
            //    ofstream ofs1(out_file);

            //    for (int i = 0; i < corrs.size(); ++i) {
            //        std::stringstream ss;
            //        ss << corrs[i].src.pos[0] << " " << corrs[i].src.pos[1] << " " << corrs[i].src.pos[2] << " " << corrs[i].ref.pos[0] << " " << corrs[i].ref.pos[1] << " " << corrs[i].ref.pos[2] << "\n";
            //        string line = ss.str();
            //        ss.clear();
            //        ofs1<<line;
            //    }
            //    ofs1.close();
            //}


            if (abs(pre_rmse - corr_rmse) < rmse_threshold) {
                final_rmse = (corr_rmse < final_rmse) ? corr_rmse : final_rmse;
                break;
            }
            final_rmse = (corr_rmse < final_rmse) ? corr_rmse : final_rmse;
        }
        if (debug_path != "" && verbose_>0) {
            ofs.close();
        }

        //print
        std::cout << "ICP result rotation: " << rot_out[0] << "," <<
            rot_out[1] << "," <<
            rot_out[2] << "," <<
            rot_out[3] << "," <<
            rot_out[4] << "," <<
            rot_out[5] << "," <<
            rot_out[6] << "," <<
            rot_out[7] << "," <<
            rot_out[8] << "," <<"\n"<<
            "ICP result translation" << trans_out[0] << "," <<
            trans_out[1] << "," <<
            trans_out[2] << endl;
    }

    void HARRIS(double* data, int width, int height,int step, double k, double& lambda1, double& lambda2, double& R) {
        lambda1 = 0;
        lambda2 = 0;
        R = 0;
        double** M = new double* [2];
        M[0] = new double[2];
        M[1] = new double[2];
        double** V = new double* [2];
        V[0] = new double[2];
        V[1] = new double[2];
        double sigma[2];
        double dx, dy;
        M[0][0] = 0.0;
        M[0][1] = 0.0;
        M[1][0] = 0.0;
        M[1][1] = 0.0;
        V[0][0] = 0.0;
        V[0][1] = 0.0;
        V[1][0] = 0.0;
        V[1][1] = 0.0;
        int valid_cnt = 0;
        double valid_ratio = 0.0;
        bool invalid = false;
        vector<double> dx_list;
        vector<double> dy_list;
        int mid1, mid2;
        int id;
        for (int row = step; row < height-step; ++row) {
            for (int col = step; col < width-step; ++col) {
                // start compute derivative
                dx_list.clear();
                dy_list.clear();
                for (int col_step = -step; col_step < step+1; ++col_step) {
                    id = col + col_step + row * width;
                    if (isnan(data[id])) {
                        invalid = true;
                        break;
                    }
                }
                for (int row_step = -step; row_step < step + 1; ++row_step) {
                    id = col + (row + row_step) * width;
                    if (isnan(data[id])) {
                        invalid = true;
                        break;
                    }
                }
                if (invalid) {
                    invalid = false;
                    continue;
                }
                for (int col_step = -step; col_step < step + 1; ++col_step) {
                    if (col_step < 0) {
                        dx_list.push_back(2 * (data[col + row * width]-data[col + col_step + row * width]));
                    }
                    else if(col_step > 0) {
                        dx_list.push_back(2 * (data[col + col_step + row * width] - data[col + row * width]));
                    }
                }
                for (int row_step = -step; row_step < step + 1; ++row_step) {
                    if (row_step < 0) {
                        dy_list.push_back(2 * (data[col + row * width]-data[col + (row + row_step) * width]));
                    }
                    else if (row_step > 0) {
                        dy_list.push_back(2 * (data[col + (row + row_step) * width] - data[col + row * width]));
                    }

                }
                sort(dx_list.begin(), dx_list.end(), cmp_descend_only_double);
                sort(dy_list.begin(), dy_list.end(), cmp_descend_only_double);

                mid1 = dx_list.size() / 2 - 1;
                mid2 = mid1 + 1;
                dx = (dx_list[mid1] + dx_list[mid2]) / 2;
                dy = (dy_list[mid1] + dy_list[mid2]) / 2;

                M[0][0] += dx * dx;
                M[0][1] += dx * dy;
                M[1][0] += dx * dy;
                M[1][1] += dy * dy;
                valid_cnt++;
            }
        }
        valid_ratio = (double)valid_cnt / (double)((width - 1) * (height - 1));
        if (valid_ratio > 0.1) {
            M[0][0] /= (double)valid_cnt;
            M[0][1] /= (double)valid_cnt;
            M[1][0] /= (double)valid_cnt;
            M[1][1] /= (double)valid_cnt;
            // compute eigenvalues
            double a, b, c, mid;
            a = 1;
            b = -M[0][0] - M[1][1];
            c = M[0][0] * M[1][1] - M[0][1] * M[1][0];
            mid = sqrt(b * b - 4 * a * c);
            lambda1 = (-b + mid) / (2 * a);
            lambda2 = (-b - mid) / (2 * a);
            R = lambda1 * lambda2 - k * (lambda1 + lambda2) * (lambda1 + lambda2);
        }

        delete[] M[0];
        delete[] M[1];
        delete[] M;
        delete[] V[0];
        delete[] V[1];
        delete[] V;
    }

    void EXTRACT_TILE_ID(int id,string tile_dir) {
        double tile_utm_bbox[4];
        int tile_ref_bbox[4];
        int tile_src_bbox[4];
        for (int i = 0; i < tile_infos_all_.size(); ++i) {
            if (tile_infos_all_[i].tile_id != id) {
                continue;
            }
            TILE_INFO tile_info = tile_infos_all_[i];
            int col = tile_info.tile_id % num_tile_x_;
            int row = (tile_info.tile_id - col) / num_tile_x_;
            tile_utm_bbox[0] = aoi_utm_bbox_[0] + col * tilesize_m_;
            tile_utm_bbox[1] = aoi_utm_bbox_[0] + (col + 1) * tilesize_m_;
            tile_utm_bbox[2] = aoi_utm_bbox_[3] - (row + 1) * tilesize_m_;
            tile_utm_bbox[3] = aoi_utm_bbox_[3] - row * tilesize_m_;
            get_aoi_bbox_pixels(tile_utm_bbox, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, tile_ref_bbox, tile_src_bbox);

            double* ref_data = new double[(tile_ref_bbox[1] - tile_ref_bbox[0] + 1) * (tile_ref_bbox[2] - tile_ref_bbox[3] + 1)];
            double* src_data = new double[(tile_src_bbox[1] - tile_src_bbox[0] + 1) * (tile_src_bbox[2] - tile_src_bbox[3] + 1)];
            ref_band_->RasterIO(GF_Read, tile_ref_bbox[0],
                tile_ref_bbox[3],
                tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                tile_ref_bbox[2] - tile_ref_bbox[3] + 1,
                ref_data,
                tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                tile_ref_bbox[2] - tile_ref_bbox[3] + 1, GDT_Float64, 0, 0);
            src_band_->RasterIO(GF_Read, tile_src_bbox[0],
                tile_src_bbox[3],
                tile_src_bbox[1] - tile_src_bbox[0] + 1,
                tile_src_bbox[2] - tile_src_bbox[3] + 1,
                src_data,
                tile_src_bbox[1] - tile_src_bbox[0] + 1,
                tile_src_bbox[2] - tile_src_bbox[3] + 1, GDT_Float64, 0, 0);

            // write before rough align ptcloud
            stringstream ss;
            string ref_out_path;
            string src_out_path;
            ss << tile_dir << "tile_" << tile_info.tile_id << "ref.txt";
            ref_out_path = ss.str();
            ss.str("");
            ss << tile_dir << "tile_" << tile_info.tile_id << "src.txt";
            src_out_path = ss.str();
            write_ptcloud(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, 0.5, ref_out_path, 
                tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);
            write_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, 0.5, src_out_path, 
                tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);

        }


    }
    // extract the tiles into ptcloud
    void EXTRACT_TILES(string dir) {
        STEP2_1_COLLECT_TILE_INFOS();
        STEP2_2_FILTER_TILE_INFOS(dir);
        stringstream ss;
        string icp_tiles_shp_path;
        ss << dir << "fine_tiles.shp";
        icp_tiles_shp_path = ss.str();
        write_shp(tile_infos_icp_, icp_tiles_shp_path);
        std::vector<TILE_INFO> tile_infos_tmp;
#pragma omp parallel
        {
            double tile_utm_bbox[4];
            int tile_ref_bbox[4];
            int tile_src_bbox[4];

            vector<TILE_INFO> tile_infos_icp_private;
#pragma omp for nowait
            for (int i = 0; i < tile_infos_icp_.size(); ++i) {
                //std::cout << i << std::endl;
                //tile_infos_icp.push_back(tile_infos[tiles_only_index[i].second]);
                TILE_INFO tile_info = tile_infos_icp_[i];
                int col = tile_info.tile_id % num_tile_x_;
                int row = (tile_info.tile_id - col) / num_tile_x_;
                tile_utm_bbox[0] = aoi_utm_bbox_[0] + col * tilesize_m_;
                tile_utm_bbox[1] = aoi_utm_bbox_[0] + (col + 1) * tilesize_m_;
                tile_utm_bbox[2] = aoi_utm_bbox_[3] - (row + 1) * tilesize_m_;
                tile_utm_bbox[3] = aoi_utm_bbox_[3] - row * tilesize_m_;
                get_aoi_bbox_pixels(tile_utm_bbox, ref_utm_bbox_, src_utm_bbox_, ref_width_, ref_height_, src_width_, src_height_, tile_ref_bbox, tile_src_bbox);

                double* ref_data = new double[(tile_ref_bbox[1] - tile_ref_bbox[0] + 1) * (tile_ref_bbox[2] - tile_ref_bbox[3] + 1)];
                double* src_data = new double[(tile_src_bbox[1] - tile_src_bbox[0] + 1) * (tile_src_bbox[2] - tile_src_bbox[3] + 1)];
                ref_band_->RasterIO(GF_Read, tile_ref_bbox[0],
                    tile_ref_bbox[3],
                    tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                    tile_ref_bbox[2] - tile_ref_bbox[3] + 1,
                    ref_data,
                    tile_ref_bbox[1] - tile_ref_bbox[0] + 1,
                    tile_ref_bbox[2] - tile_ref_bbox[3] + 1, GDT_Float64, 0, 0);
                src_band_->RasterIO(GF_Read, tile_src_bbox[0],
                    tile_src_bbox[3],
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1,
                    src_data,
                    tile_src_bbox[1] - tile_src_bbox[0] + 1,
                    tile_src_bbox[2] - tile_src_bbox[3] + 1, GDT_Float64, 0, 0);

                // write before rough align ptcloud
                ss.str("");
                string ref_out_path;
                ss << dir << "tile_" << tile_info.tile_id << "ref.txt";
                ref_out_path = ss.str();
                ss.str("");
                ss << dir << "tile_" << tile_info.tile_id << "src.txt";
                string src_out_path = ss.str();
                write_ptcloud(ref_data, tile_ref_bbox[1] - tile_ref_bbox[0] + 1, tile_ref_bbox[2] - tile_ref_bbox[3] + 1, 0.5, ref_out_path,tile_utm_bbox[0]+GLOBAL_OFFSET_X_m_,tile_utm_bbox[3]+GLOBAL_OFFSET_Y_m_);
                write_ptcloud(src_data, tile_src_bbox[1] - tile_src_bbox[0] + 1, tile_src_bbox[2] - tile_src_bbox[3] + 1, 0.5, src_out_path, tile_utm_bbox[0] + GLOBAL_OFFSET_X_m_, tile_utm_bbox[3] + GLOBAL_OFFSET_Y_m_);
                delete[] ref_data;
                delete[] src_data;
            }
#pragma omp critical
            {
                tile_infos_icp_.clear();
                for (int i = 0; i < tile_infos_icp_private.size(); ++i) {
                    tile_infos_icp_.push_back(tile_infos_icp_private[i]);
                }
            }
        }
    }

    void non_min_suppresion(int width,int height, double** rmses, int resolution, vector<pair<int,int>>& min_ids_list) {
        int col_min, col_max, row_min, row_max;
        vector<pair<int, int>> candidate_inds;
        vector<double> candidate_rmse;

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                bool minimal = true;
                col_min = ((col - resolution) < 0) ? 0 : (col - resolution);
                col_max = ((col + resolution) > width - 1) ? width - 1 : (col + resolution);
                row_min = ((row - resolution) < 0) ? 0 : (row - resolution);
                row_max = ((row + resolution) > height - 1) ? height - 1 : (row + resolution);
                for (int row1 = row_min; row1 < row_max; ++row1) {
                    for (int col1 = col_min; col1 < col_max; ++col1) {
                        if (rmses[row][col] > rmses[row1][col1]) {
                            minimal = false;
                        }
                    }
                }
                if (minimal) {
                    candidate_inds.push_back(make_pair(row, col));
                    candidate_rmse.push_back(rmses[row][col]);
                }
                else {
                    minimal = true;
                }
            }
        }
        for (auto i : sort_indexes(candidate_rmse)) {
            min_ids_list.push_back(candidate_inds[i]);
        }
    }

    void print_info() {
        std::cout << "Registration Start\n";
        if (multi_or_not_) {
            std::cout << "Multi-link function on\n";
        }
        else {
            std::cout << "Multi-link function off\n";
        }
        if (plane_or_not_) {
            std::cout << "Point-to-Plane function on\n";
        }
        else {
            std::cout << "Point-to-Plane function off\n";
        }
        if (brute_force_search_or_not_) {
            std::cout << "Brute-Force search on X-Y plane on\n";
        }
        else {
            std::cout << "Brute-Force search on X-Y plane off\n";
        }

    }
};

class DSM_TRANSFORM {
public:
    DSM_TRANSFORM(std::string in_path,std::string out_path,double* rotation,double* translation,double offx,double offy) {
        in_zmin_ = -9999;
        in_zmax_ = -9999;
        int tile_size = 3000;
        std::copy(&rotation[0], &rotation[9], &rotation_[0]);
        std::copy(&translation[0], &translation[3], &translation_[0]);
        GLOBAL_OFFSET_X_m_ = offx;
        GLOBAL_OFFSET_Y_m_ = offy;
        double transform[6],transform_out[6];
        int xSize, ySize;
        out_path_ = out_path;
        in_path_ = in_path;
        GDALAllRegister();
        pdriver_ = GetGDALDriverManager()->GetDriverByName("GTiff");
        in_dataset_ = (GDALDataset*)GDALOpen(in_path_.c_str(), GA_ReadOnly);
        in_dataset_->GetGeoTransform(transform);
        xSize = in_dataset_->GetRasterXSize();
        ySize = in_dataset_->GetRasterYSize();
        in_width_ = xSize;
        in_height_ = ySize;
        in_utm_bbox_[0] = transform[0]+transform[1]/2;
        in_utm_bbox_[1] = transform[0] + transform[1] / 2 + (xSize - 1) * transform[1];
        in_utm_bbox_[2] = transform[3]-transform[1] / 2 - (ySize - 1) * transform[1];
        in_utm_bbox_[3] = transform[3]- transform[1] / 2;
        resolution_ = transform[1];
        in_datatype_=GDALGetRasterDataType(GDALGetRasterBand(in_dataset_, 1));
        in_band_ = in_dataset_->GetRasterBand(1);
        //Collect zmin and zmax of in_raster
        int num_tile_x = ceil((double)in_width_ / (double)tile_size);
        int num_tile_y = ceil((double)in_height_ / (double)tile_size);
        int fill_x = 0;
        int fill_y = 0;
        int init_cnt = 0;
        for (int row = 0; row < num_tile_y; ++row) {
            for (int col = 0; col < num_tile_x; ++col) {
                
                fill_x = (tile_size * (col + 1) <= in_width_) ? tile_size * (col + 1) : in_width_;
                fill_x -= tile_size * col;
                fill_y = (tile_size * (row + 1) <= in_height_) ? tile_size * (row + 1) : in_height_;
                fill_y -= tile_size * row;
                double* in_data = new double[fill_x*fill_y];
                in_band_->RasterIO(GF_Read, col, row, fill_x, fill_y,
                    in_data, fill_x, fill_y, GDT_Float64, 0, 0);
                for (int i = 0; i < fill_x * fill_y; ++i) {
                    if (isnan(in_data[i])) {
                        continue;
                    }
                    init_cnt++;
                    if (init_cnt==1) {
                        in_zmin_ = in_data[i];
                        in_zmax_ = in_data[i];
                    }
                    if (in_data[i] > in_zmax_) {
                        in_zmax_ = in_data[i];
                    }
                    if (in_data[i] < in_zmin_) {
                        in_zmin_ = in_data[i];
                    }
                }
                delete[] in_data;
            }
        }
        // compute output bbox size
        double pt1[3], pt2[3], pt3[3], pt4[3], pt5[3], pt6[3], pt7[3], pt8[3],
               pt1_trans[3],pt2_trans[3],pt3_trans[3],pt4_trans[3], pt5_trans[3], pt6_trans[3], pt7_trans[3], pt8_trans[3];
        pt1[0] = in_utm_bbox_[0];
        pt1[1] = in_utm_bbox_[3];
        pt1[2] = in_zmin_;
        pt5[0] = in_utm_bbox_[0];
        pt5[1] = in_utm_bbox_[3];
        pt5[2] = in_zmax_;
        pt2[0] = in_utm_bbox_[1];
        pt2[1] = in_utm_bbox_[3];
        pt2[2] = in_zmin_;
        pt6[0] = in_utm_bbox_[1];
        pt6[1] = in_utm_bbox_[3];
        pt6[2] = in_zmax_;
        pt3[0] = in_utm_bbox_[1];
        pt3[1] = in_utm_bbox_[2];
        pt3[2] = in_zmin_;
        pt7[0] = in_utm_bbox_[1];
        pt7[1] = in_utm_bbox_[2];
        pt7[2] = in_zmax_;
        pt4[0] = in_utm_bbox_[0];
        pt4[1] = in_utm_bbox_[2];
        pt4[2] = in_zmin_;
        pt8[0] = in_utm_bbox_[0];
        pt8[1] = in_utm_bbox_[2];
        pt8[2] = in_zmax_;
        point_trans(pt1, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt1_trans);
        point_trans(pt2, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt2_trans);
        point_trans(pt3, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt3_trans);
        point_trans(pt4, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt4_trans);
        point_trans(pt5, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt5_trans);
        point_trans(pt6, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt6_trans);
        point_trans(pt7, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt7_trans);
        point_trans(pt8, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt8_trans);
        out_utm_bbox_[0] = min4(pt1_trans[0], pt4_trans[0], pt5_trans[0], pt8_trans[0]);
        out_utm_bbox_[1] = max4(pt2_trans[0], pt3_trans[0], pt6_trans[0], pt7_trans[0]);
        out_utm_bbox_[2] = min4(pt3_trans[1], pt4_trans[1], pt7_trans[1], pt8_trans[1]);
        out_utm_bbox_[3] = max4(pt1_trans[1], pt2_trans[1], pt5_trans[1], pt6_trans[1]);
        out_width_ = floor((out_utm_bbox_[1] - out_utm_bbox_[0]+transform[1]/2) / transform[1])+1;
        out_height_ = floor((out_utm_bbox_[3] - out_utm_bbox_[2] + transform[1] / 2) / transform[1]) + 1;

        std::copy(std::begin(transform), std::end(transform), std::begin(transform_out));
        transform_out[0] = out_utm_bbox_[0];
        transform_out[3] = out_utm_bbox_[3];
        // Create Empty output dsm
        out_dataset_ = pdriver_->Create(out_path_.c_str(), out_width_, out_height_, 1, in_datatype_, NULL);
        out_dataset_->SetGeoTransform(transform_out);
        out_band_ = out_dataset_->GetRasterBand(1);
        num_tile_x = ceil((double)out_width_ / (double)tile_size);
        num_tile_y= ceil((double)out_height_ / (double)tile_size);
        for (int row = 0; row < num_tile_y; ++row) {
            for (int col = 0; col < num_tile_x; ++col) {
                fill_x = (tile_size * (col + 1) <=out_width_)? tile_size * (col + 1): out_width_;
                fill_x -= tile_size * col;
                fill_y = (tile_size * (row + 1) <= out_height_) ? tile_size * (row + 1) : out_height_;
                fill_y -= tile_size * row;
                rasterio_write_nan(out_band_, tile_size * col, tile_size * row, fill_x, fill_y);
            }
        }
        std::string out_tfw_path = out_path_;
        strutil::replace_last(out_tfw_path, ".tif", ".tfw");
        export_tfw(out_tfw_path, transform_out);
        //GDALClose(out_dataset_);
    };
    std::string in_path_, out_path_;
    GDALDriver* pdriver_;
    GDALDataset* out_dataset_;
    GDALDataset* in_dataset_;
    GDALRasterBand* in_band_;
    GDALRasterBand* out_band_;
    GDALDataType in_datatype_;
    int out_width_, out_height_, in_width_, in_height_;
    double GLOBAL_OFFSET_X_m_;
    double GLOBAL_OFFSET_Y_m_;
    double resolution_;
    double out_utm_bbox_[4];
    double in_utm_bbox_[4];
    double in_zmin_, in_zmax_;
    double rotation_[9];
    double translation_[3];

    void START() {
        double* write_value = new double[1];
        double* read_value = new double[1];
        double pt[3],pt_out[3];
        int col_out, row_out;
        for (int row = 0; row < in_height_; ++row)
        {
            for (int col = 0; col < in_width_; ++col) {
                pt[0] = in_utm_bbox_[0] + resolution_ * col;
                pt[1] = in_utm_bbox_[3] - resolution_ * row;
                in_band_->RasterIO(GF_Read, col, row, 1, 1,
                    read_value, 1, 1, GDT_Float64, 0, 0);
                pt[2] = read_value[0];
                if (isnan(pt[2])) {
                    continue;
                }
                point_trans(pt, rotation_, translation_, GLOBAL_OFFSET_X_m_, GLOBAL_OFFSET_Y_m_, pt_out);
                col_out = floor((pt_out[0] - out_utm_bbox_[0] + resolution_ / 2) / resolution_);
                row_out= floor((out_utm_bbox_[3]- pt_out[1]+ resolution_ / 2) / resolution_);
                if (row_out < 0) {
                    int a = 0;
                }
                //std::cout << col_out << "," << row_out << std::endl;
                out_band_->RasterIO(GF_Read, col_out, row_out, 1, 1,
                    read_value, 1, 1, GDT_Float64, 0, 0);
                if (pt_out[2] > read_value[0] || isnan(read_value[0])) {
                    out_band_->RasterIO(GF_Write, col_out, row_out, 1, 1,
                        &pt_out[2], 1, 1, GDT_Float64, 0, 0);
                }
            }
        }
        GDALClose(in_dataset_);
        GDALClose(out_dataset_);
    }
    double min4(double a, double b, double c, double d) {
        double tmp;
        tmp = min(a, b);
        tmp = min(tmp, c);
        tmp = min(tmp, d);
        return tmp;
    }
    double max4(double a, double b, double c, double d) {
        double tmp;
        tmp = max(a, b);
        tmp = max(tmp, c);
        tmp = max(tmp, d);
        return tmp;
    }
    void gen_rotation_zaxis(double* rot, double angle) { // in degree
        gs::initDiagonal(rot);
        angle = angle * 3.1415926 / 180.0;
        rot[0] = cos(angle);
        rot[1] = -sin(angle);
        rot[3] = sin(angle);
        rot[4] = cos(angle);
    }
    void rasterio_write_nan(GDALRasterBand* band, int offsetx, int offsety, int xsize, int ysize) {
        int cnt = xsize * ysize;
        GDALDataType dt_in = GDALGetRasterDataType(band);
        int bits = GDALGetDataTypeSize(dt_in);
        //cout << "rasterio_write_nan: " << bits << endl;
        switch (bits) {
        case 16: {
            //GInt16* buf16 = new GInt16[cnt];
            GInt16* buf16 = (GInt16*)CPLMalloc(sizeof(GInt16) * cnt);
            for (int i = 0; i < cnt; ++i) {
                buf16[i] = std::numeric_limits<GInt16>::quiet_NaN();
            }
            band->RasterIO(GF_Write, offsetx, offsety, xsize, ysize, buf16, xsize, ysize, GDT_Int16, 0, 0);
            //delete[] buf16;
            CPLFree(buf16);
            break;
        }
        case 32: {
            //float* buf32 = new float[cnt];
            float* buf32 = (float*)CPLMalloc(sizeof(float) * cnt);
            for (int i = 0; i < cnt; ++i) {
                buf32[i] = std::numeric_limits<float>::quiet_NaN();
            }
            band->RasterIO(GF_Write, offsetx, offsety, xsize, ysize, buf32, xsize, ysize, GDT_Float32, 0, 0);
            CPLFree(buf32);
            //delete[] buf32;
            break;

        }

        case 64: {
            float* buf64 = new float[cnt];
            for (int i = 0; i < cnt; ++i) {
                buf64[i] = std::numeric_limits<double>::quiet_NaN();
            }
            band->RasterIO(GF_Write, offsetx, offsety, xsize, ysize, buf64, xsize, ysize, GDT_Float64, 0, 0);
            break;
        }

        }

    }
    void export_tfw(string path, double* transform) {
        std::ofstream out(path.c_str());
        out << std::fixed << std::setprecision(8) << transform[1] << endl;
        out << std::fixed << std::setprecision(8) << transform[2] << endl;
        out << std::fixed << std::setprecision(8) << transform[4] << endl;
        out << std::fixed << std::setprecision(8) << transform[5] << endl;
        out << std::fixed << std::setprecision(8) << transform[0] << endl;
        out << std::fixed << std::setprecision(8) << transform[3] << endl;
        out.close();

    }
    void point_trans(double* in_pt, double* rotation, double* translation, double offx, double offy, double* out_pt) {
        in_pt[0] -= offx;
        in_pt[1] -= offy;
        gs::rotate_mat(in_pt, rotation, out_pt);
        out_pt[0] = out_pt[0] + translation[0] + offx;
        out_pt[1] = out_pt[1] + translation[1] + offy;
        out_pt[2] = out_pt[2] + translation[2] ;
        in_pt[0] += offx;
        in_pt[1] += offy;
    }
};

void LIDARDSM_REG(std::string dsm_ref_path,std::string dsm_src_path) {
    DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
    dsm_reg.START("");
    std::string dsm_out_name = fs::path(dsm_src_path).filename().string();
    dsm_out_name=dsm_out_name.replace(dsm_out_name.find(".tif"), sizeof(".tif") - 1, "_reg.tif");
    std::string dsm_out_path = fs::path(dsm_src_path).replace_filename(dsm_out_name).string();
    DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_out_path, dsm_reg.final_rot_, dsm_reg.final_trans_, dsm_reg.GLOBAL_OFFSET_X_m_, dsm_reg.GLOBAL_OFFSET_Y_m_);
    dsm_trans.START();  
}

void SATDSM_REG(std::string dsm_ref_path, std::string dsm_src_path) {
    DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
    dsm_reg.search_half_size_pixel_ = 10;
    dsm_reg.rough_icp_max_iter_ = 200;
    dsm_reg.fine_icp_max_iter_ = 200;
    dsm_reg.START("");
    std::string dsm_out_name = fs::path(dsm_src_path).filename().string();
    dsm_out_name = dsm_out_name.replace(dsm_out_name.find(".tif"), sizeof(".tif") - 1, "_reg.tif");
    std::string dsm_out_path = fs::path(dsm_src_path).replace_filename(dsm_out_name).string();
    DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_out_path, dsm_reg.final_rot_, dsm_reg.final_trans_, dsm_reg.GLOBAL_OFFSET_X_m_, dsm_reg.GLOBAL_OFFSET_Y_m_);
    dsm_trans.START();
}

void SATDSM_REG_only_translation(std::string dsm_ref_path, std::string dsm_src_path) {
    DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
    dsm_reg.search_half_size_pixel_ = 10;
    dsm_reg.rough_icp_max_iter_ = 200;
    dsm_reg.fine_icp_max_iter_ = 200;
    dsm_reg.START("");
    std::string dsm_out_name = fs::path(dsm_src_path).filename().string();
    dsm_out_name = dsm_out_name.replace(dsm_out_name.find(".tif"), sizeof(".tif") - 1, "_reg.tif");
    std::string dsm_out_path = fs::path(dsm_src_path).replace_filename(dsm_out_name).string();
    double rotation[9];
    rotation[0] = 1;
    rotation[1] = 0;
    rotation[2] = 0;
    rotation[3] = 0;
    rotation[4] = 1;
    rotation[5] = 0;
    rotation[6] = 0;
    rotation[7] = 0;
    rotation[8] = 1;


    DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_out_path, rotation, dsm_reg.final_trans_, dsm_reg.GLOBAL_OFFSET_X_m_, dsm_reg.GLOBAL_OFFSET_Y_m_);
    dsm_trans.START();
}

void DSM_REG_v1(std::string dsm_ref_path, std::string dsm_src_path) {
    DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
    dsm_reg.multi_or_not_ = true;
    dsm_reg.START("");
    std::string dsm_out_name = fs::path(dsm_src_path).filename().string();
    dsm_out_name = dsm_out_name.replace(dsm_out_name.find(".tif"), sizeof(".tif") - 1, "_reg.tif");
    std::string dsm_out_path = fs::path(dsm_src_path).replace_filename(dsm_out_name).string();
    DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_out_path, dsm_reg.final_rot_, dsm_reg.final_trans_, dsm_reg.GLOBAL_OFFSET_X_m_, dsm_reg.GLOBAL_OFFSET_Y_m_);
    dsm_trans.START();
}


//void DSM_REG_v1(std::string dsm_ref_path, std::string dsm_src_path,std::string dsm_src_out_path) {
//    DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
//    dsm_reg.multi_or_not_ = true;
//    dsm_reg.START("");
//    DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_src_out_path, dsm_reg.final_rot_, dsm_reg.final_trans_, dsm_reg.GLOBAL_OFFSET_X_m_, dsm_reg.GLOBAL_OFFSET_Y_m_);
//    dsm_trans.START();
//}

int main(int argc, char* argv[]) {
    //argparse::ArgumentParser program("Large-scale DSM registration based on ICP");
    //program.add_argument("-src").required().help("source/moving DSM file");
    //program.add_argument("-dst").required().help("reference/fixed DSM file");


    //try {
    //    program.parse_args(argc, argv);    // Example: ./main --color orange
    //}
    //catch (const std::runtime_error& err) {
    //    std::cerr << err.what() << std::endl;
    //    std::cerr << program;
    //    std::exit(1);
    //}


    //std::string src_file= program.get<std::string>("-src");
    //std::string dst_file = program.get<std::string>("-dst");
    //int a = 0;


    if (argc != 4) {
        std::cout << "USAGE: reg.exe [0: LiDAR-SAT, 1:small-scale SAT-SAT,2: general(high time cost), 3: no registration, only apply init transform to src_dsm, 4: only estimation translation ] [reference_dsm_path] source_dsm_path init_trans_file" << std::endl;
        return 0;
    }
    string mode = argv[1];
    if (mode == "0") {
        LIDARDSM_REG(argv[2], argv[3]);
    }
    else if (mode == "1") {
        SATDSM_REG(argv[2], argv[3]);
    }
    else if (mode == "2") {
        DSM_REG_v1(argv[2], argv[3]);
    }
    else if (mode == "3") {
        std::string dsm_src_path = argv[2];
        std::string init_file = argv[3];
        std::string dsm_out_name = fs::path(dsm_src_path).filename().string();
        dsm_out_name = dsm_out_name.replace(dsm_out_name.find(".tif"), sizeof(".tif") - 1, "_reg.tif");
        std::string dsm_out_path = fs::path(dsm_src_path).replace_filename(dsm_out_name).string();

        double rotation[9];
        double translation[3];
        double offset[2];

        std::ifstream file(init_file);
        std::string str;
        std::getline(file, str);
        offset[0] = std::stod(str);
        std::getline(file, str);
        offset[1] = std::stod(str);
        std::getline(file, str);
        rotation[0] = std::stod(str);
        std::getline(file, str);
        rotation[1] = std::stod(str);
        std::getline(file, str);
        rotation[2] = std::stod(str);
        std::getline(file, str);
        rotation[3] = std::stod(str);
        std::getline(file, str);
        rotation[4] = std::stod(str);
        std::getline(file, str);
        rotation[5] = std::stod(str);
        std::getline(file, str);
        rotation[6] = std::stod(str);
        std::getline(file, str);
        rotation[7] = std::stod(str);
        std::getline(file, str);
        rotation[8] = std::stod(str);
        std::getline(file, str);
        translation[0] = std::stod(str);
        std::getline(file, str);
        translation[1] = std::stod(str);
        std::getline(file, str);
        translation[2] = std::stod(str);

        DSM_TRANSFORM dsm_trans(dsm_src_path, dsm_out_path, rotation, translation, offset[0], offset[1]);
        dsm_trans.START();

    }
    else if (mode == "4") {
        SATDSM_REG_only_translation(argv[2], argv[3]);
    }





    //
    
    //using namespace gs;
    //std::string shp_out_all_path = "F:\\tasks\\reg\\tile_250m_all.shp";
    //std::string shp_out_only_path = "F:\\tasks\\reg\\tile_250m_only.shp";
    //std::string shp_out_icp_path = "F:\\tasks\\reg\\tile_250m_icp.shp";
    //std::string icp_out_path = "F:\\tasks\\reg\\icp_shift_150.txt";
    //std::string tileinfo_out_path = "F:\\tasks\\reg\\tile_150.txt";

    //string dsm_ref_path = "F:\\tasks\\reg\\rough_reg2\\lidar.tif";
    //string dsm_src_path = "F:\\tasks\\reg\\rough_reg2\\dsm.tif";
    //string tile_dir = "F:\\tasks\\reg\\tile_250m\\";
    //
    //string dsm_in_path = "F:\\tasks\\reg\\reg3\\uav_dsm.tif";
    //string dsm_out_path= "F:\\tasks\\reg\\reg3\\uav_dsm_5_5_2.tif";
    
    // create test dsm
    //create_dsm("N:\\tasks\\reg\\rough_reg1\\test\\src.tif");
    
    /*
    Test Large-scale DSM registration
    */
    //double** rmses = new double* [41];
    //for (int i = 0; i < 41; ++i) {
    //    rmses[i] = new double[41];
    //}
    //vector<pair<int, int>> ids;
    //load_rmse("N:\\tasks\\reg\\rough_reg1\\test\\xysearch.log",rmses);
    //DSM_REG dsm_reg(dsm_ref_path, dsm_src_path);
    //dsm_reg.START("F:\\tasks\\reg\\rough_reg2\\point\\");
    //dsm_reg.non_min_suppresion(41, 41, rmses, 3, ids);
    //double x = 10.0246-10.0;
    //double y = -3.37484-10.0;
    //for (int i = 0; i < ids.size(); ++i) {
    //    std::cout << "trans[3]: " << ids[i].second * 0.5 + x << "," << ids[i].first * 0.5 + y << ", RMSE: " << rmses[ids[i].first][ids[i].second] << std::endl;
    //}
    //dsm_reg.STEP1_1_ROUGH_ALIGN("N:\\tasks\\reg\\rough_reg1\\test\\");
    //dsm_reg.STEP2_1_COLLECT_TILE_INFOS();
    //dsm_reg.STEP2_2_FILTER_TILE_INFOS();
    //dsm_reg.STEP2_3_ICP_TILE();
    //dsm_reg.STEP2_3_FINE_ALIGN("N:\\tasks\\reg\\rough_reg1\\fine_trans_multi-link.txt");
    //dsm_reg.STEP_END();
    //dsm_reg.EXTRACT_TILES("N:\\tasks\\reg\\rough_reg1\\tile_fine\\");
    //dsm_reg.EXTRACT_TILE_ID(693, "N:\\tasks\\reg\\rough_reg1\\tile_rough_multilink\\");
    //dsm_reg.EXTRACT_TILE_ID(1897, "N:\\tasks\\reg\\rough_reg1\\tile_rough_multilink\\");
    //dsm_reg.EXTRACT_TILE_ID(3602, "N:\\tasks\\reg\\rough_reg1\\tile_rough_multilink\\");
    //dsm_reg.EXTRACT_TILE_ID(7589, "N:\\tasks\\reg\\rough_reg1\\tile_raw_100\\");

    /*
    Test DSM transformation
    */
    //double rot[9], trans[3];
    ////gen_rotation(rot, -0.05,0,0);
    //gs::initDiagonal(rot);
    // rough
    //rot[0] = 1;
    //rot[1] = 4.54618e-06;
    //rot[2] = -6.82097e-05;
    //rot[3] = -4.60764e-06;
    //rot[4] = 1;
    //rot[5] = -0.000901014;
    //rot[6] = 6.82055e-05;
    //rot[7] = 0.000901014;
    //rot[9] = 1;
    //trans[0] = 4.88790000e-02;
    //trans[1] = 1.38450800e-01;
    //trans[2] = 1.81101625e+00;
    //double offx = 525770.479999999980000;
    //double offy = 5390477.879999999900000;
    //rough 1
    //rot[0] = 1;
    //rot[1] = 9.12793074e-05;
    //rot[2] = -7.77366603e-06;
    //rot[3] = -9.12788400e-05;
    //rot[4] = 1;
    //rot[5] = 4.63998405e-05;
    //rot[6] = 7.77784728e-06;
    //rot[7] = -4.63990394e-05;
    //rot[8] = 1;
    //trans[0] = 3.14588829e+00;
    //trans[1] = -7.39058104e-01;
    //trans[2] = -1.22622324e+01;
    //double offx = 517673.960000000020000;
    //double offy = 5380091.139999999700000;

    //rot[0] = 0.999999;
    //rot[1] = -5.2082947275e-05;
    //rot[2] = -0.00086686327519;
    //rot[3] = 5.1265347529e-05;
    //rot[4] = 0.9999995539;
    //rot[5] = -0.0009431658615;
    //rot[6] = 0.00086691201135;
    //rot[7] = 0.0009431210658;
    //rot[8] = 0.99999917949;
    //trans[0] = 5;
    //trans[1] = 5;
    //trans[2] = 2;
    //double offx = 327874.871856689450000;
    //double offy = 4430200.130371093800000;
    //DSM_TRANSFORM dsm_trans(dsm_in_path, dsm_out_path,rot,trans,offx,offy);
    //dsm_trans.START();
	
}