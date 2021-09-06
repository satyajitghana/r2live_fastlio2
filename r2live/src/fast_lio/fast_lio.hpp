#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** EKF inputs and output ***/
extern MeasureGroup Measures;
extern esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
extern state_ikfom state_point;
extern vect3 pos_lid;

class Fast_lio
{
public:
    std::mutex  m_mutex_lio_process;


	float res_last[100000] = {0.0};
	float DET_RANGE = 300.0f;
	const float MOV_THRESHOLD = 1.5f;

	double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
    int time_log_counter = 0;

	// IMU related variables
	std::mutex mtx_buffer;
	std::condition_variable sig_buffer;

	string root_dir = ROOT_DIR;
	string map_file_path, lid_topic, imu_topic;

	double res_mean_last = 0.05, total_residual = 0.0;
	double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
	double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
	double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
	double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
	int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
	int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
	bool   point_selected_surf[100000] = {0};
	bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
	bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

	vector<vector<int>>  pointSearchInd_surf; 
	vector<BoxPointType> cub_needrm;
	vector<PointVector>  Nearest_Points; 
	vector<double>       extrinT(3, 0.0);
	vector<double>       extrinR(9, 0.0);
	deque<double>                     time_buffer;
	deque<PointCloudXYZI::Ptr>        lidar_buffer;
	deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

	PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
	PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
	PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
	PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
	PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
	PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
	PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
	PointCloudXYZI::Ptr _featsArray;

	pcl::VoxelGrid<PointType> downSizeFilterSurf;
	pcl::VoxelGrid<PointType> downSizeFilterMap;

	KD_TREE ikdtree;

	V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
	V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
	V3D euler_cur;
	V3D position_last(Zero3d);
	V3D Lidar_T_wrt_IMU(Zero3d);
	M3D Lidar_R_wrt_IMU(Eye3d);


	nav_msgs::Path path;
	nav_msgs::Odometry odomAftMapped;
	geometry_msgs::Quaternion geoQuat;
	geometry_msgs::PoseStamped msg_body_pose;

    ros::Subscriber sub_pcl;
    ros::Subscriber sub_imu;
    ros::Publisher pubLaserCloudFull;
    ros::Publisher pubLaserCloudFull_body;
    ros::Publisher pubLaserCloudEffect;
    ros::Publisher pubLaserCloudMap;
    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubPath;

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";

    ofstream fout_pre, fout_out, fout_dbg;

    ros::NodeHandle             nh;

	void SigHandle(int sig)
	{
		flg_exit = true;
		ROS_WARN("catch sig %d", sig);
		sig_buffer.notify_all();
	}

	void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
	{
		V3D p_body(pi->x, pi->y, pi->z);
		V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

		po->x = p_global(0);
		po->y = p_global(1);
		po->z = p_global(2);
		po->intensity = pi->intensity;
	}


	void pointBodyToWorld(PointType const * const pi, PointType * const po)
	{
		V3D p_body(pi->x, pi->y, pi->z);
		V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

		po->x = p_global(0);
		po->y = p_global(1);
		po->z = p_global(2);
		po->intensity = pi->intensity;
	}

	template<typename T>
	void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
	{
		V3D p_body(pi[0], pi[1], pi[2]);
		V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

		po[0] = p_global(0);
		po[1] = p_global(1);
		po[2] = p_global(2);
	}

	void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
	{
		V3D p_body(pi->x, pi->y, pi->z);
		V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

		po->x = p_global(0);
		po->y = p_global(1);
		po->z = p_global(2);
		po->intensity = pi->intensity;
	}

	void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
	{
		V3D p_body_lidar(pi->x, pi->y, pi->z);
		V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

		po->x = p_body_imu(0);
		po->y = p_body_imu(1);
		po->z = p_body_imu(2);
		po->intensity = pi->intensity;
	}

	void points_cache_collect()
	{
		PointVector points_history;
		ikdtree.acquire_removed_points(points_history);
		for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
	}

	BoxPointType LocalMap_Points;
	bool Localmap_Initialized = false;
	void lasermap_fov_segment()
	{
		cub_needrm.clear();
		kdtree_delete_counter = 0;
		kdtree_delete_time = 0.0;    
		pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
		V3D pos_LiD = pos_lid;
		if (!Localmap_Initialized){
			for (int i = 0; i < 3; i++){
				LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
				LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
			}
			Localmap_Initialized = true;
			return;
		}
		float dist_to_map_edge[3][2];
		bool need_move = false;
		for (int i = 0; i < 3; i++){
			dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
			dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
			if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
		}
		if (!need_move) return;
		BoxPointType New_LocalMap_Points, tmp_boxpoints;
		New_LocalMap_Points = LocalMap_Points;
		float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
		for (int i = 0; i < 3; i++){
			tmp_boxpoints = LocalMap_Points;
			if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
				New_LocalMap_Points.vertex_max[i] -= mov_dist;
				New_LocalMap_Points.vertex_min[i] -= mov_dist;
				tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
				cub_needrm.push_back(tmp_boxpoints);
			} else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
				New_LocalMap_Points.vertex_max[i] += mov_dist;
				New_LocalMap_Points.vertex_min[i] += mov_dist;
				tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
				cub_needrm.push_back(tmp_boxpoints);
			}
		}
		LocalMap_Points = New_LocalMap_Points;

		points_cache_collect();
		double delete_begin = omp_get_wtime();
		if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
		kdtree_delete_time = omp_get_wtime() - delete_begin;
	}

	void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
	{
		mtx_buffer.lock();
		scan_count ++;
		double preprocess_start_time = omp_get_wtime();
		if (msg->header.stamp.toSec() < last_timestamp_lidar)
		{
			ROS_ERROR("lidar loop back, clear buffer");
			lidar_buffer.clear();
		}

		PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
		p_pre->process(msg, ptr);
		lidar_buffer.push_back(ptr);
		time_buffer.push_back(msg->header.stamp.toSec());
		last_timestamp_lidar = msg->header.stamp.toSec();
		s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
		mtx_buffer.unlock();
		sig_buffer.notify_all();
	}

	double timediff_lidar_wrt_imu = 0.0;
	bool   timediff_set_flg = false;
	void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
	{
		mtx_buffer.lock();
		double preprocess_start_time = omp_get_wtime();
		scan_count ++;
		if (msg->header.stamp.toSec() < last_timestamp_lidar)
		{
			ROS_ERROR("lidar loop back, clear buffer");
			lidar_buffer.clear();
		}
		last_timestamp_lidar = msg->header.stamp.toSec();
		
		if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
		{
			printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
		}

		if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
		{
			timediff_set_flg = true;
			timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
			printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
		}

		PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
		p_pre->process(msg, ptr);
		lidar_buffer.push_back(ptr);
		time_buffer.push_back(last_timestamp_lidar);
		
		s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
		mtx_buffer.unlock();
		sig_buffer.notify_all();
	}

	void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
	{
		publish_count ++;
		// cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
		sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

		if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
		{
			msg->header.stamp = \
			ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
		}

		double timestamp = msg->header.stamp.toSec();

		mtx_buffer.lock();

		if (timestamp < last_timestamp_imu)
		{
			ROS_WARN("imu loop back, clear buffer");
			imu_buffer.clear();
		}

		last_timestamp_imu = timestamp;

		imu_buffer.push_back(msg);
		mtx_buffer.unlock();
		sig_buffer.notify_all();
	}

	double lidar_mean_scantime = 0.0;
	int    scan_num = 0;
	bool sync_packages(MeasureGroup &meas)
	{
		if (lidar_buffer.empty() || imu_buffer.empty()) {
			return false;
		}

		/*** push a lidar scan ***/
		if(!lidar_pushed)
		{
			meas.lidar = lidar_buffer.front();
			meas.lidar_beg_time = time_buffer.front();
			if (meas.lidar->points.size() <= 1) // time too little
			{
				lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
				ROS_WARN("Too few input point cloud!\n");
			}
			else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
			{
				lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
			}
			else
			{
				scan_num ++;
				lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
				lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
			}

			meas.lidar_end_time = lidar_end_time;

			lidar_pushed = true;
		}

		if (last_timestamp_imu < lidar_end_time)
		{
			return false;
		}

		/*** push imu data, and pop from imu buffer ***/
		double imu_time = imu_buffer.front()->header.stamp.toSec();
		meas.imu.clear();
		while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
		{
			imu_time = imu_buffer.front()->header.stamp.toSec();
			if(imu_time > lidar_end_time) break;
			meas.imu.push_back(imu_buffer.front());
			imu_buffer.pop_front();
		}

		lidar_buffer.pop_front();
		time_buffer.pop_front();
		lidar_pushed = false;
		return true;
	}

    std::thread m_thread_process;
    Fast_lio()
    {
        printf_line;
        pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
        pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
        pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
        pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
        sub_imu = nh.subscribe("/livox/imu", 2000000, &Fast_lio::imu_cbk, this, ros::TransportHints().tcpNoDelay());
        sub_pcl = nh.subscribe("/laser_cloud_flat", 2000000, &Fast_lio::feat_points_cbk, this, ros::TransportHints().tcpNoDelay());

		get_ros_parameter(nh, "fastlio/publish/path_en", path_en, true);
		get_ros_parameter(nh, "fastlio/publish/scan_publish_en", scan_pub_en, true);
		get_ros_parameter(nh, "fastlio/publish/dense_publish_en", dense_pub_en, true);
		get_ros_parameter(nh, "fastlio/publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
		get_ros_parameter(nh, "fastlio/max_iteration", NUM_MAX_ITERATIONS, 4);
		get_ros_parameter(nh, "fastlio/map_file_path", map_file_path, "");
		get_ros_parameter(nh, "fastlio/common/lid_topic", lid_topic, "/livox/lidar");
		get_ros_parameter(nh, "fastlio/common/imu_topic", imu_topic, "/livox/imu");
		get_ros_parameter(nh, "fastlio/common/time_sync_en", time_sync_en, false);
		get_ros_parameter(nh, "fastlio/filter_size_corner", filter_size_corner_min, 0.5);
		get_ros_parameter(nh, "fastlio/filter_size_surf", filter_size_surf_min, 0.5);
		get_ros_parameter(nh, "fastlio/filter_size_map", filter_size_map_min, 0.5);
		get_ros_parameter(nh, "fastlio/cube_side_length", cube_len, 200);
		get_ros_parameter(nh, "fastlio/mapping/det_range", DET_RANGE, 300.f);
		get_ros_parameter(nh, "fastlio/mapping/fov_degree", fov_deg, 180);
		get_ros_parameter(nh, "fastlio/mapping/gyr_cov", gyr_cov, 0.1);
		get_ros_parameter(nh, "fastlio/mapping/acc_cov", acc_cov, 0.1);
		get_ros_parameter(nh, "fastlio/mapping/b_gyr_cov", b_gyr_cov, 0.0001);
		get_ros_parameter(nh, "fastlio/mapping/b_acc_cov", b_acc_cov, 0.0001);
		get_ros_parameter(nh, "fastlio/preprocess/blind", p_pre->blind, 0.01);
		get_ros_parameter(nh, "fastlio/preprocess/lidar_type", p_pre->lidar_type, AVIA);
		get_ros_parameter(nh, "fastlio/preprocess/scan_line", p_pre->N_SCANS, 16);
		get_ros_parameter(nh, "fastlio/preprocess/point_filter_num", p_pre->point_filter_num, 2);
		get_ros_parameter(nh, "fastlio/feature_extract_enable", p_pre->feature_enabled, false);
		get_ros_parameter(nh, "fastlio/runtime_pos_log_enable", runtime_pos_log, 0);
		get_ros_parameter(nh, "fastlio/mapping/extrinsic_est_en", extrinsic_est_en, true);
		get_ros_parameter(nh, "fastlio/pcd_save/pcd_save_en", pcd_save_en, false);
		get_ros_parameter(nh, "fastlio/pcd_save/interval", pcd_save_interval, -1);
		get_ros_parameter(nh, "fastlio/mapping/extrinsic_T", extrinT, vector<double>());
		get_ros_parameter(nh, "fastlio/mapping/extrinsic_R", extrinR, vector<double>());
		cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
        printf_line;

		/*** variables definition ***/
		int effect_feat_num = 0, frame_num = 0;
		double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
		bool flg_EKF_converged, EKF_stop_flg = 0;
		
		FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
		HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

		_featsArray.reset(new PointCloudXYZI());

		memset(point_selected_surf, true, sizeof(point_selected_surf));
		memset(res_last, -1000.0f, sizeof(res_last));
		downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
		downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
		memset(point_selected_surf, true, sizeof(point_selected_surf));
		memset(res_last, -1000.0f, sizeof(res_last));

		Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
		Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
		p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
		p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
		p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
		p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
		p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

		double epsi[23] = {0.001};
		fill(epsi, epsi+23, 0.001);
		kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        printf_line;
        m_thread_process = std::thread(&Fast_lio::process, this);
        printf_line;
    }
    ~Fast_lio(){};

    int process()
    {
        nav_msgs::Path path;
        path.header.stamp = ros::Time::now();
        path.header.frame_id = "/world";

    	fp = fopen(pos_log_dir.c_str(),"w");
		fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
		fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
		fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
		if (fout_pre && fout_out)
			cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
		else
			cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;


		ros::Rate rate(5000);
		bool status = ros::ok();
		while (status)
		{
			if (flg_exit) break;
			ros::spinOnce();
			if(sync_packages(Measures)) 
			{
				if (flg_first_scan)
				{
					first_lidar_time = Measures.lidar_beg_time;
					p_imu->first_lidar_time = first_lidar_time;
					flg_first_scan = false;
					continue;
				}

				double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

				match_time = 0;
				kdtree_search_time = 0.0;
				solve_time = 0;
				solve_const_H_time = 0;
				svd_time   = 0;
				t0 = omp_get_wtime();

				p_imu->Process(Measures, kf, feats_undistort);
				state_point = kf.get_x();
				pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

				if (feats_undistort->empty() || (feats_undistort == NULL))
				{
					ROS_WARN("No point, skip this scan!\n");
					continue;
				}

				flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
								false : true;
				/*** Segment the map in lidar FOV ***/
				lasermap_fov_segment();

				/*** downsample the feature points in a scan ***/
				downSizeFilterSurf.setInputCloud(feats_undistort);
				downSizeFilterSurf.filter(*feats_down_body);
				t1 = omp_get_wtime();
				feats_down_size = feats_down_body->points.size();
				/*** initialize the map kdtree ***/
				if(ikdtree.Root_Node == nullptr)
				{
					if(feats_down_size > 5)
					{
						ikdtree.set_downsample_param(filter_size_map_min);
						feats_down_world->resize(feats_down_size);
						for(int i = 0; i < feats_down_size; i++)
						{
							pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
						}
						ikdtree.Build(feats_down_world->points);
					}
					continue;
				}
				int featsFromMapNum = ikdtree.validnum();
				kdtree_size_st = ikdtree.size();
				
				// cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

				/*** ICP and iterated Kalman filter update ***/
				if (feats_down_size < 5)
				{
					ROS_WARN("No point, skip this scan!\n");
					continue;
				}
				
				normvec->resize(feats_down_size);
				feats_down_world->resize(feats_down_size);

				V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
				fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
				<<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

				if(0) // If you need to see map point, change to "if(1)"
				{
					PointVector ().swap(ikdtree.PCL_Storage);
					ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
					featsFromMap->clear();
					featsFromMap->points = ikdtree.PCL_Storage;
				}

				pointSearchInd_surf.resize(feats_down_size);
				Nearest_Points.resize(feats_down_size);
				int  rematch_num = 0;
				bool nearest_search_en = true; //

				t2 = omp_get_wtime();
				
				/*** iterated state estimation ***/
				double t_update_start = omp_get_wtime();
				double solve_H_time = 0;
				kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
				state_point = kf.get_x();
				euler_cur = SO3ToEuler(state_point.rot);
				pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
				geoQuat.x = state_point.rot.coeffs()[0];
				geoQuat.y = state_point.rot.coeffs()[1];
				geoQuat.z = state_point.rot.coeffs()[2];
				geoQuat.w = state_point.rot.coeffs()[3];

				double t_update_end = omp_get_wtime();

				/******* Publish odometry *******/
				publish_odometry(pubOdomAftMapped);

				/*** add the feature points to map kdtree ***/
				t3 = omp_get_wtime();
				map_incremental();
				t5 = omp_get_wtime();
				
				/******* Publish points *******/
				if (path_en)                         publish_path(pubPath);
				if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
				if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
				// publish_effect_world(pubLaserCloudEffect);
				// publish_map(pubLaserCloudMap);

				/*** Debug variables ***/
				if (runtime_pos_log)
				{
					frame_num ++;
					kdtree_size_end = ikdtree.size();
					aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
					aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
					aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
					aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
					aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
					aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
					T1[time_log_counter] = Measures.lidar_beg_time;
					s_plot[time_log_counter] = t5 - t0;
					s_plot2[time_log_counter] = feats_undistort->points.size();
					s_plot3[time_log_counter] = kdtree_incremental_time;
					s_plot4[time_log_counter] = kdtree_search_time;
					s_plot5[time_log_counter] = kdtree_delete_counter;
					s_plot6[time_log_counter] = kdtree_delete_time;
					s_plot7[time_log_counter] = kdtree_size_st;
					s_plot8[time_log_counter] = kdtree_size_end;
					s_plot9[time_log_counter] = aver_time_consu;
					s_plot10[time_log_counter] = add_point_size;
					time_log_counter ++;
					printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
					ext_euler = SO3ToEuler(state_point.offset_R_L_I);
					fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
					<<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
					dump_lio_state_to_log(fp);
				}
			}
        status = ros::ok();
        rate.sleep();
		}

		return 0;
	}

	int process_increments = 0;
	void map_incremental()
	{
		PointVector PointToAdd;
		PointVector PointNoNeedDownsample;
		PointToAdd.reserve(feats_down_size);
		PointNoNeedDownsample.reserve(feats_down_size);
		for (int i = 0; i < feats_down_size; i++)
		{
			/* transform to world frame */
			pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
			/* decide if need add to map */
			if (!Nearest_Points[i].empty() && flg_EKF_inited)
			{
				const PointVector &points_near = Nearest_Points[i];
				bool need_add = true;
				BoxPointType Box_of_Point;
				PointType downsample_result, mid_point; 
				mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
				mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
				mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
				float dist  = calc_dist(feats_down_world->points[i],mid_point);
				if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
					PointNoNeedDownsample.push_back(feats_down_world->points[i]);
					continue;
				}
				for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
				{
					if (points_near.size() < NUM_MATCH_POINTS) break;
					if (calc_dist(points_near[readd_i], mid_point) < dist)
					{
						need_add = false;
						break;
					}
				}
				if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
			}
			else
			{
				PointToAdd.push_back(feats_down_world->points[i]);
			}
		}

		double st_time = omp_get_wtime();
		add_point_size = ikdtree.Add_Points(PointToAdd, true);
		ikdtree.Add_Points(PointNoNeedDownsample, false); 
		add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
		kdtree_incremental_time = omp_get_wtime() - st_time;
	}

	PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
	PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
	void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
	{
		if(scan_pub_en)
		{
			PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
			int size = laserCloudFullRes->points.size();
			PointCloudXYZI::Ptr laserCloudWorld( \
							new PointCloudXYZI(size, 1));

			for (int i = 0; i < size; i++)
			{
				RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
									&laserCloudWorld->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudmsg;
			pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
			laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
			laserCloudmsg.header.frame_id = "camera_init";
			pubLaserCloudFull.publish(laserCloudmsg);
			publish_count -= PUBFRAME_PERIOD;
		}

		/**************** save map ****************/
		/* 1. make sure you have enough memories
		/* 2. noted that pcd save will influence the real-time performences **/
		if (pcd_save_en)
		{
			int size = feats_undistort->points.size();
			PointCloudXYZI::Ptr laserCloudWorld( \
							new PointCloudXYZI(size, 1));

			for (int i = 0; i < size; i++)
			{
				RGBpointBodyToWorld(&feats_undistort->points[i], \
									&laserCloudWorld->points[i]);
			}
			*pcl_wait_save += *laserCloudWorld;

			static int scan_wait_num = 0;
			scan_wait_num ++;
			if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
			{
				pcd_index ++;
				string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
				pcl::PCDWriter pcd_writer;
				cout << "current scan saved to /PCD/" << all_points_dir << endl;
				pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
				pcl_wait_save->clear();
				scan_wait_num = 0;
			}
		}
	}

	void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
	{
		int size = feats_undistort->points.size();
		PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

		for (int i = 0; i < size; i++)
		{
			RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
								&laserCloudIMUBody->points[i]);
		}

		sensor_msgs::PointCloud2 laserCloudmsg;
		pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
		laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudmsg.header.frame_id = "body";
		pubLaserCloudFull_body.publish(laserCloudmsg);
		publish_count -= PUBFRAME_PERIOD;
	}

	void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
	{
		PointCloudXYZI::Ptr laserCloudWorld( \
						new PointCloudXYZI(effct_feat_num, 1));
		for (int i = 0; i < effct_feat_num; i++)
		{
			RGBpointBodyToWorld(&laserCloudOri->points[i], \
								&laserCloudWorld->points[i]);
		}
		sensor_msgs::PointCloud2 laserCloudFullRes3;
		pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
		laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudFullRes3.header.frame_id = "camera_init";
		pubLaserCloudEffect.publish(laserCloudFullRes3);
	}

	void publish_map(const ros::Publisher & pubLaserCloudMap)
	{
		sensor_msgs::PointCloud2 laserCloudMap;
		pcl::toROSMsg(*featsFromMap, laserCloudMap);
		laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
		laserCloudMap.header.frame_id = "camera_init";
		pubLaserCloudMap.publish(laserCloudMap);
	}

	template<typename T>
	void set_posestamp(T & out)
	{
		out.pose.position.x = state_point.pos(0);
		out.pose.position.y = state_point.pos(1);
		out.pose.position.z = state_point.pos(2);
		out.pose.orientation.x = geoQuat.x;
		out.pose.orientation.y = geoQuat.y;
		out.pose.orientation.z = geoQuat.z;
		out.pose.orientation.w = geoQuat.w;
		
	}
};