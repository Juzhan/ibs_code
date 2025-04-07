from scipy.spatial import Voronoi
import trimesh
import pyvista as pv
import numpy as np


def compute_ibs(points_1, points_2, radius):
    pts_0 = pv.wrap(points_1)
    pts_1 = pv.wrap(points_2)

    n_0, n_1 = pts_0.n_points, pts_1.n_points

    # radius = 0.18
    center = (np.array(pts_1.center) + np.array(pts_0.center)) / 2

    poly_scene = pts_0 + pts_1
    vor = Voronoi(poly_scene.points)
    ridge_idx = []

    ridge_idx.append(np.where((
        ((vor.ridge_points<n_0).sum(-1)==1)
    ))[0])

    ridge_idx = np.concatenate(ridge_idx)
    ridge_idx = [i for i in ridge_idx if -1 not in vor.ridge_vertices[i]]

    polys = np.asarray(vor.ridge_vertices,dtype=object)[ridge_idx]
    polys = np.concatenate(list(map(lambda x:[len(x),]+x,polys)))

    # poly_ibs = pv.PolyData(vor.vertices,polys).triangulate()
    # vertices_idx = np.where(((vor.vertices-center)**2).sum(-1)>radius**2)
    # poly_ibs,_ = poly_ibs.remove_points(vertices_idx)
    # poly_ibs.clean(inplace=True)


    poly_ibs = pv.PolyData(vor.vertices,polys)
    poly_ibs= poly_ibs.clip_surface(pv.Sphere(radius,center)).triangulate()
    poly_ibs.clean(inplace=True)
    
    return poly_ibs


if __name__ == '__main__':
    import pyvista as pv
    import numpy as np
    from scipy.spatial import cKDTree

    # 创建示例点云数据（替换为实际数据）
    # points_a = np.random.rand(100, 3)  # 点云a的坐标（100个点）
    # points_b = np.random.rand(500, 3)  # 点云b的坐标（500个点）

    cloud_a = pv.wrap(trimesh.load(f'./models/H_src_mesh.obj',process=False,force='mesh'))
    cloud_b = pv.wrap(trimesh.load(f'./models/O_src_mesh.obj',process=False,force='mesh'))
    cloud_c = pv.wrap(trimesh.load(f'./models/t_pose.obj',process=False,force='mesh'))

    a = pv.wrap(trimesh.load(f'./models/H_src_mesh.obj',process=False,force='mesh').sample(3000))
    b = pv.wrap(trimesh.load(f'./models/O_src_mesh.obj',process=False,force='mesh').sample(3000))

    ibs = compute_ibs( a, b, 1 )

    target = ibs

    # 构建点云b的KDTree加速查询
    kdtree_b = cKDTree(target.points)

    # 计算点云a中每个点到b的最近距离（向量化操作，高效）
    distances, _ = kdtree_b.query(cloud_a.points, k=1)
    
    # lambda_param = 1.0  # 调节概率集中程度
    # probabilities = np.exp(-lambda_param * distances)
    # probabilities /= probabilities.sum()  # 归一化

    # 方案2：距离越大概率越高（线性反比例）
    probabilities = 1 / (distances + 1e-8)  # 避免除零
    probabilities /= probabilities.sum()
        
    # --- 基于概率采样点 ---
    
    n_samples = 500  # 采样数量
    replace = False   # 是否允许重复采样
    sampled_indices = np.random.choice(
        len(cloud_a.points), 
        size=n_samples, 
        p=probabilities, 
        replace=replace
    )
    sampled_points = cloud_a.points[sampled_indices]

    # 保存采样结果为新点云
    sampled_cloud = pv.PolyData(sampled_points)
    # sampled_cloud.save("sampled_points.ply")

    cloud_c = cloud_c.points[sampled_indices]
    cloud_c = pv.PolyData(cloud_c)
    
    # 创建可视化窗口
    plotter = pv.Plotter()


    # 添加原始点云和采样点
    plotter.add_mesh(cloud_a, color='red', point_size=3, opacity=0.3, label='Cloud A')
    plotter.add_mesh(cloud_b, color='blue', point_size=3, opacity=0.3, label='Cloud B')
    plotter.add_mesh(ibs, color='orange', point_size=3, opacity=0.3, label='IBS')
    plotter.add_mesh(sampled_cloud, color='green', point_size=10, label='Sampled Points')

    # 添加连线显示采样点与对应的最近点
    for pt in sampled_points:
        _, closest_idx = kdtree_b.query(pt, k=1)
        closest_pt = target.points[closest_idx]
        line = pv.Line(pt, closest_pt)
        plotter.add_mesh(line, color='pink', line_width=2)

    plotter.add_legend()
    plotter.show()