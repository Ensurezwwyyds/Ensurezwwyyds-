data_folder = fileparts(mfilename('fullpath')); % 获取当前脚本所在目录
assert(exist(fullfile(data_folder, 'ERA5_surface_pressure_201201.nc'), 'file') == 2, '数据文件不存在');
assert(exist(fullfile(data_folder, 'ERA5_temperature_201201.nc'), 'file') == 2, '温度文件不存在');

surface_pressure = ncread(fullfile(data_folder, 'ERA5_surface_pressure_201201.nc'), 'sp');
temperature = ncread(fullfile(data_folder, 'ERA5_temperature_201201.nc'), 't');
pressure_levels = ncread(fullfile(data_folder, 'ERA5_temperature_201201.nc'), 'level') * 100;
lons = ncread([data_folder, 'ERA5_temperature_201201.nc'], 'longitude'); % 读取气温数据的经度信息，为插值做准备
lats = ncread([data_folder, 'ERA5_temperature_201201.nc'], 'latitude'); % 读取气温数据的纬度信息，为插值做准备

sigma_levels = [0.99500 0.97999 0.94995 0.89988 0.82977 0.74468 0.64954 ...
                0.54946 0.45447 0.36948 0.29450 0.22953 0.17457 0.12440 0.0846830 ...
                0.0598005 0.0449337 0.0349146 0.0248800 0.00829901]; % 设置等σ面为0-1之间的单调递减序列

nlev = numel(sigma_levels); % σ层的数量
nlat = size(temperature, 2); % 纬度的维数
nlon = size(temperature, 1); % 经度的维数
temperature_sigma = zeros(nlon, nlat, nlev); % 初始化σ面上的气温场

for i = 1:nlon
    for j = 1:nlat
        sigma_plev = surface_pressure(i,j) * sigma_levels; % 根据公式计算各个格点σ面的气压
        temperature_sigma(i,j,:) = interp1(pressure_levels, squeeze(temperature(i,j,:)), sigma_plev, 'spline'); 
        % 采用样条插值方法，将等压面上的气温场插值到σ面上
    end
end

p_levels = [925 875 825 775 725 675 625 575 550 475 425 375 325 275 225 175 125] * 100; % 设置等压面值

nlev = numel(p_levels); % 等压面层数
temperature_pressure = zeros(nlon, nlat, nlev); % 初始化等压面上的气温场

for i = 1:nlon
    for j = 1:nlat
        pressure_siglev = p_levels / surface_pressure(i,j); % 根据公式计算等压面对应的σ值
        temperature_pressure(i,j,:) = interp1(sigma_levels, squeeze(temperature_sigma(i,j,:)), pressure_siglev, 'spline', nan);
        % 将σ面上的气温场插值到等压面上
    end
end

subplot(3,1,2)
m_proj('Robinson','clo',181); % 设置地图投影方式为Robinson，中央经度181
[cs,h] = m_contour(lons,lats,temperature_sigma(:,:,1)','-k'); % 绘制σ面上的气温场（第一层）
h.LevelStep = 10; % 设置等值线间隔为10
clabel(cs,h,'LabelSpacing',1000,'fontsize',7); % 添加等值线标签
m_coast('linewidth',1,'color',[123,123,123]/255); % 添加海岸线
m_grid('Fontsize',8); % 添加经纬度坐标
text(-3.5,5.2,'(b)','Fontsize',12); % 添加子图标题

subplot(3,1,3)
m_proj('Robinson','clo',181); % 设置地图投影方式为Robinson，中央经度181
[cs,h] = m_contour(lons,lats,temperature_pressure(:,:,p_levels==82500)','-k'); % 绘制插值回到p坐标系的825 hPa气温场
h.LevelStep = 10; % 设置等值线间隔为10
clabel(cs,h,'LabelSpacing',1000,'fontsize',7); % 添加等值线标签
m_coast('linewidth',1,'color',[123,123,123]/255); % 添加海岸线
m_grid('Fontsize',8); % 添加经纬度坐标
text(-3.5,5.2,'(c)','Fontsize',12); % 添加子图标题

max(max(abs(temperature_pressure(:,:,p_levels==82500) - temperature(:,:,pressure_levels==82500)))) 
% 计算坐标转换前后气温数据差值的最大绝对值，定量比较插值带来的影响
