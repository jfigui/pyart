"""
pyart.retrieve.visibility
==================

Functions for visibility and ground echoes estimation from a DEM.

"""

import numpy as np
import warnings
from pyproj import Transformer, Proj

from ..config import get_field_name, get_metadata, get_fillvalue
from ..core import antenna_vectors_to_cartesian as rad_to_cart

nngrid = 0
transformer = 0

def _clip_DEM(dem_grid, xr_proj, yr_proj, extra_m = 5000):
    min_x = np.min(xr_proj)
    min_y = np.min(yr_proj)
    max_x = np.max(xr_proj)
    max_y = np.max(yr_proj)
    
    mask_x = np.logical_and(dem_grid.x['data']  > min_x - extra_m,
                            dem_grid.x['data']  < max_x + extra_m)
    mask_y = np.logical_and(dem_grid.y['data']  > min_y - extra_m,
                            dem_grid.y['data']  < max_y + extra_m)
    
    dem_grid.x['data'] = dem_grid.x['data'][mask_x]
    dem_grid.y['data'] = dem_grid.y['data'][mask_y]
    for f in dem_grid.fields.keys():
        dem_grid.fields[f]['data'] = dem_grid.fields[f]['data'][np.ix_(mask_y,
                                                                      mask_x)]
    return dem_grid

def visibility_processing(radar, dem_grid, frequency, beamwidth, 
                         fill_value = None, 
                         terrain_altitude_field = None,
                         bent_terrain_altitude_field = None,
                         terrain_slope_field = None,
                         terrain_aspect_field = None,
                         theta_angle_field = None,
                         visibility_field = None,
                         min_vis_altitude_field = None,
                         min_vis_theta_field = None,
                         incident_angle_field = None,
                         sigma_0_field = None,
                         ke = 4/3.
                         clip = True):
            
    # parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # parse field names
    if terrain_altitude_field is None:
        terrain_altitude_field = get_field_name('terrain_altitude')
    if bent_terrain_altitude_field is None:
        bent_terrain_altitude_field = get_field_name('bent_terrain_altitude')
    if terrain_slope_field is None:
        terrain_slope_field = get_field_name('terrain_slope')
    if terrain_aspect_field is None:
        terrain_aspect_field = get_field_name('terrain_aspect')
    if theta_angle_field is None:
        theta_angle_field = get_field_name('theta_angle')
    if visibility_field is None:
        visibility_field = get_field_name('visibility')
    if min_vis_altitude_field is None:
        min_vis_altitude_field = get_field_name('min_vis_altitude')
    if incident_angle_field is None:
        incident_angle_field = get_field_name('incident_angle')    
    if sigma_0_field is None:
        sigma_0_field = get_field_name('sigma_0')   
        
    # Define aeqd projection for radar local Cartesian coords
    pargs = Proj(proj="aeqd", lat_0 = radar.latitude['data'][0], 
                 lon_0 = radar.longitude['data'][0], datum = "WGS84", 
                 units="m")    
    
    # Define coordinate transform: (local radar Cart coords) -> (DEM coords)
    transformer = Transformer.from_proj(pargs, dem_grid.projection)
    
    # Get local radar coordinates at elevaiton = 0
    xr, yr, zr = antenna_vectors_to_cartesian(radar.range['data'], 
                          radar.get_azimuth(0), 0, ke = ke)
    
    # Project them in DEM proj
    xr_proj, yr_proj = transformer.transform(xr,yr)
    rad_x = xr_proj[0,0] # radar x coord in DEM proj
    rad_y = yr_proj[0,0] # radar y coord in DEM proj


    # Clip DEM outside radar domain
    if clip:
        dem_grid = _clip_DEM(dem_grid, xr_proj, yr_proj)
        
    res_dem = dem_grid.metadata['resolution']
    xmin_dem = np.min(dem_grid.x['data'])
    ymin_dem = np.min(dem_grid.y['data'])

    # Processing starts here...
    
    # 1) Compute range map
    X_dem, Y_dem = np.meshgrid(dem_grid.x['data'], dem_grid.y['data'])
    range_map = np.sqrt((X_dem - rad_x) ** 2 + (Y_dem - rad_y) ** 2)
    
    # 2) Compute azimuth map
    az_map = (np.arctan2((X_dem - rad_x), (Y_dem - rad_y)) + 2*np.pi) % (2*np.pi)
    az_map *= 180 / np.pi
    
    # 3) Compute bent DEM map
    dem = np.ma.filled(dem_grid.fields['terrain_altitude']['data'], np.nan)
    _, _, zgrid = antenna_to_cartesian(range_map / 1000., az_map, 0, ke = ke)
    bent_map = dem - (zgrid + radar.altitude['data'])
    
    # 4) Compute slope and aspect
    gx = sobel(dem, axis = 1) / (8 * res_dem) # gradient w-e direction
    gy = sobel(dem, axis = 0) /  (8 * res_dem) # gradient s-n direction
    slope_map = np.arctan(np.sqrt(gy**2 + gx**2)) * 180 / np.pi
    aspect_map = (np.arctan2(gy, -gx) + np.pi) * 180 / np.pi
    
    # 5) Compute theta (elevation) angle at topography
    theta_map = (np.arctan2(bent_map, range_map) * 180 / np.pi)
    
    # 6) COmpute visibility map and minimum visible elevation angle map
    # Compute min theta angle with sufficient visibility
    visib_map, minviselev_map = visibility(az_map, range_map, theta_map, 
                    res_dem, xmin_dem, ymin_dem, 100, 0.2, rad_x, rad_y)
    
    # np.save('visib',visib_map)
    # np.save('minviselev',minviselev_map)
    visib_map = np.load('visib.npy')
    minviselev_map = np.load('minviselev.npy')
    # # COmpute min visible altitude
    R = 6371.0 * 1000.0 * ke     # effective radius of earth in meters.
    minvisalt_dem = ((range_map ** 2 + R ** 2 + 2.0 * range_map * R *
           np.sin((minviselev_map + beamwidth_3dB / 2.) * np.pi / 180.)) ** 0.5 - R)
    
    # COmpute effective area
    areaeff_map = res_dem**2 / np.cos(slope_map * np.pi / 180.0)
    
    # Inc angle
    slope  = slope_map* np.pi / 180.0
    aspect = aspect_map * np.pi / 180.0
    zenith = (90. - theta_map * np.pi / 180.0 )
    az     = az_map* np.pi / 180.0 
    
    inc_ang = np.arccos(-( np.sin(slope) * np.sin(zenith) * 
          (np.sin(aspect) * np.sin(az) + np.cos(aspect) * np.cos(az)) + 
                  np.cos(slope) * np.cos(zenith))) * 180 / np.pi
    plt.imshow(visib_map)
    # Compute sigma 0
    sigma0_map = calc_sigma0(inc_ang, freq)
    
    # Compute rcs        
    # rcs_map =  rcs(az_map, theta_map, range_map, areaeff_map, sigma0_map,
    #                     visib_map, res_dem, xmin_dem, ymin_dem, rad_x,
    #                     rad_y, 100, 0.2, [0,1.], 
    #                     beamwidth_3dB, tau)
    # np.save('rcs',rcs_map)
    rcs_map = np.load('rcs.npy')
    
    range_log = 10* np.log10(range_map)
    sigma_map = 10 * np.log10(rcs_map[:,:,0])
    # # Compute clutter dBm
    lambd = 3. / (freq * 10.) 
    pconst = (10 * np.log10(power) + 2 * gain + 20 * np.log10(lambd) - loss - 
              30 * np.log10(4 * np.pi))
       
    clutter_dBm_map = (pconst - 4 * range_log - 2 * atm_att * range_map / 1000. +
                            sigma_map)
    
                     
    # # Compute clutter dBZ
    lambd = 3. / (freq * 10.) 
    dbzconst = (10 * np.log10(16 * np.log(2)) + 40 * np.log10(lambd) - 
                10 * np.log10(tau * 3e8) - 
                20 * np.log10(beamwidth_3dB + np.pi / 180.) - 
                60 * np.log10(np.pi) - 20 * np.log10(mosotti))
    
    convert_dbzm_to_dbz = 180. # 10*log10(1 m^6 / 1 mm^6) = 180
    clutter_dBZ_map = sigma_map - 2 * range_log + dbzconst + convert_dbzm_to_dbz
    
    ranges = np.arange(25, 50000,50)
    az = np.arange(0,360)
    a,b = visibility_angle(minviselev_map, az_map, range_map, [0,1,5,7,10,20], res_dem,
                           xmin_dem, ymin_dem, rad_x, rad_y, ranges, az,
                           beamwidth_3dB,
                           tau)


    theta_dic = get_metadata(theta_angle_field)
    theta_dic['data'] = theta
    
    slope_dic = get_metadata(terrain_slope_field)
    slope_dic['data'] = terrain_slope
    
    aspect_dic = get_metadata(terrain_aspect_field)
    aspect_dic['data'] = terrain_aspect
    
    min_vis_theta_dic = get_metadata(min_vis_theta_field)
    min_vis_theta_dic['data'] = min_vis_theta

    min_vis_altitude_dic = get_metadata(min_vis_altitude_field)
    min_vis_altitude_dic['data'] = min_vis_altitude  
    
    bent_terrain_altitude_dic = get_metadata(bent_terrain_altitude_field)
    bent_terrain_altitude_dic['data'] = dem_bent  

    incident_angle_dic = get_metadata(incident_angle_field)
    incident_angle_dic['data'] = incident_angle  
    
    sigma_0_dic = get_metadata(sigma_0_field)
    sigma_0_dic['data'] = sigma_0  
    
    if quad_pts_range >= 3:
        """ 
        Interpolate range array based on how many quadrature points in range
        are wanted (at least 3)
        For example if quad_pts_range = 5 and original range array is 
        [25,75,125] (bin centers), it will give [0,25,50,50,75,100,125,150]
        i.e. bin 0 - 50 (with center 25) is decomposed into ranges [0,25,50]
        """
        ranges = radar.range['data']
        nrange = len(radar.range['data'])
        dr = np.mean(np.diff(radar.range['data'])) # range res
        intervals = np.arange(ranges[0] - dr / 2, dr * nrange + dr / 2, dr)
        range_resampled = []
        for i in range(len(intervals) - 1):
            range_resampled.extend(np.linspace(intervals[i], intervals[i + 1],
                                               quad_pts_range))
    else:
        # No resampling
        range_resampled = radar.range['data']
    print('done')
    
    # # create parallel computing instance
    # if parallel:
    #     pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
    #     map_ = pool.map
    # else:
    #     map_ = map
        
    # # Loop on fixed angles : el for ppi, az for rhi
    # idx = 0
    # for i, fixangle in enumerate(radar.fixed_angle['data']): 
    #     # Create partial worker func that takes only angle as input
    #     if radar.scan_type == 'ppi':
    #         angles = list((zip(radar.get_azimuth(i), repeat(fixangle))))
    #     elif radar.scan_type == 'rhi':
    #         angles = list((zip(repeat(fixangle), radar.get_elevation(i))))
            
    #     partialworker = partial(_worker_function, 
    #                             ranges = range_resampled,
    #                             rad_alt = radar.altitude['data'],
    #                             dem_data = dem_data,
    #                             ke = ke,
    #                             quad_pts_range = quad_pts_range,
    #                             quad_pts_GH = quad_pts,
    #                             weights = weights)
   
    #     results = list(map_(partialworker, angles))
    #     visibility[idx : idx + len(results), :] = results
    #     idx += len(results)
        
    #     if parallel:
    #         pool.close()
    #         pool.join()
    
    visibility_dic = get_metadata(visibility_field)
    visibility_dic['data'] = visibility  
    
    return (bent_terrain_altitude_dic, slope_dic, aspect_dic, 
            theta_dic, min_vis_theta_dic, min_vis_altitude_dic,
            visibility_dic, incident_angle_dic, sigma_0_dic)
