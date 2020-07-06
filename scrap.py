from numpy import interp

x1 = 550
x2 = 320
h_pix = [0,640]
h_fov = [0,62.8]

x_shift = x1 - x2
theta = interp(x_shift, h_pix, h_fov)
print(theta)
print(0.098 * (550- 320))

DC_forward = [0, 109.7]
DC_backward = [-109.7, 0]

DC_time_range = [0, 0.209]  ## for full speed for turning 62.8 degrees to positive
DC_time_range_rev = [0.209, 0]

dc = interp(theta, DC_forward, DC_time_range)
print('dc' ,dc)