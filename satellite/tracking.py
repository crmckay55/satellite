from skyfield.api import load, wgs84, EarthSatellite


satellite_url = 'http://celestrak.org/NORAD/elements/NOAA.txt'
satellites = load.tle_file(satellite_url)
print('Loaded', len(satellites), ' satellites')

# Create a timescale and ask the current time.
ts = load.timescale()
t = ts.now()

by_number = {sat.model.satnum: sat for sat in satellites}
s = by_number[33591]
print(s)

calgary = wgs84.latlon(50.903756, -114.0478306, 1042)

t0 = ts.utc(2022, 8, 15)
t1 = ts.utc(2022, 8, 17)

t, events = s.find_events(calgary, t0, t1, altitude_degrees=30)

# polar plot: https://math.stackexchange.com/questions/2820373/how-to-plot-a-satellite-trace-wrt-a-ground-station-using-polar-plot-view

for ti, event in zip(t, events):
    name = ('rise above 30 degrees', 'culminate', 'set below 30 degrees')[event]
    t_event = ti.utc_strftime('%Y %b %d %H:%M:%S')
    print(t_event, name)



