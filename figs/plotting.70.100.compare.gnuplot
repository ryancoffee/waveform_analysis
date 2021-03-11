#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 8    last modified 2019-12-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2019
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
# set terminal qt 0 font "Sans,9"
# set output
unset clip points
set clip one
unset clip two
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.000 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "% h" 
set format y "% h" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
unset grid
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key title "" center
set key fixed left top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
set label 1 "2.1 ns\n1.55 eV" at 114.000, 55.0000, 0.00000 center norotate front nopoint 
set label 2 "0.6 ns\n0.5 eV" at 115.000, 600.000, 0.00000 center norotate front nopoint
set label 3 "1.332 ns\n1.55 eV" at 94.5000, 14.0000, 0.00000 center norotate front nopoint
set label 4 "0.45 ns\n0.5 eV" at 95.2500, 250.000, 0.00000 center norotate front nopoint 
unset arrow
set arrow 1 from 112.900, 80.0000, 0.00000 to 115.000, 80.0000, 0.00000 nohead back nofilled linewidth 2.000 dashtype solid
set arrow 2 from 114.700, 350.000, 0.00000 to 115.280, 350.000, 0.00000 nohead back nofilled linewidth 2.000 dashtype solid
set arrow 3 from 93.8550, 20.0000, 0.00000 to 95.1870, 20.0000, 0.00000 nohead back nofilled linewidth 2.000 dashtype solid
set arrow 4 from 95.0850, 150.000, 0.00000 to 95.5290, 150.000, 0.00000 nohead back nofilled linewidth 2.000 dashtype solid
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
set style textbox transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding default
unset polar
unset parametric
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels 5
set cntrparam levels auto
set cntrparam firstlinetype 0 unsorted
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data histeps
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border in scale 1,0.5 mirror norotate  autojustify
set xtics  norangelimit autofreq 
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit logscale autofreq 
set ztics border in scale 1,0.5 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border in scale 1,0.5 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis in scale 1,0.5 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "" 
set title  font "" textcolor lt -1 norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" textcolor lt -1 norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "time-of-flight [ns]" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 80.0000 : 130.000 ] noreverse writeback
set x2range [ 73.6835 : 122.191 ] noreverse writeback
set ylabel "counts [arb. units]" 
set ylabel  offset character 1, 0, 0 font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ 1.00000 : 1000.00 ] noreverse writeback
set y2range [ -4.05886 : 291.110 ] noreverse writeback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse writeback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse writeback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse writeback
unset logscale
set logscale y 10
unset jitter
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "en_US.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath 
set fontpath 
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
file70(p,u) = sprintf('%s/2020_08_14_17_49_15_logic_compare_upscale%i.hist',p,u)
file30(p,u) = sprintf('%s/2020_09_04_15_59_06_logic_compare_upscale%i.hist',p,u)
file100(p,u) = sprintf('%s/2020_08_13_18_54_57_logic_compare_upscale%i.hist',p,u)
GNUTERM = "qt"
path30 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1930V_2330V/processed"
path70 = "/nvme/hpl-CookieBox_testing/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_1970V_2370V/processed"
path100 = "/nvme/hpl-CookieBox_testing/08_13_2020/walkwayToF_100V_NoAmp/400V_2000V_2400V/processed"
## Last datafile plotted: "/nvme/hpl-CookieBox_testing/08_13_2020/walkwayToF_100V_NoAmp/400V_2000V_2400V/processed/2020_08_13_18_54_57_logic_compare_upscale8.hist"
set term png size 900,500
set xrange [ 90.0000 : 120.000 ] noreverse writeback
set output 'figs/plotting.70.100.zoom.png'
plot 	file100(path100,8) title '100 V pusher',\
	file70(path70,8) title '70 V pusher'
set xrange [ 80.0000 : 190.000 ] noreverse writeback
set term png size 900,500
set output 'figs/plotting.30.70.100.png'
unset label
unset arrow
plot 	file100(path100,8) title '100 V pusher',\
	file70(path70,8) title '70 V pusher',\
	file30(path30,8) title '30 V pusher'
	
#    EOF
