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
set key fixed right top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
unset arrow
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
set ytics  norangelimit autofreq 
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
set xlabel "" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 1.35961e-07 : 1.61074e-07 ] noreverse writeback
set x2range [ 1.79633e-07 : 2.00563e-07 ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ -0.628443 : 0.0668889 ] noreverse writeback
set y2range [ -0.546289 : 0.0273462 ] noreverse writeback
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
GNUTERM = "qt"
file = "figs/data.dat"
dfile = "figs/ddata.dat"
FILE = "figs/powerspectra.dat"
DFILE = "figs/dpowerspectra.dat"
D2FILE = "figs/d2powerspectra.dat"
dataroll = "figs/outroll.dat"
ddataroll = "figs/doutroll.dat"
d2dataroll = "figs/d2outroll.dat"
## Last datafile plotted: "figs/data.dat"
set term png size 1000,1600
set output 'figs/plotting.waveforms.png'
set multiplot
set size 1,.25
set origin 0,.75
set xrange [0:3]
set lmargin screen 0.15
set rmargin screen 0.92
c(x)=(abs(x)<pi/2?cos(x):0./0)
s(x)=(abs(x)<pi?sin(x):0./0)
set xlabel 'frequency [GHz]'
set auto y
set ylabel 'power spectrum'
plot 	FILE u 1:4 notitle,\
	FILE u 1:3 notitle
set origin 0,.5
set xrange [130:170]
set xlabel 'time [ns]'
set ylabel 'signal [V]'
plot 	file u 1:4 notitle,\
	file u 1:3 notitle
set auto y
set ylabel 'derivative signal'
set origin 0,0.25
plot	dfile u 1:4 notitle,\
	dfile u 1:3 notitle
set origin 0,0
set xrange [0:3]
set xlabel 'frequency [GHz]'
set ylabel 'power spectrum'
set auto y
plot 	DFILE u 1:4 notitle,\
	DFILE u 1:3 notitle
unset multiplot

set term png size 1000,1600
set output 'figs/plotting.waveforms.roll.png'
set multiplot 
set lmargin screen 0.15
set rmargin screen 0.92
set size 1,.25
set origin 0,.75
set xrange [-1:1]
set auto y
set xlabel 'time [ns]'
t0=-14.51
set ylabel 'signal [V]'
plot 	dataroll u ($1-t0):4 notitle ,\
	dataroll u ($1-t0):3 notitle 
set origin 0,0.5
set ylabel 'derivative signal'
plot 	ddataroll u ($1-t0):4 notitle,\
	ddataroll u ($1-t0):3 notitle,\
	.4*c(pi*x)*s(2*pi*x) lw 2 lc rgb 'red' title 'convolution kernel'
set origin 0,0.25
set ylabel '2nd derivative signal'
plot	d2dataroll u ($1-t0):4 notitle,\
	d2dataroll u ($1-t0):3 notitle
set origin 0,0
set xrange [0:3]
set xlabel 'frequency [GHz]'
set ylabel 'power spectrum'
set auto y
plot 	D2FILE u 1:4 notitle,\
	D2FILE u 1:3 notitle
unset multiplot
#    EOF
