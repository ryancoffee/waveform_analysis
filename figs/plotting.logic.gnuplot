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
set style data linespoints
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
set xrange [ 23.0000 : 35.0000 ] noreverse writeback
set x2range [ 63.1040 : 82.2887 ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ -0.191078 : 0.135433 ] noreverse writeback
set y2range [ -0.191078 : 0.135433 ] noreverse writeback
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
file(x)=sprintf('waveforms/08_20_2020/backAndTubeAt_100V_400_2200_2600/frontAt_100V/2020_08_20_19_45_07%s',x)
f(x) = a + b*x
g(x) = aa + bb*x
h(x) = aaa + bbb*x
GNUTERM = "qt"
a = 20.8617396820322
b = -0.713994446530576
FIT_CONVERGED = 1
FIT_NDF = 3
FIT_STDFIT = 0.000174914009130131
FIT_WSSR = 9.17847317699267e-08
FIT_P = 0.999999999992604
FIT_NITER = 7
b_err = 0.0214444508203186
x0 = 1.1
a_err = 0.626178422952555
aa = 30.2035224233033
bb = -1.25682337319522
aa_err = 0.812697280488235
bb_err = 0.033809536393611
aaa = 2.70466391653539
bbb = -0.0854477061923268
aaa_err = 0.0700258792927525
bbb_err = 0.0022125066540874
## Last datafile plotted: "waveforms/08_20_2020/backAndTubeAt_100V_400_2200_2600/frontAt_100V/2020_08_20_19_45_07.samplesig"
do for [r=1:10] {
set xrange [26:32]
set term png size 800,600
set output sprintf('figs/plotting.logic.%i.png',r)
set multiplot
set origin 0,.66
set size 1,.33
set lmargin screen .12
set rmargin screen .9
set ylabel 'sig [V]'
set yrange [-.3:.05]
set style data histeps
set key bottom right
set xlabel 'ToF [ns]'
plot file('.samplesig') u ($1-35):r lw 2 title 'signal'
set ylabel 'convolutions [arb]'
set origin 0,.33
set yrange [-1.25:1.25]
set key top right
plot file('.0.back') u ($1-35):r title 'filt', file('.0.dback') u ($1-35):r title 'd/dt filt', file('.0.ddback') u ($1-35):r title 'd^2/dt^2 filt'
set ylabel 'logic [arb]'
set origin 0,0
set yrange [-.25:.25]
set key bottom right
if (r==8) {
a               = 18.9652 
b               = -0.698379
aa              = 19.1943 
bb              = -0.690818
aaa             = 65.3086
bbb             = -2.20678
plot f(x) lc -1 title '4pt lin fit', g(x) lc -1 notitle, h(x) lc -1 notitle,\
file('.0.dlogic') u (exp($1)):r lw 2 lc 1 title 'logic'
#,f(x) lc -1 notitle ,g(x) lc -1 notitle ,h(x) lc -1 notitle
} else {
plot file('.0.dlogic') u (exp($1)):r title 'logic'
}
unset multiplot
}
## fit [31.6:31.71] h(x) file('.0.dlogic') u (exp($1)):2 via aaa,bbb
#    EOF
