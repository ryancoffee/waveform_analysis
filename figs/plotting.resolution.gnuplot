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
# set terminal qt 2 font "Sans,9"
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
set samples 1000, 1000
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
set xlabel "" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 30.0000 : 55.0000 ] noreverse writeback
set x2range [ 29.4443 : 42.5060 ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ 1.00000 : 3000.00 ] noreverse writeback
set y2range [ -10.4987 : 823.858 ] noreverse writeback
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
f(x)= a + b*x + c* x**2 + d* x**3
g(x,a,x0,w)=a*exp(-((x-x0)/w)**2)
GNUTERM = "qt"
enegiesfile = "data_fs/energies.notes"
energiesfile = "data_fs/energies.notes"
x = 0.0
a = -99.8062660724402
b = 72.8783855110483
c = -16.7727586388828
d = 1.26496665767551
FIT_CONVERGED = 1
FIT_NDF = 12
FIT_STDFIT = 0.0012444243467108
FIT_WSSR = 1.85831034562391e-05
FIT_P = 1.0
FIT_NITER = 8
a_err = 10.3057405726716
b_err = 6.79041095657291
c_err = 1.49050794213357
d_err = 0.108991645953526
thist = "data_fs/03_16_2020_CX60_V63+_and_6013E-S+_highcount/C1--ATI_attempt_03_16_2020.thist"
ehist = "data_fs/03_16_2020_CX60_V63+_and_6013E-S+_highcount/C1--ATI_attempt_03_16_2020.ehist"
thist2 = "data_fs/03_16_2020_ZJL4g+_20Vacc/C1--ATI_attempt_20V_03_16_2020.thist"
ehist2 = "data_fs/03_16_2020_ZJL4g+_20Vacc/C1--ATI_attempt_20V_03_16_2020.ehist"
## Last datafile plotted: "data_fs/03_16_2020_CX60_V63+_and_6013E-S+_highcount/C1--ATI_attempt_03_16_2020.ehist"
set xlabel 'energy [eV]'
set ylabel 'counts'
set term png size 1200,600
set output 'figs/plotting.resolution.both.png'
set multiplot
set size 1,.5
set origin 0,.5
set lmargin screen .08
set rmargin screen .97
set xrange [35:55]
plot	ehist lw 1 lc -1 title '30V acceleration ATI spectrum',\
	g(x,170,42.3,.25) lw 2 lc 2 title '{/Symbol s} = .25 eV',\
	g(x,600,39.2,.25) lw 2 lc 2 notitle,\
	g(x,35,48.6,.25) lw 2 lc 2 notitle,\
	g(x,18,48.6+2*1.55,.25) lw 2 lc 2 notitle
set origin 0,0
set xrange [30:70]
set yrange [1:200]
plot	ehist2 lw 1 lc -1 title '20V acceleration ATI spectrum',\
	g(x,100,33.1,.25) lw 2 lc 2 title '{/Symbol s} = .25 eV',\
	g(x,40,45.1,.25) lw 2 lc 2 notitle,\
	g(x,12,55.8,.25) lw 2 lc 2 notitle,\
	g(x,5,66.8,.25) lw 2 lc 2 notitle
unset multiplot
	
## fit f(x) energiesfile u (log($1-50)):(log(1.55*($0+1)+30)) via a,b,c,d
#    EOF
